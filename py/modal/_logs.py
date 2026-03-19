# Copyright Modal Labs 2026
from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

from google.protobuf.timestamp_pb2 import Timestamp

from modal._utils.async_utils import TaskContext
from modal.exception import LogsFetchError
from modal_proto import api_pb2

if TYPE_CHECKING:
    from modal.client import _Client


@dataclasses.dataclass(frozen=True)
class LogsFilters:
    source: "api_pb2.FileDescriptor.ValueType" = api_pb2.FILE_DESCRIPTOR_UNSPECIFIED
    function_id: str = ""
    function_call_id: str = ""
    task_id: str = ""
    sandbox_id: str = ""
    search_text: str = ""


# Maximum number of logs to request per fetch RPC call. This is a safety cap
# to avoid silently dropping logs from dense intervals, not a target size.
# Derived from a ~4 MiB target response size at ~200 bytes per log entry.
_FETCH_LIMIT = 20_000

# Minimum number of logs to accumulate per interval before splitting into
# a new one. Buckets are merged until this threshold is reached. Intervals
# with a single bucket exceeding this value are emitted as-is.
_INTERVAL_LOG_THRESHOLD = 2_000

# Maximum number of fetch RPCs per fetch_logs call. Together with _FETCH_LIMIT,
# this bounds total retrievable entries to _MAX_FETCHES * _FETCH_LIMIT = 10M.
_MAX_FETCHES = 500

# Maximum allowed range between `since` and `until`, enforced server-side.
_MAX_FETCH_RANGE = timedelta(days=35)

# Lookback windows for tailing logs, tried in order from shortest to longest.
_TAIL_LOOKBACKS = [
    timedelta(hours=1),
    timedelta(days=1),
    timedelta(days=7),
    timedelta(days=30),
]

# Maximum number of concurrent AppFetchLogs requests.
_MAX_CONCURRENT_FETCHES = 10

# Maximum number of concurrent AppCountLogs requests (responses are small).
_MAX_CONCURRENT_COUNTS = 20

# Maximum number of refinement iterations (each fires a parallel batch of AppCountLogs RPCs).
_MAX_REFINE_ITERATIONS = 3

# Number of buckets for the initial count
_APPROX_INITIAL_BUCKETS = 100

# Predefined bucket sizes (in seconds)
# We pick the smallest size such that the total number of buckets stays <= _APPROX_INITIAL_BUCKETS.
_BUCKET_SIZES_SECS = [
    2,
    4,
    6,
    12,
    20,
    30,
    60,
    120,
    180,
    240,
    300,
    360,
    600,
    720,
    900,
    1200,
    1800,
    3600,
    7200,
    10800,
    14400,
    28800,
    43200,
    86400,
]


def _datetime_to_timestamp(dt: datetime) -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


def _timestamp_to_seconds(ts: Timestamp) -> float:
    return ts.seconds + ts.nanos / 1e9


def _seconds_to_timestamp(secs: float) -> Timestamp:
    ts = Timestamp()
    ts.seconds = int(secs)
    ts.nanos = int((secs - int(secs)) * 1e9)
    return ts


def _pick_bucket_secs(since: datetime, until: datetime) -> int:
    """Pick the smallest bucket size that yields _APPROX_INITIAL_BUCKETS."""
    duration_secs = (until - since).total_seconds()
    for bucket_secs in _BUCKET_SIZES_SECS:
        if duration_secs / bucket_secs <= _APPROX_INITIAL_BUCKETS:
            return bucket_secs
    return _BUCKET_SIZES_SECS[-1]


def _bucket_count(bucket: api_pb2.AppCountLogsResponse.LogBucket) -> int:
    return bucket.stdout_logs + bucket.stderr_logs + bucket.system_logs


def _buckets_to_ranges(
    buckets: list[api_pb2.AppCountLogsResponse.LogBucket],
    bucket_secs: int,
) -> list[tuple[float, float, int]]:
    """Convert proto buckets to (start, end, count) tuples."""
    return [
        (
            _timestamp_to_seconds(b.bucket_start_at),
            _timestamp_to_seconds(b.bucket_start_at) + bucket_secs,
            _bucket_count(b),
        )
        for b in buckets
    ]


def _build_fetch_intervals(
    ranges: list[tuple[float, float, int]],
) -> list[tuple[float, float]]:
    """Build time intervals for parallel fetching from (start, end, count) ranges.

    Merges adjacent non-empty ranges into intervals, splitting when the
    accumulated log count reaches _INTERVAL_LOG_THRESHOLD or when adding the
    next range would push the interval over _FETCH_LIMIT.
    """
    intervals: list[tuple[float, float]] = []
    current_start: Optional[float] = None
    current_end: float = 0.0
    current_count = 0

    for range_start, range_end, total in ranges:
        if total == 0:
            if current_start is not None:
                intervals.append((current_start, current_end))
                current_start = None
                current_count = 0
            continue

        if current_start is not None and current_count + total > _FETCH_LIMIT:
            intervals.append((current_start, current_end))
            current_start = None
            current_count = 0

        if current_start is None:
            current_start = range_start
            current_count = total
        else:
            current_count += total
        current_end = range_end

        if current_count >= _INTERVAL_LOG_THRESHOLD:
            intervals.append((current_start, current_end))
            current_start = None
            current_count = 0

    # Flush remaining interval
    if current_start is not None:
        intervals.append((current_start, current_end))

    return intervals


def _next_smaller_bucket_secs(current_bucket_secs: int) -> Optional[int]:
    """Return the next smaller bucket size, or None if already at the smallest."""
    for size in reversed(_BUCKET_SIZES_SECS):
        if size < current_bucket_secs:
            return size
    return None


async def _refine_dense_ranges(
    client: _Client,
    app_id: str,
    ranges: list[tuple[float, float, int]],
    filters: LogsFilters,
    max_ranges: int,
    max_iterations: int,
) -> list[tuple[float, float, int]]:
    """Refine ranges exceeding _FETCH_LIMIT by subdividing with smaller buckets.

    Uses breadth-first refinement: each iteration splits ALL over-limit ranges
    one level, ensuring fair treatment across dense regions. Stops when all
    ranges fit within _FETCH_LIMIT, no further subdivision is possible,
    the total range count would exceed max_ranges, or the iteration budget
    is exhausted.
    """
    refined = list(ranges)
    iterations = 0

    while iterations < max_iterations:
        # Identify dense ranges that can be subdivided
        to_refine: list[tuple[int, int]] = []  # (index, smaller_secs)
        for i, (start, end, count) in enumerate(refined):
            if count <= _FETCH_LIMIT:
                continue
            duration = end - start
            smaller_secs = _next_smaller_bucket_secs(int(duration))
            if smaller_secs is not None and smaller_secs < duration:
                to_refine.append((i, smaller_secs))

        if not to_refine:
            break

        # Estimate new range count: each refined range of duration D
        # with bucket size S produces ceil(D/S) sub-ranges, replacing 1.
        estimated_new = len(refined)
        for i, smaller_secs in to_refine:
            start, end, _ = refined[i]
            duration = end - start
            sub_count = -(-int(duration) // smaller_secs)  # ceil division
            estimated_new += sub_count - 1  # replacing 1 range with sub_count
        if estimated_new > max_ranges:
            break

        # Fire off all re-count RPCs in parallel with bounded concurrency
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_COUNTS)

        async def _recount(start: float, end: float, smaller_secs: int) -> list[tuple[float, float, int]]:
            async with semaphore:
                sub_req = api_pb2.AppCountLogsRequest(
                    app_id=app_id,
                    since=_seconds_to_timestamp(start),
                    until=_seconds_to_timestamp(end),
                    bucket_secs=smaller_secs,
                    source=filters.source,
                    function_id=filters.function_id,
                    function_call_id=filters.function_call_id,
                    task_id=filters.task_id,
                    sandbox_id=filters.sandbox_id,
                    search_text=filters.search_text,
                )
                sub_resp = await client.stub.AppCountLogs(sub_req)
                sub_ranges = _buckets_to_ranges(list(sub_resp.buckets), smaller_secs)
                # Clamp the last sub-range to the parent's end. When bucket_secs
                # doesn't evenly divide the parent duration, the last bucket
                # boundary overshoots (e.g. parent (0,30) with 20s buckets
                # produces (20,40)), causing overlaps with sibling ranges.
                if sub_ranges:
                    s, _, c = sub_ranges[-1]
                    sub_ranges[-1] = (s, min(s + smaller_secs, end), c)
                return sub_ranges

        refine_set = {i for i, _ in to_refine}
        tasks = [_recount(refined[i][0], refined[i][1], s) for i, s in to_refine]
        iterations += 1
        sub_results = await TaskContext.gather(*tasks)

        # Rebuild the list, splicing in sub-ranges for refined entries
        result_iter = iter(sub_results)
        new_refined: list[tuple[float, float, int]] = []
        for i, r in enumerate(refined):
            if i in refine_set:
                new_refined.extend(next(result_iter))
            else:
                new_refined.append(r)

        refined = new_refined

    return refined


async def _fetch_interval(
    client: _Client,
    app_id: str,
    since: float,
    until: float,
    limit: int,
    filters: LogsFilters,
) -> list[api_pb2.TaskLogsBatch]:
    """Fetch logs for a single time interval."""
    req = api_pb2.AppFetchLogsRequest(
        app_id=app_id,
        since=_seconds_to_timestamp(since),
        until=_seconds_to_timestamp(until),
        limit=limit,
        source=filters.source,
        function_id=filters.function_id,
        function_call_id=filters.function_call_id,
        task_id=filters.task_id,
        sandbox_id=filters.sandbox_id,
        search_text=filters.search_text,
    )
    resp: api_pb2.AppFetchLogsResponse = await client.stub.AppFetchLogs(req)
    return list(resp.batches)


async def tail_logs(
    client: _Client,
    app_id: str,
    n: int,
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    filters: Optional[LogsFilters] = None,
) -> AsyncGenerator[api_pb2.TaskLogsBatch]:
    """Fetch up to `n` of the most recent log entries for an app.

    When `since` is provided, it is used as a hard floor (no progressive
    widening). When omitted, starts with a short lookback window and widens
    until `n` results are returned or all windows are exhausted. `until`
    defaults to now.

    `n` must not exceed _FETCH_LIMIT.
    """
    if filters is None:
        filters = LogsFilters()
    if n > _FETCH_LIMIT:
        raise LogsFetchError(f"--tail value must not exceed {_FETCH_LIMIT}.")

    now = datetime.now(timezone.utc)

    if since is not None and since.tzinfo is None:
        raise ValueError("since must be timezone-aware")
    if until is not None and until.tzinfo is None:
        raise ValueError("until must be timezone-aware")

    until_ts = _datetime_to_timestamp(until) if until else _datetime_to_timestamp(now)

    if since is not None:
        # Explicit floor — single fetch, no widening.
        effective_until = until or now
        if effective_until - since > _MAX_FETCH_RANGE:
            raise LogsFetchError(f"Time range cannot exceed {_MAX_FETCH_RANGE.days} days.")
        req = api_pb2.AppFetchLogsRequest(
            app_id=app_id,
            since=_datetime_to_timestamp(since),
            until=until_ts,
            limit=n,
            source=filters.source,
            function_id=filters.function_id,
            function_call_id=filters.function_call_id,
            task_id=filters.task_id,
            sandbox_id=filters.sandbox_id,
            search_text=filters.search_text,
        )
        resp = await client.stub.AppFetchLogs(req)
        for batch in resp.batches:
            yield batch
        return

    # No explicit since — progressively widen the lookback.
    anchor = until or now
    for lookback in _TAIL_LOOKBACKS:
        lookback_since = anchor - lookback
        req = api_pb2.AppFetchLogsRequest(
            app_id=app_id,
            since=_datetime_to_timestamp(lookback_since),
            until=until_ts,
            limit=n,
            source=filters.source,
            function_id=filters.function_id,
            function_call_id=filters.function_call_id,
            task_id=filters.task_id,
            sandbox_id=filters.sandbox_id,
            search_text=filters.search_text,
        )
        resp = await client.stub.AppFetchLogs(req)

        total_items = sum(len(b.items) for b in resp.batches)
        if total_items >= n or lookback == _TAIL_LOOKBACKS[-1]:
            for batch in resp.batches:
                yield batch
            return


async def fetch_logs(
    client: _Client,
    app_id: str,
    since: datetime,
    until: datetime,
    *,
    filters: Optional[LogsFilters] = None,
) -> AsyncGenerator[api_pb2.TaskLogsBatch]:
    """Fetch logs for an app over a time range using count-then-fetch strategy.

    Yields TaskLogsBatch objects in chronological order.
    """
    if filters is None:
        filters = LogsFilters()

    if since.tzinfo is None:
        raise ValueError("since must be timezone-aware")
    if until.tzinfo is None:
        raise ValueError("until must be timezone-aware")

    if until - since > _MAX_FETCH_RANGE:
        raise LogsFetchError(f"Time range cannot exceed {_MAX_FETCH_RANGE.days} days.")

    # Phase 1: Count logs per bucket
    bucket_secs = _pick_bucket_secs(since, until)
    count_req = api_pb2.AppCountLogsRequest(
        app_id=app_id,
        since=_datetime_to_timestamp(since),
        until=_datetime_to_timestamp(until),
        bucket_secs=bucket_secs,
        source=filters.source,
        function_id=filters.function_id,
        function_call_id=filters.function_call_id,
        task_id=filters.task_id,
        sandbox_id=filters.sandbox_id,
        search_text=filters.search_text,
    )
    count_resp: api_pb2.AppCountLogsResponse = await client.stub.AppCountLogs(count_req)

    ranges = _buckets_to_ranges(list(count_resp.buckets), bucket_secs)
    total_logs = sum(count for _, _, count in ranges)
    if total_logs == 0:
        return

    # Trim leading/trailing empty buckets so they don't consume refinement
    # budget. Interior zeros are kept to prevent merging across gaps.
    while ranges and ranges[0][2] == 0:
        ranges.pop(0)
    while ranges and ranges[-1][2] == 0:
        ranges.pop()

    # Phase 1b: Refine any ranges that exceed the per-fetch limit.
    # Budget: at most _MAX_FETCHES intervals, each fetching up to _FETCH_LIMIT entries.
    ranges = await _refine_dense_ranges(
        client,
        app_id,
        ranges,
        filters,
        max_ranges=_MAX_FETCHES,
        max_iterations=_MAX_REFINE_ITERATIONS,
    )
    fetch_error_message = "Too many logs to fetch in time range. Consider narrowing the range or adding filters."
    # Check that all ranges fit within the per-fetch limit.
    over_limit = [r for r in ranges if r[2] > _FETCH_LIMIT]
    if over_limit:
        raise LogsFetchError(fetch_error_message)

    # Phase 2: Build intervals from non-empty ranges and fetch in parallel
    since_secs = since.timestamp()
    until_secs = until.timestamp()
    intervals = _build_fetch_intervals(ranges)
    # Clamp intervals to the requested time range. Bucket boundaries may
    # extend beyond the user's since/until (e.g. an 86400s bucket could
    # include up to a full extra day).
    intervals = [(max(s, since_secs), min(u, until_secs)) for s, u in intervals]
    intervals = [(s, u) for s, u in intervals if s < u]
    if not intervals:
        return

    if len(intervals) > _MAX_FETCHES:
        raise LogsFetchError(fetch_error_message)

    # Fetch all intervals concurrently (bounded by semaphore), but yield
    # results in interval order so logs stay chronological. Awaiting each
    # task in sequence lets us stream batches to the caller as soon as all
    # preceding intervals have completed.
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)

    async def _bounded_fetch(since_s: float, until_s: float) -> list[api_pb2.TaskLogsBatch]:
        async with semaphore:
            return await _fetch_interval(
                client,
                app_id,
                since_s,
                until_s,
                _FETCH_LIMIT,
                filters,
            )

    tasks = [asyncio.create_task(_bounded_fetch(s, u)) for s, u in intervals]

    try:
        for task in tasks:
            for batch in await task:
                yield batch
    finally:
        for task in tasks:
            task.cancel()
