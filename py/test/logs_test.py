# Copyright Modal Labs 2026
import pytest
from datetime import datetime, timedelta, timezone
from typing import Any

from modal._logs import (
    _FETCH_LIMIT,
    _MAX_FETCHES,
    LogsFilters,
    _buckets_to_ranges,
    _build_fetch_intervals,
    _next_smaller_bucket_secs,
    _pick_bucket_secs,
    _refine_dense_ranges,
    _seconds_to_timestamp,
    _timestamp_to_seconds,
)
from modal._output.pty import _build_log_prefix
from modal.cli.app import _parse_time_arg
from modal.exception import LogsFetchError
from modal_proto import api_pb2

from .conftest import run_cli_command

# ---------------------------------------------------------------------------
# _pick_bucket_secs
# ---------------------------------------------------------------------------


def test_pick_bucket_secs_small_range():
    since = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    until = since + timedelta(seconds=100)
    # 100s / 2s = 50 buckets <= 100, so smallest bucket size (2) works
    assert _pick_bucket_secs(since, until) == 2


def test_pick_bucket_secs_medium_range():
    since = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    until = since + timedelta(hours=1)
    # 3600s / 60s = 60 buckets <= 100
    bucket = _pick_bucket_secs(since, until)
    assert bucket <= 60
    assert 3600 / bucket <= 100


def test_pick_bucket_secs_large_range():
    since = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    until = since + timedelta(days=30)
    bucket = _pick_bucket_secs(since, until)
    assert 30 * 86400 / bucket <= 100


# ---------------------------------------------------------------------------
# _build_fetch_intervals
# ---------------------------------------------------------------------------


def _make_bucket(start_secs: float, stdout=0, stderr=0, system=0):
    ts = _seconds_to_timestamp(start_secs)
    return api_pb2.AppCountLogsResponse.LogBucket(
        bucket_start_at=ts,
        stdout_logs=stdout,
        stderr_logs=stderr,
        system_logs=system,
    )


def _ranges(buckets, bucket_secs):
    return _buckets_to_ranges(buckets, bucket_secs)


def test_build_fetch_intervals_empty_buckets():
    ranges = _ranges([_make_bucket(100.0), _make_bucket(110.0), _make_bucket(120.0)], 10)
    assert _build_fetch_intervals(ranges) == []


def test_build_fetch_intervals_single_bucket():
    ranges = _ranges([_make_bucket(100.0, stdout=10)], 10)
    intervals = _build_fetch_intervals(ranges)
    assert intervals == [(100.0, 110.0)]


def test_build_fetch_intervals_merges_adjacent():
    ranges = _ranges(
        [
            _make_bucket(100.0, stdout=10),
            _make_bucket(110.0, stdout=10),
            _make_bucket(120.0, stdout=10),
        ],
        10,
    )
    intervals = _build_fetch_intervals(ranges)
    # All under _INTERVAL_LOG_THRESHOLD, so merged into one
    assert intervals == [(100.0, 130.0)]


def test_build_fetch_intervals_splits_at_target():
    from modal._logs import _INTERVAL_LOG_THRESHOLD

    ranges = _ranges(
        [
            _make_bucket(100.0, stdout=_INTERVAL_LOG_THRESHOLD),
            _make_bucket(110.0, stdout=10),
        ],
        10,
    )
    intervals = _build_fetch_intervals(ranges)
    assert len(intervals) == 2
    assert intervals[0] == (100.0, 110.0)
    assert intervals[1] == (110.0, 120.0)


def test_build_fetch_intervals_splits_before_exceeding_fetch_limit():
    """Merging a small range with a _FETCH_LIMIT-sized range must not produce
    an interval that exceeds _FETCH_LIMIT, since excess logs would be silently
    dropped by the server."""
    ranges = _ranges(
        [
            _make_bucket(100.0, stdout=1_000),
            _make_bucket(110.0, stdout=_FETCH_LIMIT),
        ],
        10,
    )
    intervals = _build_fetch_intervals(ranges)
    # Should split into two intervals, not merge into one 21,000-log interval
    assert len(intervals) == 2
    assert intervals[0] == (100.0, 110.0)
    assert intervals[1] == (110.0, 120.0)


def test_build_fetch_intervals_merges_when_under_fetch_limit():
    """Adjacent ranges that together fit within _FETCH_LIMIT can be merged
    (when also under _INTERVAL_LOG_THRESHOLD)."""
    from modal._logs import _INTERVAL_LOG_THRESHOLD

    small = _INTERVAL_LOG_THRESHOLD // 3
    ranges = _ranges(
        [
            _make_bucket(100.0, stdout=small),
            _make_bucket(110.0, stdout=small),
        ],
        10,
    )
    intervals = _build_fetch_intervals(ranges)
    assert len(intervals) == 1
    assert intervals[0] == (100.0, 120.0)


def test_build_fetch_intervals_gaps():
    ranges = _ranges(
        [
            _make_bucket(100.0, stdout=10),
            _make_bucket(110.0),  # empty
            _make_bucket(120.0, stdout=10),
        ],
        10,
    )
    intervals = _build_fetch_intervals(ranges)
    assert len(intervals) == 2
    assert intervals[0] == (100.0, 110.0)
    assert intervals[1] == (120.0, 130.0)


# ---------------------------------------------------------------------------
# _timestamp_to_seconds / _seconds_to_timestamp round-trip
# ---------------------------------------------------------------------------


def test_timestamp_roundtrip():
    original = 1700000000.5
    ts = _seconds_to_timestamp(original)
    result = _timestamp_to_seconds(ts)
    assert abs(result - original) < 1e-6


# ---------------------------------------------------------------------------
# _parse_time_arg
# ---------------------------------------------------------------------------


def test_parse_time_arg_none():
    default = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _parse_time_arg(None, default) == default


def test_parse_time_arg_relative():
    before = datetime.now(timezone.utc)
    result = _parse_time_arg("30m", datetime.min)
    expected = before - timedelta(minutes=30)
    # Should be approximately 30 minutes ago
    assert abs((result - expected).total_seconds()) < 2


def test_parse_time_arg_iso():
    result = _parse_time_arg("2026-01-15T12:00:00", datetime.min)
    # Naive datetimes are interpreted in the user's local timezone
    from modal._utils.time_utils import locale_tz

    assert result == datetime(2026, 1, 15, 12, 0, 0, tzinfo=locale_tz())


def test_parse_time_arg_iso_with_explicit_tz_in_value():
    result = _parse_time_arg("2026-01-15T12:00:00+03:00", datetime.min)
    assert result.utcoffset() == timedelta(hours=3)


def test_parse_time_arg_invalid():
    from click import UsageError

    with pytest.raises(UsageError, match="Invalid time format"):
        _parse_time_arg("not-a-time", datetime.min)


# ---------------------------------------------------------------------------
# _build_log_prefix
# ---------------------------------------------------------------------------


def test_build_log_prefix_empty():
    batch = api_pb2.TaskLogsBatch(function_id="fu-123", task_id="ta-456")
    log = api_pb2.TaskLogs(function_call_id="fc-789")
    assert _build_log_prefix(batch, log, []) == ""


def test_build_log_prefix_single():
    batch = api_pb2.TaskLogsBatch(function_id="fu-123", task_id="ta-456")
    log = api_pb2.TaskLogs(function_call_id="fc-789")
    assert _build_log_prefix(batch, log, ["fu"]) == "fu-123"


def test_build_log_prefix_multiple():
    batch = api_pb2.TaskLogsBatch(function_id="fu-123", task_id="ta-456")
    log = api_pb2.TaskLogs(function_call_id="fc-789")
    assert _build_log_prefix(batch, log, ["fu", "ta"]) == "fu-123 ta-456"


def test_build_log_prefix_respects_order():
    batch = api_pb2.TaskLogsBatch(function_id="fu-123", task_id="ta-456")
    log = api_pb2.TaskLogs(function_call_id="fc-789")
    assert _build_log_prefix(batch, log, ["ta", "fu"]) == "ta-456 fu-123"


def test_build_log_prefix_skips_empty_fields():
    batch = api_pb2.TaskLogsBatch(function_id="fu-123", task_id="")
    log = api_pb2.TaskLogs(function_call_id="fc-789")
    assert _build_log_prefix(batch, log, ["fu", "ta", "fc"]) == "fu-123 fc-789"


def test_build_log_prefix_fc_from_log():
    batch = api_pb2.TaskLogsBatch()
    log = api_pb2.TaskLogs(function_call_id="fc-abc")
    assert _build_log_prefix(batch, log, ["fc"]) == "fc-abc"


# ---------------------------------------------------------------------------
# _next_smaller_bucket_secs
# ---------------------------------------------------------------------------


def test_next_smaller_bucket_secs():
    assert _next_smaller_bucket_secs(60) == 30
    assert _next_smaller_bucket_secs(6) == 4
    assert _next_smaller_bucket_secs(2) is None


def test_next_smaller_bucket_secs_smallest():
    # 2 is the smallest bucket size — no smaller exists
    assert _next_smaller_bucket_secs(2) is None
    assert _next_smaller_bucket_secs(1) is None


# ---------------------------------------------------------------------------
# _refine_dense_ranges
# ---------------------------------------------------------------------------


class _MockStub:
    """Stub that returns pre-configured count responses keyed by (since, until, bucket_secs)."""

    def __init__(self, responses: dict):
        self._responses = responses

    async def AppCountLogs(self, req):
        since = req.since.seconds + req.since.nanos / 1e9
        until = req.until.seconds + req.until.nanos / 1e9
        key = (since, until, req.bucket_secs)
        return self._responses[key]


class _MockClient:
    def __init__(self, responses: dict):
        self.stub = _MockStub(responses)


def _count_response(buckets_data: list[tuple[float, int]], bucket_secs: int):
    """Build an AppCountLogsResponse from [(start_secs, count), ...] pairs."""
    buckets = []
    for start, count in buckets_data:
        buckets.append(
            api_pb2.AppCountLogsResponse.LogBucket(
                bucket_start_at=_seconds_to_timestamp(start),
                stdout_logs=count,
            )
        )
    return api_pb2.AppCountLogsResponse(buckets=buckets)


@pytest.mark.asyncio
async def test_refine_dense_ranges_no_refinement_needed():
    """Ranges under _FETCH_LIMIT should pass through unchanged."""
    ranges = [(0.0, 60.0, 100), (60.0, 120.0, 200)]
    client: Any = _MockClient({})
    result = await _refine_dense_ranges(
        client,
        "app-1",
        ranges,
        LogsFilters(),
        max_ranges=500,
        max_iterations=20,
    )
    assert result == ranges


@pytest.mark.asyncio
async def test_refine_dense_ranges_subdivides():
    """A range exceeding _FETCH_LIMIT should be subdivided into smaller buckets."""
    # One 60s range with more logs than _FETCH_LIMIT
    ranges = [(0.0, 60.0, _FETCH_LIMIT + 1)]

    # When refined with 30s buckets, split into two 30s sub-ranges
    client: Any = _MockClient(
        {
            (0.0, 60.0, 30): _count_response([(0.0, 100), (30.0, 100)], 30),
        }
    )
    result = await _refine_dense_ranges(
        client,
        "app-1",
        ranges,
        LogsFilters(),
        max_ranges=500,
        max_iterations=20,
    )
    assert len(result) == 2
    assert result[0] == (0.0, 30.0, 100)
    assert result[1] == (30.0, 60.0, 100)


@pytest.mark.asyncio
async def test_refine_dense_ranges_budget_limit():
    """Refinement should stop if it would exceed max_ranges."""
    # One 60s range that's dense — would split into two 30s sub-ranges
    ranges = [(0.0, 60.0, _FETCH_LIMIT + 1)]

    client: Any = _MockClient(
        {
            (0.0, 60.0, 30): _count_response([(0.0, 100), (30.0, 100)], 30),
        }
    )
    # max_ranges=1 means we can't add any sub-ranges (would go from 1 to 2)
    result = await _refine_dense_ranges(
        client,
        "app-1",
        ranges,
        LogsFilters(),
        max_ranges=1,
        max_iterations=20,
    )
    # Should return unrefined since splitting would exceed budget
    assert result == ranges


@pytest.mark.asyncio
async def test_refine_dense_ranges_breadth_first():
    """Multiple dense ranges should all be refined in the same iteration."""
    # Two dense 60s ranges
    ranges = [
        (0.0, 60.0, _FETCH_LIMIT + 1),
        (60.0, 120.0, _FETCH_LIMIT + 1),
    ]

    client: Any = _MockClient(
        {
            (0.0, 60.0, 30): _count_response([(0.0, 100), (30.0, 100)], 30),
            (60.0, 120.0, 30): _count_response([(60.0, 100), (90.0, 100)], 30),
        }
    )
    result = await _refine_dense_ranges(
        client,
        "app-1",
        ranges,
        LogsFilters(),
        max_ranges=500,
        max_iterations=20,
    )
    assert len(result) == 4
    assert result[0] == (0.0, 30.0, 100)
    assert result[1] == (30.0, 60.0, 100)
    assert result[2] == (60.0, 90.0, 100)
    assert result[3] == (90.0, 120.0, 100)


@pytest.mark.asyncio
async def test_refine_dense_ranges_recursive():
    """Refinement should recurse when sub-ranges are still dense."""
    # One 60s range that's very dense
    ranges = [(0.0, 60.0, _FETCH_LIMIT + 1)]

    # First refinement: 60s → two 30s, but first sub-range is still dense
    # Second refinement: 30s → two 20s sub-ranges (next smaller from 30 is 20)
    client: Any = _MockClient(
        {
            (0.0, 60.0, 30): _count_response([(0.0, _FETCH_LIMIT + 1), (30.0, 100)], 30),
            (0.0, 30.0, 20): _count_response([(0.0, 100), (20.0, 100)], 20),
        }
    )
    result = await _refine_dense_ranges(
        client,
        "app-1",
        ranges,
        LogsFilters(),
        max_ranges=500,
        max_iterations=20,
    )
    # First 30s got split into two sub-ranges: (0,20) and (20,30) — the last
    # sub-range is clamped to the parent's end of 30, not 20+20=40.
    assert len(result) == 3  # two sub-ranges from first half + one 30s from second half
    assert result[0] == (0.0, 20.0, 100)
    assert result[1] == (20.0, 30.0, 100)  # clamped to parent end
    assert result[2] == (30.0, 60.0, 100)  # second half unchanged


@pytest.mark.asyncio
async def test_refine_clamps_sub_ranges_to_parent_boundary():
    ranges = [
        (0.0, 30.0, _FETCH_LIMIT + 1),
        (30.0, 60.0, 100),
    ]
    client: Any = _MockClient(
        {
            (0.0, 30.0, 20): _count_response([(0.0, 5000), (20.0, 5000)], 20),
        }
    )
    result = await _refine_dense_ranges(
        client,
        "app-1",
        ranges,
        LogsFilters(),
        max_ranges=500,
        max_iterations=20,
    )
    assert result[0] == (0.0, 20.0, 5000)
    assert result[1] == (20.0, 30.0, 5000)  # clamped, not (20, 40)
    assert result[2] == (30.0, 60.0, 100)


@pytest.mark.asyncio
async def test_refine_stops_at_smallest_bucket():
    """Refinement should stop when ranges are already at the smallest bucket size (2s)."""
    # A 2s range that's dense — can't subdivide further
    ranges = [(0.0, 2.0, _FETCH_LIMIT + 1)]
    client: Any = _MockClient({})
    result = await _refine_dense_ranges(
        client,
        "app-1",
        ranges,
        LogsFilters(),
        max_ranges=500,
        max_iterations=20,
    )
    # Should return unchanged — no smaller bucket available
    assert result == ranges


# ---------------------------------------------------------------------------
# fetch_logs error conditions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_logs_errors_on_too_dense_interval():
    """fetch_logs should raise ValueError if a range exceeds _FETCH_LIMIT after refinement."""
    from modal._logs import fetch_logs

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=1)
    until = since + timedelta(seconds=4)

    # Single 2s bucket with way more than _FETCH_LIMIT logs — can't refine further
    count_resp = _count_response([(since.timestamp(), _FETCH_LIMIT + 1)], 2)

    client: Any = _MockClient({})
    # Override AppCountLogs to return our dense response
    original = client.stub.AppCountLogs

    async def patched_count(req):
        return count_resp

    client.stub.AppCountLogs = patched_count

    with pytest.raises(LogsFetchError, match="Too many logs to fetch"):
        async for _ in fetch_logs(client, "app-1", since, until):
            pass


@pytest.mark.asyncio
async def test_fetch_logs_errors_on_too_many_intervals():
    """fetch_logs should raise ValueError if intervals exceed _MAX_FETCHES."""
    from modal._logs import _INTERVAL_LOG_THRESHOLD, fetch_logs

    n_buckets = _MAX_FETCHES + 1
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=1)
    # Use a range that yields bucket_secs=2 (needs duration / 2 <= 100)
    until = since + timedelta(seconds=2 * n_buckets)

    # Each bucket has enough logs to prevent merging
    buckets = []
    since_epoch = since.timestamp()
    for i in range(n_buckets):
        start = since_epoch + i * 2.0
        buckets.append((start, _INTERVAL_LOG_THRESHOLD + 1))

    count_resp = _count_response(buckets, 2)

    client: Any = _MockClient({})

    async def patched_count(req):
        return count_resp

    client.stub.AppCountLogs = patched_count

    with pytest.raises(LogsFetchError, match="Too many logs to fetch"):
        async for _ in fetch_logs(client, "app-1", since, until):
            pass


@pytest.mark.asyncio
async def test_fetch_logs_trims_leading_trailing_zeros():
    """fetch_logs should trim empty buckets at the edges, giving refinement more budget."""
    from modal._logs import fetch_logs

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=4)
    until = now

    # 5 buckets: two empty at the start, one dense in the middle, two empty at the end.
    # The dense bucket has count > _FETCH_LIMIT so it needs refinement.
    since_epoch = since.timestamp()
    bucket_secs = 3600  # 1h buckets for a 4h range (fits _APPROX_INITIAL_BUCKETS)
    buckets = [
        (since_epoch, 0),
        (since_epoch + 3600, 0),
        (since_epoch + 7200, 100),
        (since_epoch + 10800, 0),
    ]
    count_resp = _count_response(buckets, bucket_secs)

    fetch_log = api_pb2.TaskLogs(data="hello\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
    fetch_resp = api_pb2.AppFetchLogsResponse(batches=[api_pb2.TaskLogsBatch(items=[fetch_log])])

    count_calls = []
    fetch_calls = []

    class _TrimMockStub:
        async def AppCountLogs(self, req):
            count_calls.append(req)
            return count_resp

        async def AppFetchLogs(self, req):
            fetch_calls.append(req)
            return fetch_resp

    class _TrimMockClient:
        stub = _TrimMockStub()

    client: Any = _TrimMockClient()
    batches = [b async for b in fetch_logs(client, "app-1", since, until)]

    # Should have fetched successfully
    assert len(batches) == 1
    assert batches[0].items[0].data == "hello\n"
    # Only one fetch call — the dense middle bucket
    assert len(fetch_calls) == 1
    # The fetch should be scoped to the non-empty region, not the full 4h range
    fetch_since = fetch_calls[0].since.seconds + fetch_calls[0].since.nanos / 1e9
    fetch_until = fetch_calls[0].until.seconds + fetch_calls[0].until.nanos / 1e9
    assert fetch_since >= since_epoch + 7200 - 1  # starts at or near the active bucket
    assert fetch_until <= since_epoch + 10800 + 1  # ends at or near the active bucket


# ---------------------------------------------------------------------------
# tail_logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tail_logs_single_rpc():
    """tail_logs should issue a single AppFetchLogs when the first lookback returns enough rows."""
    from modal._logs import tail_logs

    fetch_requests = []
    fetch_resp = api_pb2.AppFetchLogsResponse(
        batches=[
            api_pb2.TaskLogsBatch(
                items=[
                    api_pb2.TaskLogs(data=f"line{i}\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
                    for i in range(50)
                ]
            )
        ]
    )

    class _TailMockStub:
        async def AppFetchLogs(self, req):
            fetch_requests.append(req)
            return fetch_resp

    class _TailMockClient:
        stub = _TailMockStub()

    client: Any = _TailMockClient()
    batches = [b async for b in tail_logs(client, "app-1", 50)]

    # Enough rows returned on first try — only one RPC
    assert len(fetch_requests) == 1
    req = fetch_requests[0]
    assert req.limit == 50
    assert req.HasField("since")  # bounded lookback
    assert req.HasField("until")  # defaults to now
    # Batches passed through
    assert len(batches) == 1
    assert len(batches[0].items) == 50


@pytest.mark.asyncio
async def test_tail_logs_widens_lookback():
    """tail_logs should widen the lookback window when the first attempt returns too few rows."""
    from modal._logs import tail_logs

    call_count = 0
    partial_resp = api_pb2.AppFetchLogsResponse(
        batches=[
            api_pb2.TaskLogsBatch(items=[api_pb2.TaskLogs(data="x\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)])
        ]
    )
    full_resp = api_pb2.AppFetchLogsResponse(
        batches=[
            api_pb2.TaskLogsBatch(
                items=[
                    api_pb2.TaskLogs(data=f"line{i}\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
                    for i in range(10)
                ]
            )
        ]
    )

    class _TailMockStub:
        async def AppFetchLogs(self, req):
            nonlocal call_count
            call_count += 1
            # First call returns too few, second returns enough
            return partial_resp if call_count == 1 else full_resp

    class _TailMockClient:
        stub = _TailMockStub()

    client: Any = _TailMockClient()
    batches = [b async for b in tail_logs(client, "app-1", 10)]

    assert call_count == 2
    assert len(batches) == 1
    assert len(batches[0].items) == 10


@pytest.mark.asyncio
async def test_tail_logs_with_explicit_since():
    """tail_logs with explicit since should issue a single fetch (no widening)."""
    from modal._logs import tail_logs

    fetch_requests = []
    fetch_resp = api_pb2.AppFetchLogsResponse(
        batches=[
            api_pb2.TaskLogsBatch(items=[api_pb2.TaskLogs(data="x\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)])
        ]
    )

    class _TailMockStub:
        async def AppFetchLogs(self, req):
            fetch_requests.append(req)
            return fetch_resp

    class _TailMockClient:
        stub = _TailMockStub()

    client: Any = _TailMockClient()
    since = datetime.now(timezone.utc) - timedelta(hours=2)
    # Only 1 result even though n=50 — no widening because since is explicit
    batches = [b async for b in tail_logs(client, "app-1", 50, since=since)]

    assert len(fetch_requests) == 1
    assert fetch_requests[0].HasField("since")
    assert len(batches) == 1
    assert len(batches[0].items) == 1


@pytest.mark.asyncio
async def test_tail_logs_with_explicit_until():
    """tail_logs with explicit until should pass it through."""
    from modal._logs import tail_logs

    fetch_requests = []
    fetch_resp = api_pb2.AppFetchLogsResponse(
        batches=[
            api_pb2.TaskLogsBatch(
                items=[
                    api_pb2.TaskLogs(data=f"line{i}\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
                    for i in range(10)
                ]
            )
        ]
    )

    class _TailMockStub:
        async def AppFetchLogs(self, req):
            fetch_requests.append(req)
            return fetch_resp

    class _TailMockClient:
        stub = _TailMockStub()

    client: Any = _TailMockClient()
    until = datetime.now(timezone.utc) - timedelta(hours=1)
    batches = [b async for b in tail_logs(client, "app-1", 10, until=until)]

    assert len(fetch_requests) == 1
    assert fetch_requests[0].HasField("until")


@pytest.mark.asyncio
async def test_tail_logs_exceeds_limit():
    """tail_logs should raise LogsFetchError if n > _FETCH_LIMIT."""
    from modal._logs import tail_logs

    client: Any = _MockClient({})
    with pytest.raises(LogsFetchError, match="must not exceed"):
        async for _ in tail_logs(client, "app-1", _FETCH_LIMIT + 1):
            pass


@pytest.mark.asyncio
async def test_fetch_logs_errors_on_range_exceeding_max():
    """fetch_logs should raise LogsFetchError if until - since > _MAX_FETCH_RANGE."""
    from modal._logs import _MAX_FETCH_RANGE, fetch_logs

    now = datetime.now(timezone.utc)
    since = now - _MAX_FETCH_RANGE - timedelta(days=1)
    until = now

    client: Any = _MockClient({})
    with pytest.raises(LogsFetchError, match="Time range cannot exceed"):
        async for _ in fetch_logs(client, "app-1", since, until):
            pass


@pytest.mark.asyncio
async def test_tail_logs_errors_on_range_exceeding_max():
    """tail_logs with explicit since should raise LogsFetchError if the range exceeds _MAX_FETCH_RANGE."""
    from modal._logs import _MAX_FETCH_RANGE, tail_logs

    now = datetime.now(timezone.utc)
    since = now - _MAX_FETCH_RANGE - timedelta(days=1)

    client: Any = _MockClient({})
    with pytest.raises(LogsFetchError, match="Time range cannot exceed"):
        async for _ in tail_logs(client, "app-1", 10, since=since):
            pass


# ===========================================================================
# CLI integration tests (require servicer fixtures)
# ===========================================================================

dummy_app_file = """
import modal
app = modal.App("my_app")
"""

# Use recent timestamps for realistic test data
_now = datetime.now(timezone.utc)
_TEST_TIMESTAMP = (_now - timedelta(hours=1)).timestamp()
_BUCKET_START = _TEST_TIMESTAMP - 60  # Bucket starts 60s before the log
# CLI args that bracket the test data
_SINCE_ARG = (_now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
_UNTIL_ARG = _now.strftime("%Y-%m-%dT%H:%M:%S")


def _make_count_response():
    ts = _seconds_to_timestamp(_BUCKET_START)
    bucket = api_pb2.AppCountLogsResponse.LogBucket(
        bucket_start_at=ts,
        stdout_logs=1,
    )
    return api_pb2.AppCountLogsResponse(buckets=[bucket])


def _make_fetch_response(data="hello\n", function_id="fu-test1", task_id="ta-test1", function_call_id=""):
    log = api_pb2.TaskLogs(
        data=data,
        file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
        timestamp=_TEST_TIMESTAMP,
        function_call_id=function_call_id,
    )
    batch = api_pb2.TaskLogsBatch(
        items=[log],
        function_id=function_id,
        task_id=task_id,
    )
    return api_pb2.AppFetchLogsResponse(batches=[batch])


def _make_count_responder(count_resp=None):
    if count_resp is None:
        count_resp = _make_count_response()

    async def handler(self, stream):
        await stream.recv_message()
        await stream.send_message(count_resp)

    return handler


def _make_fetch_responder(fetch_resp=None):
    if fetch_resp is None:
        fetch_resp = _make_fetch_response()

    async def handler(self, stream):
        await stream.recv_message()
        await stream.send_message(fetch_resp)

    return handler


def _setup_fetch_responders(ctx, count_resp=None, fetch_resp=None):
    ctx.set_responder("AppCountLogs", _make_count_responder(count_resp))
    ctx.set_responder("AppFetchLogs", _make_fetch_responder(fetch_resp))


def _deploy_app(mock_dir):
    with mock_dir({"myapp.py": dummy_app_file}):
        run_cli_command(["deploy", "myapp.py", "--name", "my-app"])


def test_logs_fetch(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    with servicer.intercept() as ctx:
        _setup_fetch_responders(ctx)

        res = run_cli_command(["app", "logs", "my-app", "--since", _SINCE_ARG, "--until", _UNTIL_ARG])
        assert "hello" in res.stdout


def test_logs_fetch_with_timestamps(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    with servicer.intercept() as ctx:
        _setup_fetch_responders(ctx)

        res = run_cli_command(["app", "logs", "my-app", "--since", _SINCE_ARG, "--until", _UNTIL_ARG, "--timestamps"])
        # Timestamp is displayed in the user's local timezone
        from modal._utils.time_utils import locale_tz

        expected_ts = datetime.fromtimestamp(_TEST_TIMESTAMP, tz=locale_tz()).strftime("%Y-%m-%d %H:%M:%S")
        assert expected_ts in res.stdout
        assert "hello" in res.stdout


def test_logs_fetch_with_prefix(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    with servicer.intercept() as ctx:
        _setup_fetch_responders(ctx)

        res = run_cli_command(
            ["app", "logs", "my-app", "--since", _SINCE_ARG, "--until", _UNTIL_ARG, "--show-function-id"]
        )
        assert "fu-test1" in res.stdout
        assert "hello" in res.stdout


def test_logs_fetch_with_prefix_order(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    with servicer.intercept() as ctx:
        _setup_fetch_responders(ctx)

        args = [
            "app",
            "logs",
            "my-app",
            "--since",
            _SINCE_ARG,
            "--until",
            _UNTIL_ARG,
            "--show-container-id",
            "--show-function-id",
        ]
        res = run_cli_command(args)
        # fu should come before ta in the output (fixed order: fu, fc, ta)
        line = res.stdout.strip()
        fu_pos = line.find("fu-test1")
        ta_pos = line.find("ta-test1")
        assert fu_pos >= 0 and ta_pos >= 0, f"Expected both prefixes in output: {line!r}"
        assert fu_pos < ta_pos


def test_logs_fetch_source_filter(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    with servicer.intercept() as ctx:
        count_requests = []

        async def count_handler(self, stream):
            req = await stream.recv_message()
            count_requests.append(req)
            await stream.send_message(_make_count_response())

        async def fetch_handler(self, stream):
            await stream.recv_message()
            await stream.send_message(_make_fetch_response())

        ctx.set_responder("AppCountLogs", count_handler)
        ctx.set_responder("AppFetchLogs", fetch_handler)

        run_cli_command(["app", "logs", "my-app", "--since", _SINCE_ARG, "--until", _UNTIL_ARG, "--source", "stderr"])
        assert len(count_requests) == 1
        assert count_requests[0].source == api_pb2.FILE_DESCRIPTOR_STDERR


def test_logs_fetch_function_filter(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    with servicer.intercept() as ctx:
        count_requests = []

        async def count_handler(self, stream):
            req = await stream.recv_message()
            count_requests.append(req)
            await stream.send_message(_make_count_response())

        async def fetch_handler(self, stream):
            await stream.recv_message()
            await stream.send_message(_make_fetch_response())

        ctx.set_responder("AppCountLogs", count_handler)
        ctx.set_responder("AppFetchLogs", fetch_handler)

        args = ["app", "logs", "my-app", "--since", _SINCE_ARG, "--until", _UNTIL_ARG, "--function", "fu-myfunction"]
        run_cli_command(args)
        assert len(count_requests) == 1
        assert count_requests[0].function_id == "fu-myfunction"


def test_logs_fetch_search(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    fetch_resp = _make_fetch_response(data="matched line")

    with servicer.intercept() as ctx:
        count_requests = []

        async def count_handler(self, stream):
            req = await stream.recv_message()
            count_requests.append(req)
            await stream.send_message(_make_count_response())

        async def fetch_handler(self, stream):
            await stream.recv_message()
            await stream.send_message(fetch_resp)

        ctx.set_responder("AppCountLogs", count_handler)
        ctx.set_responder("AppFetchLogs", fetch_handler)

        args = ["app", "logs", "my-app", "--since", _SINCE_ARG, "--until", _UNTIL_ARG, "--search", "matched"]
        res = run_cli_command(args)
        assert count_requests[0].search_text == "matched"
        assert "matched line" in res.stdout


def test_logs_invalid_source(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    res = run_cli_command(
        ["app", "logs", "my-app", "--source", "invalid"],
        expected_exit_code=2,
    )
    assert "Invalid source" in res.stderr


def test_logs_empty_count(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    empty_count = api_pb2.AppCountLogsResponse(buckets=[])

    with servicer.intercept() as ctx:
        ctx.set_responder("AppCountLogs", _make_count_responder(empty_count))

        res = run_cli_command(["app", "logs", "my-app", "--since", _SINCE_ARG, "--until", _UNTIL_ARG])
        assert res.stdout.strip() == ""


def test_logs_tail_uses_single_fetch(servicer, server_url_env, set_env_client, mock_dir):
    """--tail without --since/--until should use a single AppFetchLogs RPC (no count phase)."""
    _deploy_app(mock_dir)

    fetch_requests = []

    def _make_multi_fetch_response(n):
        logs = [api_pb2.TaskLogs(data=f"line-{i}\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT) for i in range(n)]
        batch = api_pb2.TaskLogsBatch(items=logs, function_id="fu-test1", task_id="ta-test1")
        return api_pb2.AppFetchLogsResponse(batches=[batch])

    with servicer.intercept() as ctx:

        async def fetch_handler(self, stream):
            req = await stream.recv_message()
            fetch_requests.append(req)
            await stream.send_message(_make_multi_fetch_response(5))

        ctx.set_responder("AppFetchLogs", fetch_handler)

        args = ["app", "logs", "my-app", "--tail", "5"]
        res = run_cli_command(args)
        lines = [line for line in res.stdout.strip().split("\n") if line]
        assert len(lines) == 5
        # Enough rows on first try — only one fetch RPC
        assert len(fetch_requests) == 1
        assert fetch_requests[0].limit == 5
        assert fetch_requests[0].HasField("since")  # bounded lookback
        assert fetch_requests[0].HasField("until")  # defaults to now


# ---------------------------------------------------------------------------
# Follow mode filter tests
# ---------------------------------------------------------------------------


def test_logs_follow_source_filter(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    get_logs_requests = []

    async def capture_get_logs(self, stream):
        req = await stream.recv_message()
        get_logs_requests.append(req)
        await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("AppGetLogs", capture_get_logs)

        run_cli_command(["app", "logs", "my-app", "-f", "--source", "stderr"])
        assert len(get_logs_requests) == 1
        assert get_logs_requests[0].file_descriptor == api_pb2.FILE_DESCRIPTOR_STDERR


def test_logs_follow_function_filter(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    get_logs_requests = []

    async def capture_get_logs(self, stream):
        req = await stream.recv_message()
        get_logs_requests.append(req)
        await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("AppGetLogs", capture_get_logs)

        run_cli_command(["app", "logs", "my-app", "-f", "--function", "fu-abc123"])
        assert len(get_logs_requests) == 1
        assert get_logs_requests[0].function_id == "fu-abc123"


def test_logs_follow_container_filter(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    get_logs_requests = []

    async def capture_get_logs(self, stream):
        req = await stream.recv_message()
        get_logs_requests.append(req)
        await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("AppGetLogs", capture_get_logs)

        run_cli_command(["app", "logs", "my-app", "-f", "--container", "ta-abc123"])
        assert len(get_logs_requests) == 1
        assert get_logs_requests[0].task_id == "ta-abc123"


def test_logs_fetch_container_filter(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    with servicer.intercept() as ctx:
        count_requests = []

        async def count_handler(self, stream):
            req = await stream.recv_message()
            count_requests.append(req)
            await stream.send_message(_make_count_response())

        async def fetch_handler(self, stream):
            await stream.recv_message()
            await stream.send_message(_make_fetch_response())

        ctx.set_responder("AppCountLogs", count_handler)
        ctx.set_responder("AppFetchLogs", fetch_handler)

        args = ["app", "logs", "my-app", "--since", _SINCE_ARG, "--until", _UNTIL_ARG, "--container", "ta-abc123"]
        run_cli_command(args)
        assert len(count_requests) == 1
        assert count_requests[0].task_id == "ta-abc123"


def test_logs_follow_search_filter(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    async def get_logs_with_data(self, stream):
        await stream.recv_message()
        log_match = api_pb2.TaskLogs(data="matched line\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
        log_no_match = api_pb2.TaskLogs(data="other line\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="1", items=[log_match, log_no_match]))
        await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("AppGetLogs", get_logs_with_data)

        res = run_cli_command(["app", "logs", "my-app", "-f", "--search", "matched"])
        assert "matched line" in res.stdout
        assert "other line" not in res.stdout


def test_logs_range_exceeds_max_error(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    since = (_now - timedelta(days=36)).strftime("%Y-%m-%dT%H:%M:%S")
    until = _now.strftime("%Y-%m-%dT%H:%M:%S")

    res = run_cli_command(
        ["app", "logs", "my-app", "--since", since, "--until", until],
        expected_exit_code=2,
    )
    assert "35 days" in res.stderr


def test_logs_tail_with_since(servicer, server_url_env, set_env_client, mock_dir):
    """--tail with --since should use tail mode with the since bound."""
    _deploy_app(mock_dir)

    fetch_requests = []

    with servicer.intercept() as ctx:

        async def fetch_handler(self, stream):
            req = await stream.recv_message()
            fetch_requests.append(req)
            await stream.send_message(_make_fetch_response())

        ctx.set_responder("AppFetchLogs", fetch_handler)

        args = ["app", "logs", "my-app", "--tail", "50", "--since", _SINCE_ARG]
        res = run_cli_command(args)
        assert "hello" in res.stdout
        assert len(fetch_requests) == 1
        assert fetch_requests[0].limit == 50
        assert fetch_requests[0].HasField("since")


def test_logs_tail_with_until(servicer, server_url_env, set_env_client, mock_dir):
    """--tail with --until should use tail mode with the until bound."""
    _deploy_app(mock_dir)

    fetch_requests = []

    with servicer.intercept() as ctx:

        async def fetch_handler(self, stream):
            req = await stream.recv_message()
            fetch_requests.append(req)
            await stream.send_message(_make_fetch_response())

        ctx.set_responder("AppFetchLogs", fetch_handler)

        args = ["app", "logs", "my-app", "--tail", "1", "--until", _UNTIL_ARG]
        res = run_cli_command(args)
        assert "hello" in res.stdout
        assert len(fetch_requests) == 1
        assert fetch_requests[0].HasField("until")


def test_logs_until_only(servicer, server_url_env, set_env_client, mock_dir):
    """--until without --since should use tail mode (default 100) anchored at until."""
    _deploy_app(mock_dir)

    fetch_requests = []
    # Return 100 items so widening doesn't kick in
    logs = [
        api_pb2.TaskLogs(data=f"line{i}\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT, timestamp=_TEST_TIMESTAMP)
        for i in range(100)
    ]
    batch = api_pb2.TaskLogsBatch(items=logs, function_id="fu-test1", task_id="ta-test1")
    full_resp = api_pb2.AppFetchLogsResponse(batches=[batch])

    with servicer.intercept() as ctx:

        async def fetch_handler(self, stream):
            req = await stream.recv_message()
            fetch_requests.append(req)
            await stream.send_message(full_resp)

        ctx.set_responder("AppFetchLogs", fetch_handler)

        args = ["app", "logs", "my-app", "--until", _UNTIL_ARG]
        res = run_cli_command(args)
        assert "line0" in res.stdout
        assert len(fetch_requests) == 1
        assert fetch_requests[0].HasField("until")
        assert fetch_requests[0].limit == 100  # _DEFAULT_TAIL


def test_logs_since_only(servicer, server_url_env, set_env_client, mock_dir):
    """--since without --tail should use range mode (count-then-fetch)."""
    _deploy_app(mock_dir)

    count_requests = []

    with servicer.intercept() as ctx:
        _setup_fetch_responders(ctx)

        async def count_handler(self, stream):
            req = await stream.recv_message()
            count_requests.append(req)
            await stream.send_message(_make_count_response())

        ctx.set_responder("AppCountLogs", count_handler)

        args = ["app", "logs", "my-app", "--since", _SINCE_ARG]
        res = run_cli_command(args)
        assert "hello" in res.stdout
        # Range mode should have called AppCountLogs
        assert len(count_requests) == 1


def test_logs_tail_with_since_and_until(servicer, server_url_env, set_env_client, mock_dir):
    """--tail with both --since and --until should use tail mode with both bounds."""
    _deploy_app(mock_dir)

    fetch_requests = []

    with servicer.intercept() as ctx:

        async def fetch_handler(self, stream):
            req = await stream.recv_message()
            fetch_requests.append(req)
            await stream.send_message(_make_fetch_response())

        ctx.set_responder("AppFetchLogs", fetch_handler)

        args = ["app", "logs", "my-app", "--tail", "1", "--since", _SINCE_ARG, "--until", _UNTIL_ARG]
        res = run_cli_command(args)
        assert "hello" in res.stdout
        assert len(fetch_requests) == 1
        assert fetch_requests[0].limit == 1
        assert fetch_requests[0].HasField("since")
        assert fetch_requests[0].HasField("until")


def test_logs_tail_exceeds_limit(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    res = run_cli_command(
        ["app", "logs", "my-app", "--tail", str(_FETCH_LIMIT + 1)],
        expected_exit_code=2,
    )
    assert "must not exceed" in res.stderr


def test_logs_follow_incompatible_with_range(servicer, server_url_env, set_env_client, mock_dir):
    _deploy_app(mock_dir)

    for extra_args in [["--since", "1h"], ["--until", "1h"], ["--tail", "10"]]:
        res = run_cli_command(
            ["app", "logs", "my-app", "-f", *extra_args],
            expected_exit_code=2,
        )
        assert "cannot be combined" in res.stderr


# ===========================================================================
# `modal container logs` tests
# ===========================================================================


def _make_task_get_info_responder(app_id="ap-test123", started_at=_TEST_TIMESTAMP - 3600, finished_at=_TEST_TIMESTAMP):
    async def handler(self, stream):
        req = await stream.recv_message()
        await stream.send_message(
            api_pb2.TaskGetInfoResponse(
                app_id=app_id,
                info=api_pb2.TaskInfo(
                    id=req.task_id,
                    started_at=started_at,
                    finished_at=finished_at,
                ),
            )
        )

    return handler


def _setup_container_log_responders(ctx, count_resp=None, fetch_resp=None, task_info_resp=None):
    ctx.set_responder("TaskGetInfo", task_info_resp or _make_task_get_info_responder())
    _setup_fetch_responders(ctx, count_resp=count_resp, fetch_resp=fetch_resp)


def _make_multi_line_fetch_response(n, task_id="ta-test1"):
    logs = [
        api_pb2.TaskLogs(data=f"line-{i}\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT, timestamp=_TEST_TIMESTAMP)
        for i in range(n)
    ]
    batch = api_pb2.TaskLogsBatch(items=logs, function_id="fu-test1", task_id=task_id)
    return api_pb2.AppFetchLogsResponse(batches=[batch])


def test_container_logs_default_tail(servicer, server_url_env, set_env_client):
    """Default `container logs ta-xxx` fetches last 100 entries (tail mode)."""
    fetch_requests = []

    async def fetch_handler(self, stream):
        req = await stream.recv_message()
        fetch_requests.append(req)
        await stream.send_message(_make_multi_line_fetch_response(100))

    with servicer.intercept() as ctx:
        ctx.set_responder("TaskGetInfo", _make_task_get_info_responder())
        ctx.set_responder("AppFetchLogs", fetch_handler)

        res = run_cli_command(["container", "logs", "ta-test1"])
        assert "line-0" in res.stdout
        assert len(fetch_requests) == 1
        assert fetch_requests[0].limit == 100
        assert fetch_requests[0].task_id == "ta-test1"


def test_container_logs_tail(servicer, server_url_env, set_env_client):
    """--tail N fetches the last N entries."""
    fetch_requests = []

    async def fetch_handler(self, stream):
        req = await stream.recv_message()
        fetch_requests.append(req)
        await stream.send_message(_make_multi_line_fetch_response(25))

    with servicer.intercept() as ctx:
        ctx.set_responder("TaskGetInfo", _make_task_get_info_responder())
        ctx.set_responder("AppFetchLogs", fetch_handler)

        res = run_cli_command(["container", "logs", "ta-test1", "--tail", "25"])
        assert "line-0" in res.stdout
        assert len(fetch_requests) == 1
        assert fetch_requests[0].limit == 25


def test_container_logs_since(servicer, server_url_env, set_env_client):
    """--since without --tail uses range mode (count-then-fetch)."""
    with servicer.intercept() as ctx:
        count_requests = []

        async def count_handler(self, stream):
            req = await stream.recv_message()
            count_requests.append(req)
            await stream.send_message(_make_count_response())

        async def fetch_handler(self, stream):
            await stream.recv_message()
            await stream.send_message(_make_fetch_response())

        ctx.set_responder("TaskGetInfo", _make_task_get_info_responder())
        ctx.set_responder("AppCountLogs", count_handler)
        ctx.set_responder("AppFetchLogs", fetch_handler)

        res = run_cli_command(["container", "logs", "ta-test1", "--since", _SINCE_ARG, "--until", _UNTIL_ARG])
        assert "hello" in res.stdout
        assert len(count_requests) == 1
        assert count_requests[0].task_id == "ta-test1"


def test_container_logs_all(servicer, server_url_env, set_env_client):
    """--all fetches range from started_at to finished_at via TaskGetInfo."""
    with servicer.intercept() as ctx:
        count_requests = []

        async def count_handler(self, stream):
            req = await stream.recv_message()
            count_requests.append(req)
            await stream.send_message(_make_count_response())

        async def fetch_handler(self, stream):
            await stream.recv_message()
            await stream.send_message(_make_fetch_response())

        ctx.set_responder("TaskGetInfo", _make_task_get_info_responder())
        ctx.set_responder("AppCountLogs", count_handler)
        ctx.set_responder("AppFetchLogs", fetch_handler)

        res = run_cli_command(["container", "logs", "ta-test1", "--all"])
        assert "hello" in res.stdout
        # --all uses range mode (count-then-fetch)
        assert len(count_requests) == 1


def test_container_logs_follow(servicer, server_url_env, set_env_client):
    """--follow streams logs via AppGetLogs."""
    get_logs_requests = []

    async def capture_get_logs(self, stream):
        req = await stream.recv_message()
        get_logs_requests.append(req)
        await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("AppGetLogs", capture_get_logs)

        run_cli_command(["container", "logs", "ta-test1", "-f"])
        assert len(get_logs_requests) == 1
        assert get_logs_requests[0].task_id == "ta-test1"


def test_container_logs_follow_incompatible_with_range(servicer, server_url_env, set_env_client):
    for extra_args in [["--since", "1h"], ["--until", "1h"], ["--tail", "10"]]:
        res = run_cli_command(
            ["container", "logs", "ta-test1", "-f", *extra_args],
            expected_exit_code=2,
        )
        assert "cannot be combined" in res.stderr


def test_container_logs_all_incompatible_with_flags(servicer, server_url_env, set_env_client):
    for extra_args in [["--since", "1h"], ["--until", "1h"], ["--tail", "10"]]:
        res = run_cli_command(
            ["container", "logs", "ta-test1", "--all", *extra_args],
            expected_exit_code=2,
        )
        assert "cannot be combined" in res.stderr


def test_container_logs_all_incompatible_with_follow(servicer, server_url_env, set_env_client):
    res = run_cli_command(
        ["container", "logs", "ta-test1", "--all", "-f"],
        expected_exit_code=2,
    )
    assert "cannot be combined" in res.stderr


def test_container_logs_invalid_id(servicer, server_url_env, set_env_client):
    res = run_cli_command(
        ["container", "logs", "invalid-id"],
        expected_exit_code=1,
        expected_error="Invalid container ID",
    )


def test_container_logs_source_filter(servicer, server_url_env, set_env_client):
    fetch_requests = []

    async def fetch_handler(self, stream):
        req = await stream.recv_message()
        fetch_requests.append(req)
        await stream.send_message(_make_multi_line_fetch_response(100))

    with servicer.intercept() as ctx:
        ctx.set_responder("TaskGetInfo", _make_task_get_info_responder())
        ctx.set_responder("AppFetchLogs", fetch_handler)

        run_cli_command(["container", "logs", "ta-test1", "--source", "stderr"])
        assert len(fetch_requests) == 1
        assert fetch_requests[0].source == api_pb2.FILE_DESCRIPTOR_STDERR


def test_container_logs_sandbox_id(servicer, server_url_env, set_env_client):
    """Passing sb- ID resolves task_id via SandboxGetTaskId, then fetches logs."""
    sandbox_get_task_requests = []
    task_get_info_requests = []
    fetch_requests = []

    async def sandbox_get_task_handler(self, stream):
        req = await stream.recv_message()
        sandbox_get_task_requests.append(req)
        await stream.send_message(api_pb2.SandboxGetTaskIdResponse(task_id="ta-from-sandbox"))

    async def task_get_info_handler(self, stream):
        req = await stream.recv_message()
        task_get_info_requests.append(req)
        await stream.send_message(
            api_pb2.TaskGetInfoResponse(
                app_id="ap-sandbox-app",
                info=api_pb2.TaskInfo(id=req.task_id, started_at=_TEST_TIMESTAMP - 3600, finished_at=_TEST_TIMESTAMP),
            )
        )

    async def fetch_handler(self, stream):
        req = await stream.recv_message()
        fetch_requests.append(req)
        await stream.send_message(_make_multi_line_fetch_response(100, task_id="ta-from-sandbox"))

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetTaskId", sandbox_get_task_handler)
        ctx.set_responder("TaskGetInfo", task_get_info_handler)
        ctx.set_responder("AppFetchLogs", fetch_handler)

        res = run_cli_command(["container", "logs", "sb-test1"])
        assert "line-0" in res.stdout
        # Verify the full resolution chain
        assert len(sandbox_get_task_requests) == 1
        assert sandbox_get_task_requests[0].sandbox_id == "sb-test1"
        assert len(task_get_info_requests) == 1
        assert task_get_info_requests[0].task_id == "ta-from-sandbox"
        assert len(fetch_requests) == 1
        assert fetch_requests[0].app_id == "ap-sandbox-app"


def test_container_logs_search(servicer, server_url_env, set_env_client):
    fetch_requests = []

    async def fetch_handler(self, stream):
        req = await stream.recv_message()
        fetch_requests.append(req)
        await stream.send_message(_make_multi_line_fetch_response(100))

    with servicer.intercept() as ctx:
        ctx.set_responder("TaskGetInfo", _make_task_get_info_responder())
        ctx.set_responder("AppFetchLogs", fetch_handler)

        res = run_cli_command(["container", "logs", "ta-test1", "--search", "matched"])
        assert fetch_requests[0].search_text == "matched"
