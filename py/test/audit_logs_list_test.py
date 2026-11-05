# Copyright Modal Labs 2026
import asyncio
import pytest
from typing import Any

from grpclib import GRPCError, Status

from modal._audit_logs.list import fetch_audit_logs as _fetch_audit_logs, get_bucket_cursors as _get_bucket_cursors
from modal._utils.async_utils import synchronize_api
from modal.exception import InvalidError
from modal_proto import api_pb2

from .audit_logs_test_helpers import RANGE, RATE_LIMIT_ERROR, make_settings, no_retry_backoff  # noqa: F401

fetch_audit_logs = synchronize_api(_fetch_audit_logs)
get_bucket_cursors = synchronize_api(_get_bucket_cursors)


def _serve(ctx: Any, **handlers: Any) -> None:
    """Serves each method with an async function from its request to its response."""
    for method, handler in handlers.items():

        async def responder(servicer, stream, handler=handler):
            await stream.send_message(await handler(await stream.recv_message()))

        ctx.set_responder(method, responder)


# ---------------------------------------------------------------------------
# bucket reads
# ---------------------------------------------------------------------------


def _cursor(id: str) -> api_pb2.AuditLogCursor:
    return api_pb2.AuditLogCursor(event_at_desc=-int(id[1:]) * 1000, id=id)


def _ids(entries) -> list[str]:
    return [e.id for e in entries]


def _page(*ids: str, next_cursor: str | None = None) -> api_pb2.AuditLogListResponse:
    resp = api_pb2.AuditLogListResponse(items=[api_pb2.AuditLogEntry(id=id) for id in ids])
    if next_cursor:
        resp.next_cursor.CopyFrom(_cursor(next_cursor))
    return resp


def _start(req) -> str | None:
    return req.newest_cursor.id if req.HasField("newest_cursor") else None


def _add_plan(ctx: Any, *cursor_ids: str) -> None:
    ctx.add_response("AuditLogPlan", api_pb2.AuditLogPlanResponse(cursors=[_cursor(i) for i in cursor_ids]))


def _add_page(ctx: Any, page: api_pb2.AuditLogListResponse, *, start: str | None = ...) -> None:  # type: ignore[assignment]
    """Queues a List response, for the request starting at `start` when given (None means no newest cursor)."""
    if start is ...:
        ctx.add_response("AuditLogList", page)
    else:
        ctx.add_response("AuditLogList", page, request_filter=lambda req: _start(req) == start)


async def _read(client: Any, config: api_pb2.AuditLogSettingsGetResponse | None = None) -> list:
    return [e async for e in fetch_audit_logs.aio(**RANGE, client=client, config=config or make_settings())]


@pytest.mark.asyncio
async def test_a_bucket_follows_next_cursor_newest_to_oldest_until_it_is_unset(servicer, client):
    with servicer.intercept() as ctx:
        _add_plan(ctx)
        _add_page(ctx, _page("e9", "e8", next_cursor="e8"))
        _add_page(ctx, _page("e7"))
        entries = await _read(client)
    requests = ctx.get_requests("AuditLogList")
    assert _ids(entries) == ["e7", "e8", "e9"]
    assert [r.limit for r in requests] == [5000, 5000]
    assert not requests[0].HasField("newest_cursor")
    assert requests[1].newest_cursor == _cursor("e8")


@pytest.mark.asyncio
async def test_each_bucket_is_read_between_its_cursors_with_the_configured_page_size(servicer, client):
    with servicer.intercept() as ctx:
        _add_plan(ctx, "e7", "e3")
        for _ in range(3):
            _add_page(ctx, _page("e5"))
        await _read(client, make_settings(page_size=1234))
    requests = ctx.get_requests("AuditLogList")
    assert {r.limit for r in requests} == {1234}
    middle = next(r for r in requests if r.HasField("newest_cursor") and r.HasField("oldest_cursor"))
    assert (middle.newest_cursor, middle.oldest_cursor) == (_cursor("e7"), _cursor("e3"))


@pytest.mark.asyncio
async def test_a_bucket_stops_at_the_page_that_ends_on_its_oldest_cursor_even_when_the_page_is_full(servicer, client):
    with servicer.intercept() as ctx:
        _add_plan(ctx, "e7")
        _add_page(ctx, _page("e9", "e8", "e7", next_cursor="e7"), start=None)  # a full page of 3, ending on e7
        _add_page(ctx, _page("e6"), start="e7")
        entries = await _read(client, make_settings(page_size=3))
    assert _ids(entries) == ["e6", "e7", "e8", "e9"]
    starts = sorted(_start(r) or "" for r in ctx.get_requests("AuditLogList"))
    assert starts == ["", "e7"]  # no extra request for the newest bucket


@pytest.mark.asyncio
async def test_a_bucket_continues_when_a_full_page_does_not_end_on_its_oldest_cursor(servicer, client):
    with servicer.intercept() as ctx:
        _add_plan(ctx, "e5")
        _add_page(ctx, _page("e9", "e8", next_cursor="e8"), start=None)
        _add_page(ctx, _page("e7", "e6", next_cursor="e6"), start="e8")
        _add_page(ctx, _page("e5"), start="e6")
        _add_page(ctx, _page("e4"), start="e5")
        entries = await _read(client, make_settings(page_size=2))
    assert _ids(entries) == ["e4", "e5", "e6", "e7", "e8", "e9"]


@pytest.mark.asyncio
async def test_a_rate_limited_list_request_is_retried_with_the_same_limit(servicer, client):
    with servicer.intercept() as ctx:
        _add_plan(ctx)
        ctx.add_response("AuditLogList", RATE_LIMIT_ERROR)
        _add_page(ctx, _page("e9", next_cursor="e9"))
        _add_page(ctx, _page("e8"))
        entries = await _read(client)
    assert _ids(entries) == ["e8", "e9"]
    assert [r.limit for r in ctx.get_requests("AuditLogList")] == [5000, 5000, 5000]  # retries keep the page size


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cursor_ids, expected",
    [
        ([], [(None, None)]),
        (["e5"], [("e5", None), (None, "e5")]),
        (["e7", "e4"], [("e4", None), ("e7", "e4"), (None, "e7")]),
    ],
)
async def test_plan_buckets_run_between_consecutive_cursors(servicer, client, cursor_ids, expected):
    with servicer.intercept() as ctx:
        _add_plan(ctx, *cursor_ids)
        _add_plan(ctx, *cursor_ids)
        buckets = await get_bucket_cursors.aio(client, **RANGE, config=make_settings(bucket_size=2000))
        await get_bucket_cursors.aio(client, **RANGE, config=make_settings())
    assert buckets == [(_cursor(n) if n else None, _cursor(o) if o else None) for n, o in expected]
    assert [r.bucket_size for r in ctx.get_requests("AuditLogPlan")] == [2000, 5000]


# ---------------------------------------------------------------------------
# the whole export
# ---------------------------------------------------------------------------

# Plan cut points e7 and e4 give these buckets, newest to oldest, each ending on its own cut point.
BUCKET_ROWS = {None: ["e9", "e8", "e7"], "e7": ["e6", "e5", "e4"], "e4": ["e3", "e2"]}


def _serve_export(ctx: Any, cursor_ids, bucket_rows, *, on_list=None) -> list:
    """Serves a plan with `cursor_ids` and a List that returns the rows of the bucket starting at the cursor."""
    starts: list = []

    async def AuditLogPlan(_req):
        return api_pb2.AuditLogPlanResponse(cursors=[_cursor(i) for i in cursor_ids])

    async def AuditLogList(req):
        start = _start(req)
        starts.append(start)
        if on_list is not None:
            await on_list(start)
        rows = bucket_rows[start]
        # Like the server, set next_cursor only when the page is full.
        return _page(*rows, next_cursor=rows[-1] if len(rows) == req.limit else None)

    _serve(ctx, AuditLogPlan=AuditLogPlan, AuditLogList=AuditLogList)
    return starts


async def _collect(client):
    return await _read(client, make_settings(page_size=3))


@pytest.mark.asyncio
async def test_fetch_audit_logs_yields_the_oldest_bucket_first_with_each_bucket_oldest_to_newest(servicer, client):
    with servicer.intercept() as ctx:
        starts = _serve_export(ctx, ["e7", "e4"], BUCKET_ROWS)
        entries = await _collect(client)

    assert _ids(entries) == ["e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"]
    assert sorted(s or "" for s in starts) == ["", "e4", "e7"]  # one request per bucket


@pytest.mark.asyncio
async def test_fetch_audit_logs_keeps_the_order_when_buckets_finish_out_of_order(servicer, client):
    all_started = asyncio.Event()
    others_done = asyncio.Event()
    started: list = []
    finished: list = []

    async def on_list(start):
        started.append(start)
        if len(started) == 3:
            all_started.set()
        await all_started.wait()  # all three requests are in flight together
        if start == "e4":  # the oldest bucket answers last, after the other two have answered
            await others_done.wait()
        else:
            finished.append(start)
            if len(finished) == 2:
                others_done.set()

    with servicer.intercept() as ctx:
        _serve_export(ctx, ["e7", "e4"], BUCKET_ROWS, on_list=on_list)
        entries = await _collect(client)
    assert _ids(entries) == ["e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"]


async def _run_on_client_loop_inner(scenario: Any) -> Any:
    return await scenario()


_run_on_client_loop = synchronize_api(_run_on_client_loop_inner)


async def _read_with_a_stalled_oldest_bucket(
    client, monkeypatch, max_concurrent_requests: int, max_buffered_buckets: int
):
    """Read 11 buckets while the oldest stays stalled until `max_buffered_buckets` have started and finished.

    Runs on the client's own event loop against in-process List and Plan handlers, so every yield is a loop turn and
    the result does not depend on timing. Returns the peak number of requests in flight and of buckets started but
    not yet consumed.
    """
    cursor_ids = [f"e{n}" for n in range(30, 0, -3)]  # 10 cut points, so 11 buckets
    rows = {None: ["r31"]} | {c: [f"r{c[1:]}"] for c in cursor_ids}
    config = make_settings(
        page_size=3, max_concurrent_requests=max_concurrent_requests, max_buffered_buckets=max_buffered_buckets
    )
    stub = client._stub

    async def scenario():
        release_oldest = asyncio.Event()
        started = consumed = active = peak_active = peak_outstanding = 0

        async def AuditLogPlan(_req, *, retry):
            return api_pb2.AuditLogPlanResponse(cursors=[_cursor(i) for i in cursor_ids])

        async def AuditLogList(req, *, retry):
            nonlocal started, active, peak_active, peak_outstanding
            start = _start(req)
            started += 1
            active += 1
            peak_active = max(peak_active, active)
            peak_outstanding = max(peak_outstanding, started - consumed)
            try:
                if start == cursor_ids[-1]:  # the oldest bucket is the one the consumer waits on
                    await release_oldest.wait()
                else:
                    await asyncio.sleep(0)
                    if started >= max_buffered_buckets:
                        release_oldest.set()
            finally:
                active -= 1
            return _page(*rows[start])

        monkeypatch.setattr(stub, "AuditLogPlan", AuditLogPlan)
        monkeypatch.setattr(stub, "AuditLogList", AuditLogList)
        async for _ in fetch_audit_logs.aio(**RANGE, client=client, config=config):
            if consumed == 0:
                for _ in range(20):  # any bucket started beyond the buffer limit would start within these turns
                    await asyncio.sleep(0)
            consumed += 1
        return peak_active, peak_outstanding

    return await _run_on_client_loop.aio(scenario)


@pytest.mark.asyncio
async def test_fetch_audit_logs_starts_the_next_bucket_only_after_the_consumer_takes_its_entries(client, monkeypatch):
    peak_active, peak_outstanding = await _read_with_a_stalled_oldest_bucket(client, monkeypatch, 10, 3)
    assert peak_active <= 3  # the buffer, not the request limit, is what holds this back
    assert peak_outstanding == 3


@pytest.mark.asyncio
async def test_fetch_audit_logs_keeps_fetching_past_a_slow_oldest_bucket_within_the_request_limit(client, monkeypatch):
    assert await _read_with_a_stalled_oldest_bucket(client, monkeypatch, 2, 5) == (2, 5)


@pytest.mark.asyncio
async def test_a_failed_bucket_cancels_the_rest_and_raises(servicer, client):
    cancelled = []
    all_cancelled = asyncio.Event()

    async def on_list(start):
        if start == "e4":  # the oldest bucket is read first and fails
            raise GRPCError(Status.INVALID_ARGUMENT, "bad request")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(start)
            if len(cancelled) == 2:
                all_cancelled.set()
            raise

    with servicer.intercept() as ctx:
        _serve_export(ctx, ["e7", "e4"], BUCKET_ROWS, on_list=on_list)
        with pytest.raises(InvalidError, match="bad request"):
            await asyncio.wait_for(_collect(client), timeout=10)  # fails fast if the other requests are not cancelled
        await asyncio.wait_for(
            all_cancelled.wait(), timeout=10
        )  # the server sees the cancels after the client gives up
    assert sorted(c or "" for c in cancelled) == ["", "e7"]
