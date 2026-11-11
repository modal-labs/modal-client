# Copyright Modal Labs 2026
import asyncio
import pytest
from typing import Any

from grpclib import GRPCError, Status

from modal._audit_logs import _rpcs
from modal._audit_logs.count import count_audit_logs as _count_audit_logs
from modal._utils.async_utils import synchronize_api
from modal.exception import ConnectionError as ModalConnectionError, InvalidError, ResourceExhaustedError, ServiceError
from modal_proto import api_pb2

from .audit_logs_test_helpers import (  # noqa: F401
    DEADLINE_ERROR,
    RANGE,
    RATE_LIMIT_ERROR,
    make_settings,
    no_retry_backoff,
)

RESOURCE_LIMIT_ERROR = GRPCError(Status.RESOURCE_EXHAUSTED, "Audit logs count query hit resource limit")
HANG = "hang"

count_audit_logs = synchronize_api(_count_audit_logs)
get_settings = synchronize_api(_rpcs.get_settings)
plan_buckets = synchronize_api(_rpcs.plan_buckets)


def _count(n: int) -> api_pb2.AuditLogCountResponse:
    return api_pb2.AuditLogCountResponse(count=n)


def _serve_outcomes(ctx: Any, method: str, *outcomes: Any) -> None:
    """Serves a method's calls in order: HANG never answers, an exception is raised, a message is sent."""
    remaining = iter(outcomes)

    async def responder(servicer, stream):
        await stream.recv_message()
        outcome = next(remaining)
        if outcome == HANG:
            await asyncio.Event().wait()
        if isinstance(outcome, Exception):
            raise outcome
        await stream.send_message(outcome)

    ctx.set_responder(method, responder)


@pytest.mark.asyncio
async def test_get_settings_returns_the_servers_settings(servicer, client):
    settings = make_settings(page_size=1234, max_concurrent_requests=7, request_timeout_secs=12.5)
    with servicer.intercept() as ctx:
        ctx.add_response("AuditLogSettingsGet", settings)
        assert await get_settings.aio(client) == settings
    assert ctx.get_requests("AuditLogSettingsGet") == [api_pb2.AuditLogSettingsGetRequest()]


@pytest.mark.asyncio
async def test_count_audit_logs_sends_the_range_and_returns_the_count(servicer, client):
    with servicer.intercept() as ctx:
        ctx.add_response("AuditLogCount", _count(42))
        assert await count_audit_logs.aio(client, "status:error", 1000, 9000, make_settings()) == 42
    assert ctx.get_requests("AuditLogCount") == [
        api_pb2.AuditLogCountRequest(q="status:error", oldest_ms=1000, newest_ms=9000)
    ]


@pytest.mark.asyncio
async def test_a_rate_limited_request_is_retried(servicer, client):
    with servicer.intercept() as ctx:
        ctx.add_response("AuditLogCount", RATE_LIMIT_ERROR)
        ctx.add_response("AuditLogCount", _count(7))
        assert await count_audit_logs.aio(client, **RANGE, config=make_settings()) == 7
    first, second = ctx.get_requests("AuditLogCount")
    assert first == second  # a retry repeats the request unchanged


@pytest.mark.asyncio
async def test_a_resource_limit_error_without_a_retry_hint_is_not_retried(servicer, client):
    with servicer.intercept() as ctx:
        ctx.add_response("AuditLogCount", RESOURCE_LIMIT_ERROR)
        with pytest.raises(ResourceExhaustedError, match="resource limit"):
            await count_audit_logs.aio(client, **RANGE, config=make_settings())
    assert len(ctx.get_requests("AuditLogCount")) == 1


@pytest.mark.asyncio
async def test_requests_do_not_retry_other_errors(servicer, client):
    with servicer.intercept() as ctx:
        ctx.add_response("AuditLogCount", GRPCError(Status.INVALID_ARGUMENT, "bad query"))
        with pytest.raises(InvalidError, match="bad query"):
            await count_audit_logs.aio(client, **RANGE, config=make_settings())
    assert len(ctx.get_requests("AuditLogCount")) == 1


@pytest.mark.asyncio
async def test_a_stalled_request_is_retried(servicer, client):
    with servicer.intercept() as ctx:
        ctx.add_response("AuditLogCount", DEADLINE_ERROR)
        ctx.add_response("AuditLogCount", _count(3))
        assert await count_audit_logs.aio(client, **RANGE, config=make_settings()) == 3
    assert len(ctx.get_requests("AuditLogCount")) == 2


@pytest.mark.asyncio
async def test_a_request_that_keeps_stalling_fails_after_five_attempts(servicer, client):
    with servicer.intercept() as ctx:
        for _ in range(5):
            ctx.add_response("AuditLogCount", DEADLINE_ERROR)
        with pytest.raises(ServiceError, match="Deadline Exceeded"):
            await count_audit_logs.aio(client, **RANGE, config=make_settings())
    assert len(ctx.get_requests("AuditLogCount")) == 5


@pytest.mark.asyncio
async def test_a_plan_that_keeps_stalling_fails_after_three_attempts(servicer, client):
    with servicer.intercept() as ctx:
        for _ in range(3):
            ctx.add_response("AuditLogPlan", DEADLINE_ERROR)
        with pytest.raises(ServiceError, match="Deadline Exceeded"):
            await plan_buckets.aio(client, **RANGE, config=make_settings())
    assert len(ctx.get_requests("AuditLogPlan")) == 3


@pytest.mark.asyncio
async def test_a_request_that_never_answers_exhausts_deadline_retries(servicer, client):
    config = make_settings(request_timeout_secs=0.2)
    with servicer.intercept() as ctx:
        _serve_outcomes(ctx, "AuditLogCount", HANG, HANG, HANG, HANG, HANG)
        # Both peers enforce the RPC deadline, so either timeout can win the race.
        with pytest.raises((ModalConnectionError, ServiceError)) as exc_info:
            await count_audit_logs.aio(client, **RANGE, config=config)
        if isinstance(exc_info.value, ServiceError):
            assert exc_info.value._grpc_status == Status.DEADLINE_EXCEEDED
        else:
            assert "Deadline exceeded" in str(exc_info.value)
    assert len(ctx.get_requests("AuditLogCount")) == 5
