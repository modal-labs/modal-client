# Copyright Modal Labs 2026
"""The audit log RPCs, one call each, retried with backoff. Ordering and batching live in list.py."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from modal._utils.grpc_utils import Retry
from modal_proto import api_pb2

if TYPE_CHECKING:
    from modal.client import _Client


_Resp = TypeVar("_Resp")


_MAX_RETRIES = 4
_PLAN_MAX_RETRIES = 2


async def _call(
    rpc: Callable[..., Awaitable[_Resp]], req: object, *, timeout: float, max_retries: int = _MAX_RETRIES
) -> _Resp:
    """Call with backoff on timeouts and transient errors; rate and resource limits wait as the server says."""
    retry = Retry(base_delay=1.0, delay_factor=2.0, max_delay=8.0, max_retries=max_retries, attempt_timeout=timeout)
    return await rpc(req, retry=retry)


async def get_settings(client: _Client) -> api_pb2.AuditLogSettingsGetResponse:
    return await client._stub.AuditLogSettingsGet(api_pb2.AuditLogSettingsGetRequest())


async def count_entries(
    client: _Client, q: str, oldest_ms: int, newest_ms: int, config: api_pb2.AuditLogSettingsGetResponse
) -> int:
    req = api_pb2.AuditLogCountRequest(q=q, oldest_ms=oldest_ms, newest_ms=newest_ms)
    resp = await _call(client._stub.AuditLogCount, req, timeout=config.request_timeout_secs)
    return resp.count


async def list_page(
    client: _Client,
    q: str,
    oldest_ms: int,
    newest_ms: int,
    limit: int,
    config: api_pb2.AuditLogSettingsGetResponse,
    newest_cursor: api_pb2.AuditLogCursor | None = None,
    oldest_cursor: api_pb2.AuditLogCursor | None = None,
) -> api_pb2.AuditLogListResponse:
    req = api_pb2.AuditLogListRequest(q=q, oldest_ms=oldest_ms, newest_ms=newest_ms, limit=limit)
    if newest_cursor is not None:
        req.newest_cursor.CopyFrom(newest_cursor)
    if oldest_cursor is not None:
        req.oldest_cursor.CopyFrom(oldest_cursor)
    return await _call(client._stub.AuditLogList, req, timeout=config.request_timeout_secs)


async def plan_buckets(
    client: _Client, q: str, oldest_ms: int, newest_ms: int, config: api_pb2.AuditLogSettingsGetResponse
) -> list[api_pb2.AuditLogCursor]:
    req = api_pb2.AuditLogBucketsRequest(q=q, oldest_ms=oldest_ms, newest_ms=newest_ms, bucket_size=config.bucket_size)
    resp = await _call(client._stub.AuditLogPlan, req, timeout=config.plan_timeout_secs, max_retries=_PLAN_MAX_RETRIES)
    return list(resp.cursors)
