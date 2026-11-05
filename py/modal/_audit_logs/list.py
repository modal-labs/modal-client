# Copyright Modal Labs 2026
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing

from modal._audit_logs import _rpcs
from modal._utils.async_utils import async_map_ordered
from modal.client import _Client
from modal_proto import api_pb2

# (newest_cursor, oldest_cursor); None where the bucket is open-ended.
Bucket = tuple[api_pb2.AuditLogCursor | None, api_pb2.AuditLogCursor | None]


async def get_bucket_cursors(
    client: _Client, q: str, oldest_ms: int, newest_ms: int, config: api_pb2.AuditLogSettingsGetResponse
) -> list[Bucket]:
    """Split the range into buckets, as (newest_cursor, oldest_cursor) pairs ordered oldest to newest.

    The first bucket has no oldest cursor and the last has no newest cursor.
    """
    cursors = await _rpcs.plan_buckets(client, q, oldest_ms, newest_ms, config)
    return list(zip([None, *cursors], [*cursors, None]))[::-1]


async def _fetch_bucket(
    client: _Client,
    q: str,
    oldest_ms: int,
    newest_ms: int,
    config: api_pb2.AuditLogSettingsGetResponse,
    bucket: Bucket,
) -> list[api_pb2.AuditLogEntry]:
    """Read one bucket's entries, oldest to newest, using pagination"""
    newest_cursor, oldest_cursor = bucket
    entries: list[api_pb2.AuditLogEntry] = []
    while True:
        resp = await _rpcs.list_page(
            client, q, oldest_ms, newest_ms, config.page_size, config, newest_cursor, oldest_cursor
        )
        entries.extend(resp.items)
        # Skips the empty request a full last page would otherwise trigger; an unset next_cursor also ends the loop.
        if oldest_cursor is not None and resp.items and resp.items[-1].id == oldest_cursor.id:
            break
        if not resp.HasField("next_cursor"):
            break
        newest_cursor = resp.next_cursor
    entries.reverse()
    return entries


async def fetch_audit_logs(
    q: str,
    oldest_ms: int,
    newest_ms: int,
    *,
    client: _Client | None = None,
    config: api_pb2.AuditLogSettingsGetResponse,
) -> AsyncGenerator[api_pb2.AuditLogEntry]:
    """Yield the range's entries from the oldest to the newest, one bucket at a time.

    At most `max_concurrent_requests` buckets are fetched at once, and at most `max_buffered_buckets` are fetched
    or held at once. A new bucket is admitted only after the consumer has taken all of a bucket's entries, so a slow
    consumer or a stalled request cannot grow memory.
    If any request fails after its attempts, the remaining requests are cancelled and the error is raised.
    """
    if client is None:
        client = await _Client.from_env()
    buckets = await get_bucket_cursors(client, q, oldest_ms, newest_ms, config)

    async def bucket_stream() -> AsyncGenerator[Bucket]:
        for bucket in buckets:
            yield bucket

    async def fetch(bucket: Bucket) -> list[api_pb2.AuditLogEntry]:
        return await _fetch_bucket(client, q, oldest_ms, newest_ms, config, bucket)

    fetched = async_map_ordered(bucket_stream(), fetch, config.max_concurrent_requests, config.max_buffered_buckets)
    async with aclosing(fetched) as stream:
        async for entries in stream:
            for entry in entries:
                yield entry
