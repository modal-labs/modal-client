# Copyright Modal Labs 2026
from __future__ import annotations

from typing import TYPE_CHECKING

from modal._audit_logs import _rpcs
from modal_proto import api_pb2

if TYPE_CHECKING:
    from modal.client import _Client


async def count_audit_logs(
    client: _Client, q: str, oldest_ms: int, newest_ms: int, config: api_pb2.AuditLogSettingsGetResponse
) -> int:
    """Count the entries in [oldest_ms, newest_ms) that match the query."""
    return await _rpcs.count_entries(client, q, oldest_ms, newest_ms, config)
