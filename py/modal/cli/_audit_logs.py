# Copyright Modal Labs 2026
import contextlib
import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone

import click

from modal._utils.logger import logger
from modal._workspace import _AUDIT_LOG_DEFAULT_WINDOW, _WorkspaceAuditLogsManager
from modal.types import AuditEvent

from ._logs import _parse_fetch_range


def _format_json(entry: AuditEvent) -> str:
    return json.dumps(
        {
            "id": entry.id,
            "timestamp": entry.timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
            "action": entry.action,
            "status": entry.status,
            "actor": {
                "type": entry.actor.type,
                "id": entry.actor.id,
                "name": entry.actor.name,
                "email": entry.actor.email,
            },
            "targets": [{"type": target.type, "id": target.id} for target in entry.targets],
            "context": {
                "workspace_id": entry.context.workspace_id,
                "environment_id": entry.context.environment_id,
                "environment_name": entry.context.environment_name,
                "source": entry.context.source,
                "ip_address": str(entry.context.ip_address) if entry.context.ip_address else None,
            },
            "metadata": entry.metadata,
        }
    )


async def _print_events(
    manager: _WorkspaceAuditLogsManager,
    *,
    query: str,
    since: datetime | None,
    until: datetime,
    limit: int | None,
) -> None:
    stream = manager.fetch(query, since=since, until=until, limit=limit)
    try:
        async for entry in stream:
            click.echo(_format_json(entry))
    finally:
        await stream.aclose()


class _DropRateLimitWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ("/AuditLog" in message and "RESOURCE_EXHAUSTED" in message)


@contextlib.contextmanager
def _quiet_rate_limit_warnings() -> Iterator[None]:
    """Hides the SDK's per-retry rate limit warnings so they don't interleave with the printed events."""
    log_filter = _DropRateLimitWarnings()
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


async def run_audit_logs_command(
    manager: _WorkspaceAuditLogsManager,
    *,
    query: str,
    since: str | None,
    until: str | None,
    limit: int | None,
    count_only: bool,
) -> None:
    if count_only and since is None:
        raise click.UsageError("--since is required with --count.")
    since_dt, until_dt = _parse_fetch_range(since, until, _AUDIT_LOG_DEFAULT_WINDOW)

    with _quiet_rate_limit_warnings():
        if count_only:
            click.echo(await manager.count(query, since=since_dt, until=until_dt))
            return

        await _print_events(
            manager,
            query=query,
            since=since_dt if since is not None else None,
            until=until_dt,
            limit=limit,
        )
