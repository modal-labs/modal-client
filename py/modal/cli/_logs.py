# Copyright Modal Labs 2026
from datetime import datetime, timezone

from click import UsageError

from modal._logs import _FETCH_LIMIT, _MAX_FETCH_RANGE, LogsFilters
from modal._utils.time_utils import locale_tz, parse_duration
from modal_proto import api_pb2

from .utils import _fetch_app_logs, _stream_app_logs, _tail_app_logs

_DEFAULT_LOGS_TAIL = 100
_RELATIVE_TIME_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_SOURCE_OPTIONS = {
    "stdout": api_pb2.FILE_DESCRIPTOR_STDOUT,
    "stderr": api_pb2.FILE_DESCRIPTOR_STDERR,
    "system": api_pb2.FILE_DESCRIPTOR_INFO,
}


def _parse_time_arg(value: str | None, default: datetime) -> datetime:
    """Parse a time argument that can be a relative duration (e.g. '2h', '30m') or ISO 8601 datetime.

    Naive datetime values are interpreted in the user's local timezone.
    Relative durations are always UTC-relative.
    """
    if value is None:
        return default

    try:
        duration = parse_duration(value)
    except ValueError:
        pass
    else:
        return datetime.now(timezone.utc) - duration

    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=locale_tz())
        return dt
    except ValueError:
        raise UsageError(f"Invalid time format: '{value}'. Use a relative duration (e.g. '2h') or ISO 8601 datetime.")


def _validate_logs_args(*, follow: bool, since: str | None, until: str | None, tail: int | None) -> None:
    if follow and (since or until or tail):
        raise UsageError("--follow cannot be combined with --since, --until, or --tail.")
    if tail is not None and tail <= 0:
        raise UsageError("--tail value must be positive.")
    if tail is not None and tail > _FETCH_LIMIT:
        raise UsageError(f"--tail value must not exceed {_FETCH_LIMIT}.")


async def _run_logs_command(
    app_id: str,
    *,
    follow: bool,
    since: str | None,
    until: str | None,
    tail: int | None,
    search: str | None,
    function_id: str = "",
    function_call_id: str = "",
    container_id: str = "",
    source: str | None,
    timestamps: bool,
    prefix_fields: list[str],
) -> None:
    if source is not None:
        if source not in _SOURCE_OPTIONS:
            raise UsageError(f"Invalid source: '{source}'. Must be 'stdout', 'stderr', or 'system'.")
        source_fd = _SOURCE_OPTIONS[source]
    else:
        source_fd = api_pb2.FILE_DESCRIPTOR_UNSPECIFIED

    log_filters = LogsFilters(
        source=source_fd,
        function_id=function_id,
        function_call_id=function_call_id,
        task_id=container_id,
        search_text=search or "",
    )

    if follow:
        await _stream_app_logs(
            app_id,
            task_id=container_id,
            show_timestamps=timestamps,
            follow=True,
            prefix_fields=prefix_fields,
            filters=log_filters,
        )
        return

    now = datetime.now(timezone.utc)
    since_dt = _parse_time_arg(since, default=now) if since else None
    until_dt = _parse_time_arg(until, default=now) if until else None
    if since_dt is not None:
        effective_until = until_dt or now
        if since_dt >= effective_until:
            raise UsageError("--since must be before --until.")
        if effective_until - since_dt > _MAX_FETCH_RANGE:
            raise UsageError(f"Log fetch time range cannot exceed {_MAX_FETCH_RANGE.days} days.")

    if since_dt is not None and tail is None:
        await _fetch_app_logs(
            app_id,
            since_dt,
            until_dt or now,
            show_timestamps=timestamps,
            prefix_fields=prefix_fields,
            filters=log_filters,
        )
    else:
        await _tail_app_logs(
            app_id,
            tail if tail is not None else _DEFAULT_LOGS_TAIL,
            show_timestamps=timestamps,
            since=since_dt,
            until=until_dt,
            prefix_fields=prefix_fields,
            filters=log_filters,
        )
