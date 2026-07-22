# Copyright Modal Labs 2026
from __future__ import annotations

import asyncio
import dataclasses
import enum
import socket
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime, timezone

from grpclib.exceptions import StreamTerminatedError

from modal._logs import fetch_logs, tail_logs
from modal._supports_logs import _LogQueryData, _SupportsLogs
from modal._utils.async_utils import synchronize_api
from modal._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES as _RETRYABLE_STREAM_STATUSES
from modal._utils.time_utils import locale_tz
from modal.config import logger
from modal.exception import InternalError, NotFoundError, ResourceExhaustedError, ServiceError
from modal.types import LogEntry, LogSource
from modal_proto import api_pb2

_STREAM_POLL_INTERVAL_SECONDS = 1
_STREAM_RPC_TIMEOUT_SECONDS = 55.0


@dataclasses.dataclass
class _Deadline:
    value: float | None = None

    def reset(self, timeout: float | None = None) -> None:
        self.value = time.monotonic() + timeout if timeout is not None else None


class _StreamStopReason(enum.Enum):
    IDLE_TIMEOUT = "idle_timeout"
    STOP_STREAM = "stop_stream"


class _LogsManager:
    """mdmd:namespace"""

    def __init__(self, source: _SupportsLogs, stop_stream: Callable[[], Awaitable[bool]] | None = None):
        """mdmd:hidden"""
        self._source = source
        self._stop_stream = stop_stream
        self._query_params: _LogQueryData | None = None

    async def _params(self) -> _LogQueryData:
        if self._query_params is None:
            self._query_params = await self._source._get_log_query_data()
        return self._query_params

    @staticmethod
    def _resolve_source(source: LogSource | None) -> api_pb2.FileDescriptor.ValueType:
        match source:
            case "stderr":
                return api_pb2.FILE_DESCRIPTOR_STDERR
            case "stdout":
                return api_pb2.FILE_DESCRIPTOR_STDOUT
            case "system":
                return api_pb2.FILE_DESCRIPTOR_INFO
            case None:
                return api_pb2.FILE_DESCRIPTOR_UNSPECIFIED
            case _:
                raise ValueError("source must be one of 'stdout', 'stderr', 'system', or None")

    @staticmethod
    def _normalize_utc_datetime(value: datetime, name: str) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError(f"{name} must be a datetime")
        if value.tzinfo is None:
            return value.replace(tzinfo=locale_tz()).astimezone(timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _entry_source(file_descriptor: api_pb2.FileDescriptor.ValueType) -> LogSource:
        match file_descriptor:
            case api_pb2.FILE_DESCRIPTOR_STDOUT:
                return "stdout"
            case api_pb2.FILE_DESCRIPTOR_STDERR:
                return "stderr"
            case _:
                return "system"

    @staticmethod
    def _entry_timestamp(item: api_pb2.TaskLogs) -> datetime:
        if item.timestamp_ns:
            return datetime.fromtimestamp(item.timestamp_ns / 1_000_000_000, timezone.utc)
        return datetime.fromtimestamp(item.timestamp, timezone.utc)

    def _entry_context_ids(self, item: api_pb2.TaskLogs, batch: api_pb2.TaskLogsBatch) -> list[str]:
        match self._source.object_id[:2]:
            case "fu":
                context_ids = [
                    item.function_call_id,
                    item.input_id or batch.input_id,
                    item.container_id or batch.task_id,
                ]
            case "fc":
                context_ids = [item.input_id or batch.input_id, item.container_id or batch.task_id]
            case _:
                context_ids = []
        # filter for empty strings, can happen for Server which have function_ids but no input/container
        return [c for c in context_ids if c]

    def _entry_from_item(self, item: api_pb2.TaskLogs, batch: api_pb2.TaskLogsBatch) -> LogEntry:
        return LogEntry(
            message=item.data,
            timestamp=self._entry_timestamp(item),
            source=self._entry_source(item.file_descriptor),
            object_id=self._source.object_id,
            context_ids=self._entry_context_ids(item, batch),
        )

    async def _watch_stream_stop(self, deadline: _Deadline) -> _StreamStopReason | None:
        while True:
            if deadline.value is not None and time.monotonic() >= deadline.value:
                return _StreamStopReason.IDLE_TIMEOUT
            if self._stop_stream is not None and await self._stop_stream():
                return _StreamStopReason.STOP_STREAM

            sleep_time = _STREAM_POLL_INTERVAL_SECONDS
            if deadline.value is not None:
                sleep_time = min(sleep_time, max(0.0, deadline.value - time.monotonic()))
            await asyncio.sleep(sleep_time)

    async def fetch(
        self,
        *,
        since: datetime,
        until: datetime | None = None,
        source: LogSource | None = None,
        search_text: str = "",
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch all associated logs corresponding to the date range and filters."""
        since = self._normalize_utc_datetime(since, "since")
        until = self._normalize_utc_datetime(until or datetime.now(timezone.utc), "until")
        target = await self._params()
        filters = dataclasses.replace(target.filters, source=self._resolve_source(source), search_text=search_text)
        async for batch in fetch_logs(target.client, target.app_id, since=since, until=until, filters=filters):
            for item in batch.items:
                yield self._entry_from_item(item, batch)

    async def tail(
        self,
        entries: int = 100,
        *,
        source: LogSource | None = None,
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch the most recent logs."""
        params = await self._params()
        filters = dataclasses.replace(params.filters, source=self._resolve_source(source))
        async for batch in tail_logs(
            params.client,
            params.app_id,
            entries,
            filters=filters,
        ):
            for item in batch.items:
                yield self._entry_from_item(item, batch)

    def _create_log_stream(
        self, params: _LogQueryData, last_entry_id: str, timeout: float
    ) -> AsyncGenerator[api_pb2.TaskLogsBatch, None]:
        """Helper for creating the log stream RPC."""
        request = api_pb2.AppGetLogsRequest(
            app_id=params.app_id,
            timeout=timeout,
            last_entry_id=last_entry_id,
            file_descriptor=params.filters.source,
            function_id=params.filters.function_id,
            function_call_id=params.filters.function_call_id,
            task_id=params.filters.task_id,
            sandbox_id=params.filters.sandbox_id,
        )
        return params.client.stub.AppGetLogs.unary_stream(request)

    def _stream_entries(self, batch: api_pb2.TaskLogsBatch):
        for item in batch.items:
            if item.data:
                yield self._entry_from_item(item, batch)

    @staticmethod
    def _advance_batch(batch: api_pb2.TaskLogsBatch, last_entry_id: str) -> tuple[str, bool]:
        if batch.entry_id:
            last_entry_id = batch.entry_id
        return last_entry_id, batch.app_done

    @staticmethod
    def _is_transient_stream_error(exc: Exception) -> bool:
        if isinstance(exc, (StreamTerminatedError, socket.gaierror)):
            return True
        elif isinstance(exc, (ServiceError, InternalError)):
            return getattr(exc, "_grpc_status", None) in _RETRYABLE_STREAM_STATUSES
        return isinstance(exc, AttributeError) and "_write_appdata" in str(exc)

    async def _drain_stream(
        self, params: _LogQueryData, last_entry_id: str, timeout: float
    ) -> AsyncGenerator[LogEntry, None]:
        """Do a final bounded drain in the logs to catch any remaining logs after the stop condition is met."""
        log_stream = self._create_log_stream(params, last_entry_id, timeout)
        try:
            async for batch in log_stream:
                for entry in self._stream_entries(batch):
                    yield entry
                if batch.app_done:
                    return
        finally:
            aclose = getattr(log_stream, "aclose", None)
            if aclose is not None:
                await aclose()

    async def stream(self, timeout: float | None = None) -> AsyncGenerator[LogEntry, None]:
        """Stream new logs until no logs arrive within the timeout."""
        if timeout is not None and timeout <= 0:
            return

        params = await self._params()
        last_entry_id = ""
        watcher_task: asyncio.Task[_StreamStopReason | None] | None = None
        deadline = _Deadline(value=time.monotonic() + timeout if timeout is not None else None)
        if timeout is not None or self._stop_stream is not None:
            watcher_task = asyncio.create_task(self._watch_stream_stop(deadline))

        retries_remaining = 10
        delay_ms = 1

        try:  # noqa: PLR1702
            while watcher_task is None or not watcher_task.done():
                log_stream = None
                try:
                    log_stream = self._create_log_stream(params, last_entry_id, timeout=_STREAM_RPC_TIMEOUT_SECONDS)
                    while watcher_task is None or not watcher_task.done():
                        if watcher_task is None:
                            try:
                                batch = await anext(log_stream)
                                retries_remaining = 10
                                delay_ms = 1
                            except StopAsyncIteration:
                                break
                            last_entry_id, app_done = self._advance_batch(batch, last_entry_id)
                            for log_entry in self._stream_entries(batch):
                                deadline.reset(timeout)
                                yield log_entry
                            if app_done:
                                return
                        else:
                            next_batch_task = asyncio.create_task(anext(log_stream))
                            done, _ = await asyncio.wait(
                                {next_batch_task, watcher_task}, return_when=asyncio.FIRST_COMPLETED
                            )

                            if next_batch_task in done:
                                try:
                                    batch = next_batch_task.result()
                                    retries_remaining = 10
                                    delay_ms = 1
                                except StopAsyncIteration:
                                    break
                                last_entry_id, app_done = self._advance_batch(batch, last_entry_id)
                                for log_entry in self._stream_entries(batch):
                                    deadline.reset(timeout)
                                    yield log_entry
                                if app_done:
                                    return
                                # drain after the stop condition to catch stragglers.
                                if watcher_task.done():
                                    stop_reason = watcher_task.result()
                                    if stop_reason == _StreamStopReason.STOP_STREAM:
                                        async for item in self._drain_stream(params, last_entry_id, 0.5):
                                            yield item
                                    return
                                continue

                            if watcher_task in done:
                                next_batch_task.cancel()
                                await self._suppress_cancelled(next_batch_task)
                                stop_reason = watcher_task.result()
                                # Drain stream for any remaining logs before returning.
                                if stop_reason == _StreamStopReason.STOP_STREAM:
                                    async for item in self._drain_stream(params, last_entry_id, 0.5):
                                        yield item
                                return
                except (ServiceError, InternalError, StreamTerminatedError, socket.gaierror, AttributeError) as exc:
                    if not self._is_transient_stream_error(exc):
                        raise
                    logger.debug("Log stream interrupted. Retrying ...")
                    if watcher_task is not None and watcher_task.done():
                        stop_reason = watcher_task.result()
                        if stop_reason == _StreamStopReason.STOP_STREAM:
                            async for item in self._drain_stream(params, last_entry_id, 0.5):
                                yield item
                        return
                    if retries_remaining <= 0:
                        raise
                    retries_remaining -= 1
                    await asyncio.sleep(delay_ms / 1000)
                    delay_ms = min(1000, delay_ms * 10)
                    continue
                finally:
                    if log_stream is not None:
                        aclose = getattr(log_stream, "aclose", None)
                        if aclose is not None:
                            await aclose()
        finally:
            if watcher_task is not None:
                watcher_task.cancel()
                await self._suppress_cancelled(watcher_task)

    @staticmethod
    async def _suppress_cancelled(task: asyncio.Task) -> None:
        try:
            await task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass


LogsManager = synchronize_api(_LogsManager, target_module=__name__)


class _FunctionLogsManager(_LogsManager):
    """mdmd:namespace"""

    def __init__(self, source: _SupportsLogs):
        """mdmd:hidden"""
        super().__init__(source)

    async def fetch(
        self,
        *,
        since: datetime,
        until: datetime | None = None,
        source: LogSource | None = None,
        search_text: str = "",
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch Function logs corresponding to the date range and filters.

        Args:
            since: Start date to fetch logs from. Must be in UTC or timezone-naive, which is interpreted as local time.
            until: Defaults to current date if None. Must be in UTC or timezone-naive, which is interpreted
                as local time.
            source: Filter by source: 'stdout', 'stderr', or 'system'.
            search_text: Filter by search text.

        Yields:
            `LogEntry` objects in chronological order.

        Examples:

            ```python notest
            function = modal.Function.from_name("my-app", "train")

            for entry in function.logs.fetch(
                since=datetime.now() - timedelta(hours=4),
                source="stdout",
            ):
                print(entry.message, end="")
            ```
        """
        async for log_entry in super().fetch(since=since, until=until, source=source, search_text=search_text):
            yield log_entry

    async def tail(
        self,
        entries: int = 100,
        *,
        source: LogSource | None = None,
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch the most recent Function logs.

        Args:
            entries: The number of log entries to return.
            source: Filter by source: 'stdout', 'stderr', or 'system'.

        Yields:
            `LogEntry` objects in chronological order.

        Examples:

            ```python notest
            function = modal.Function.from_name("my-app", "train")

            for entry in function.logs.tail(20):
                print(entry.message, end="")
            ```
        """
        async for log_entry in super().tail(entries, source=source):
            yield log_entry

    async def stream(self, timeout: float | None = None) -> AsyncGenerator[LogEntry, None]:
        """Stream new Function logs until the timeout is reached.

        Args:
            timeout: Number of seconds to wait between log entries before terminating the stream.
                By default, this will block until it is interrupted.

        Yields:
            `LogEntry` objects as they arrive.

        Examples:

            ```python notest
            function = modal.Function.from_name("my-app", "train")

            for entry in function.logs.stream(timeout=60):
                print(entry.message, end="")
            ```
        """
        async for log_entry in super().stream(timeout=timeout):
            yield log_entry


FunctionLogsManager = synchronize_api(_FunctionLogsManager, target_module=__name__)


class _ServerLogsManager(_LogsManager):
    """mdmd:namespace"""

    def __init__(self, source: _SupportsLogs):
        """mdmd:hidden"""
        super().__init__(source)

    async def fetch(
        self,
        *,
        since: datetime,
        until: datetime | None = None,
        source: LogSource | None = None,
        search_text: str = "",
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch Server logs corresponding to the date range and filters.

        Args:
            since: Start date to fetch logs from. Must be in UTC or timezone-naive, which is interpreted as local time.
            until: Defaults to current date if None. Must be in UTC or timezone-naive, which is interpreted
                as local time.
            source: Filter by source: 'stdout', 'stderr', or 'system'.
            search_text: Filter by search text.

        Yields:
            `LogEntry` objects in chronological order.

        Examples:

            ```python notest
            server = modal.Server.from_name("my-app", "web")

            for entry in server.logs.fetch(
                since=datetime.now() - timedelta(minutes=25),
                source="stdout",
            ):
                print(entry.message, end="")
            ```
        """
        async for log_entry in super().fetch(since=since, until=until, source=source, search_text=search_text):
            yield log_entry

    async def tail(
        self,
        entries: int = 100,
        *,
        source: LogSource | None = None,
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch the most recent Server logs.

        Args:
            entries: The number of log entries to return.
            source: Filter by source: 'stdout', 'stderr', or 'system'.

        Yields:
            `LogEntry` objects in chronological order.

        Examples:

            ```python notest
            server = modal.Server.from_name("my-app", "web")

            for entry in server.logs.tail(20):
                print(entry.message, end="")
            ```
        """
        async for log_entry in super().tail(entries, source=source):
            yield log_entry

    async def stream(self, timeout: float | None = None) -> AsyncGenerator[LogEntry, None]:
        """Stream new Server logs until the timeout is reached.

        Args:
            timeout: Number of seconds to wait between log entries before terminating the stream.
                By default, this will block until it is interrupted.

        Yields:
            `LogEntry` objects as they arrive.

        Examples:

            ```python notest
            server = modal.Server.from_name("my-app", "web")

            for entry in server.logs.stream(timeout=60):
                print(entry.message, end="")
            ```
        """
        async for log_entry in super().stream(timeout=timeout):
            yield log_entry


ServerLogsManager = synchronize_api(_ServerLogsManager, target_module=__name__)


class _FunctionCallLogsManager(_LogsManager):
    """mdmd:namespace"""

    def __init__(self, source: _SupportsLogs):
        """mdmd:hidden"""
        super().__init__(source, stop_stream=self._determine_function_call_stop)
        self._function_id: str | None = None

    async def _params(self) -> _LogQueryData:
        params = await super()._params()
        if self._function_id is None:
            self._function_id = getattr(self._source, "_function_id", None)
        return params

    async def _get_function_call_info(self) -> api_pb2.FunctionCallInfo:
        for i in range(5):
            try:
                params = await self._params()
                assert hasattr(self, "_function_id") and self._function_id is not None, (
                    "Function ID should be set during hydration"
                )
                request = api_pb2.FunctionCallGetInfoRequest(
                    function_id=self._function_id,
                    function_call_id=params.filters.function_call_id,
                )
                response = await params.client.stub.FunctionCallGetInfo(request)
                return response.info
            except NotFoundError:
                if i < 4:
                    await asyncio.sleep(1)
                    continue
                raise

    async def _determine_function_call_stop(self) -> bool:
        try:
            info = await self._get_function_call_info()
        except ResourceExhaustedError:
            # This is a best effort check. If we get rate-limited, back off but try again later.
            await asyncio.sleep(1)
            return False
        except NotFoundError:
            return False
        except (ServiceError, InternalError, StreamTerminatedError, socket.gaierror, AttributeError) as exc:
            if self._is_transient_stream_error(exc):
                logger.debug("Function call status check interrupted. Retrying ...")
                return False
            raise
        terminal_inputs = (
            info.succeeded_inputs.total
            + info.failed_inputs.total
            + info.timeout_inputs.total
            + info.cancelled_inputs.total
        )
        return info.total_inputs > 0 and terminal_inputs == info.total_inputs

    async def stream(
        self,
        timeout: float | None = None,
    ) -> AsyncGenerator[LogEntry, None]:
        """Stream new FunctionCall logs until the timeout is reached.
        The timeout specifies the number of seconds to wait between log entries before terminating the stream.
        This method will stop when the FunctionCall is observed to have completed,
        or when the timeout is reached. The completion check is best-effort; if completion
        cannot be determined, the stream will continue until the timeout is reached.

        Args:
            timeout: Number of seconds to wait between log entries before terminating the stream.
               By default, this will block until it is interrupted.

        Yields:
            `LogEntry` objects as they arrive.

        Examples:

            ```python notest
            function = modal.Function.from_name("my-app", "train")
            call = function.spawn()

            for entry in call.logs.stream():
                print(entry.message, end="")
            ```
        """
        async for log_entry in super().stream(timeout=timeout):
            yield log_entry

    async def tail(
        self,
        entries: int = 100,
        *,
        source: LogSource | None = None,
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch the most recent FunctionCall logs.

        Args:
            entries: The number of log entries to return.
            source: Filter by source: 'stdout', 'stderr', or 'system'.

        Yields:
            `LogEntry` objects in chronological order.

        Examples:

            ```python notest
            function = modal.Function.from_name("my-app", "train")
            call = function.spawn()

            for entry in call.logs.tail(entries=10):
                print(entry.timestamp, entry.message, end="")
            ```
        """
        async for log_entry in super().tail(entries, source=source):
            yield log_entry

    async def fetch(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        source: LogSource | None = None,
        search_text: str = "",
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch all associated logs corresponding to the date range and filters.

        Args:
            since: Start date to fetch logs from. Must be in UTC or timezone-naive, which is interpreted as local time.
                By default, this will fetch logs from the start of the function call.
            until: Defaults to current date if None. Must be in UTC or timezone-naive, which is interpreted
                as local time.
            source: Filter by source: 'stdout', 'stderr', or 'system'.
            search_text: Filter by search text.

        Yields:
            `LogEntry` objects in chronological order.

        Examples:

            ```python notest
            function = modal.Function.from_name("my-app", "train")
            call = function.spawn()

            for entry in call.logs.fetch():
                print(entry.timestamp, entry.message, end="")
            ```
        """
        since_was_defaulted = since is None
        until_was_defaulted = until is None

        if since_was_defaulted:
            function_call_info = await self._get_function_call_info()
            since = datetime.fromtimestamp(function_call_info.created_at, timezone.utc)

        if until_was_defaulted:
            until = datetime.now(timezone.utc)

        if since_was_defaulted and until_was_defaulted and since > until:
            until = since

        async for log_entry in super().fetch(since=since, until=until, source=source, search_text=search_text):
            yield log_entry


FunctionCallLogsManager = synchronize_api(_FunctionCallLogsManager, target_module=__name__)
