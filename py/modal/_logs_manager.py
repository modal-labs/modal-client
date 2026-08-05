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

from modal._logs import LogsFilters, fetch_logs, tail_logs
from modal._supports_logs import _LogQueryData, _SupportsImageLogs, _SupportsLogs
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


def _normalize_utc_datetime(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=locale_tz()).astimezone(timezone.utc)
    return value.astimezone(timezone.utc)


def _entry_source(file_descriptor: api_pb2.FileDescriptor.ValueType) -> LogSource:
    match file_descriptor:
        case api_pb2.FILE_DESCRIPTOR_STDOUT:
            return "stdout"
        case api_pb2.FILE_DESCRIPTOR_STDERR:
            return "stderr"
        case _:
            return "system"


def _entry_timestamp(item: api_pb2.TaskLogs) -> datetime:
    if item.timestamp_ns:
        return datetime.fromtimestamp(item.timestamp_ns / 1_000_000_000, timezone.utc)
    return datetime.fromtimestamp(item.timestamp, timezone.utc)


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


def _entry_context_ids(object_id: str, item: api_pb2.TaskLogs, batch: api_pb2.TaskLogsBatch) -> list[str]:
    match object_id[:2]:
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


def _entry_from_item(object_id: str, item: api_pb2.TaskLogs, batch: api_pb2.TaskLogsBatch) -> LogEntry:
    return LogEntry(
        message=item.data,
        timestamp=_entry_timestamp(item),
        source=_entry_source(item.file_descriptor),
        object_id=object_id,
        context_ids=_entry_context_ids(object_id, item, batch),
    )


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

    async def _watch_stream_stop(self, deadline: _Deadline) -> _StreamStopReason | None:
        while True:
            if deadline.value is not None and time.monotonic() >= deadline.value:
                return _StreamStopReason.IDLE_TIMEOUT
            if self._stop_stream is not None and await self._stop_stream():
                return _StreamStopReason.STOP_STREAM

            sleep_time: float = _STREAM_POLL_INTERVAL_SECONDS
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
        since = _normalize_utc_datetime(since, "since")
        until = _normalize_utc_datetime(until or datetime.now(timezone.utc), "until")
        target = await self._params()
        filters = dataclasses.replace(target.filters, source=_resolve_source(source), search_text=search_text)
        async for batch in fetch_logs(target.client, target.app_id, since=since, until=until, filters=filters):
            for item in batch.items:
                yield _entry_from_item(self._source.object_id, item, batch)

    async def tail(
        self,
        entries: int = 100,
        *,
        source: LogSource | None = None,
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch the most recent logs."""
        params = await self._params()
        filters = dataclasses.replace(params.filters, source=_resolve_source(source))
        async for batch in tail_logs(
            params.client,
            params.app_id,
            entries,
            filters=filters,
        ):
            for item in batch.items:
                yield _entry_from_item(self._source.object_id, item, batch)

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
                yield _entry_from_item(self._source.object_id, item, batch)

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


class _FunctionLogsManager:
    """mdmd:namespace"""

    def __init__(self, source: _SupportsLogs):
        """mdmd:hidden"""
        self._manager = _LogsManager(source)

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
            since: Start date to fetch logs from. Must be in UTC or timezone-naive,
                which is interpreted as local time.
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
        async for log_entry in self._manager.fetch(since=since, until=until, source=source, search_text=search_text):
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
        async for log_entry in self._manager.tail(entries, source=source):
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
        async for log_entry in self._manager.stream(timeout=timeout):
            yield log_entry


FunctionLogsManager = synchronize_api(_FunctionLogsManager, target_module=__name__)


class _ServerLogsManager:
    """mdmd:namespace"""

    def __init__(self, source: _SupportsLogs):
        """mdmd:hidden"""
        self._manager = _LogsManager(source)

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
            since: Start date to fetch logs from. Must be in UTC or timezone-naive,
                which is interpreted as local time.
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
        async for log_entry in self._manager.fetch(since=since, until=until, source=source, search_text=search_text):
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
        async for log_entry in self._manager.tail(entries, source=source):
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
        async for log_entry in self._manager.stream(timeout=timeout):
            yield log_entry


ServerLogsManager = synchronize_api(_ServerLogsManager, target_module=__name__)


class _FunctionCallLogsManager:
    """mdmd:namespace"""

    def __init__(self, source: _SupportsLogs):
        """mdmd:hidden"""
        self._source = source
        self._function_id: str | None = None
        self._manager = _LogsManager(source, stop_stream=self._determine_function_call_stop)

    async def _get_function_call_info(self) -> api_pb2.FunctionCallInfo:
        for i in range(5):
            try:
                params = await self._manager._params()
                if self._function_id is None:
                    self._function_id = getattr(self._source, "_function_id", None)
                assert self._function_id is not None, "Function ID should be set during hydration"
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
        raise NotFoundError(f"Function call {self._source.object_id} not found after retries.")

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
            if self._manager._is_transient_stream_error(exc):
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
        async for log_entry in self._manager.stream(timeout=timeout):
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
        async for log_entry in self._manager.tail(entries, source=source):
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
            since: Start date to fetch logs from. Must be in UTC or timezone-naive,
                which is interpreted as local time.
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

        async for log_entry in self._manager.fetch(since=since, until=until, source=source, search_text=search_text):
            yield log_entry


FunctionCallLogsManager = synchronize_api(_FunctionCallLogsManager, target_module=__name__)


class _ImageLogsManager:
    """mdmd:namespace"""

    _MAX_CONCURRENT_FETCHES = 10

    def __init__(self, source: _SupportsImageLogs):
        """mdmd:hidden"""
        self._source = source

    async def fetch(
        self,
        layers: int | None = 1,
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch logs for the most recent Image build steps.

        Args:
            layers: The number of build layers to fetch, counting backward
                from the final Image. If None, logs are fetched for all build steps.
        """
        params = await self._source._get_log_query_data()
        if layers is None:
            layer_count = len(params.build_steps)
        elif layers < 1:
            raise ValueError("layers must be at least 1")
        else:
            layer_count = min(layers, len(params.build_steps))

        selected_layers = params.build_steps[-layer_count:]

        semaphore = asyncio.Semaphore(self._MAX_CONCURRENT_FETCHES)
        queues = [asyncio.Queue(maxsize=100) for _ in selected_layers]
        DONE = object()

        async def _bounded_fetch(step: api_pb2.ImageBuildStep, queue: asyncio.Queue[LogEntry | object]) -> None:
            cancelled = False
            # Awkward but intentional double try/except to track whether the task was cancelled.
            # In Python 3.11+ we can use Task.cancelling().
            try:
                try:
                    async with semaphore:
                        async for batch in fetch_logs(
                            params.client,
                            step.builder_app_id,
                            # Protobuf timestamps represent UTC time.
                            since=step.started_at.ToDatetime(tzinfo=timezone.utc),
                            until=step.finished_at.ToDatetime(tzinfo=timezone.utc),
                            filters=LogsFilters(
                                task_id=step.builder_task_id,
                            ),
                        ):
                            for item in batch.items:
                                await queue.put(_entry_from_item(params.object_id, item, batch))
                except Exception as e:
                    await queue.put(e)
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                if not cancelled:
                    await queue.put(DONE)

        tasks = [asyncio.create_task(_bounded_fetch(s, q)) for s, q in zip(selected_layers, queues)]

        try:
            for q in queues:
                while (item := await q.get()) is not DONE:
                    if isinstance(item, Exception):
                        raise item
                    yield item
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def tail(
        self,
        entries: int = 100,
    ) -> AsyncGenerator[LogEntry, None]:
        """Fetch the most recent Image logs.

        Args:
            entries: The number of log entries to return.
        """
        params = await self._source._get_log_query_data()
        remaining_entries = entries
        entries_by_step: list[list[LogEntry]] = []

        for step in reversed(params.build_steps):
            if remaining_entries <= 0:
                break

            step_entries: list[LogEntry] = []
            async for batch in tail_logs(
                params.client,
                step.builder_app_id,
                remaining_entries,
                since=step.started_at.ToDatetime(tzinfo=timezone.utc),
                until=step.finished_at.ToDatetime(tzinfo=timezone.utc),
                filters=LogsFilters(task_id=step.builder_task_id),
            ):
                for item in batch.items:
                    if len(step_entries) >= remaining_entries:
                        break
                    step_entries.append(_entry_from_item(params.object_id, item, batch))

            if step_entries:
                entries_by_step.append(step_entries)
                remaining_entries -= len(step_entries)

        # Steps are queried newest-to-oldest so we can stop as soon as we have
        # enough entries, but results are exposed in chronological build order.
        for step_entries in reversed(entries_by_step):
            for entry in step_entries:
                yield entry


ImageLogsManager = synchronize_api(_ImageLogsManager, target_module=__name__)
