# Copyright Modal Labs 2022
import asyncio
import codecs
import contextlib
import io
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Generic,
    TextIO,
    TypeVar,
    cast,
)

from grpclib.exceptions import StreamTerminatedError

from modal.exception import ExecTimeoutError, InvalidError
from modal_proto import api_pb2

from ._utils.async_utils import aclosing, synchronize_api, synchronizer
from ._utils.task_command_router_client import TaskCommandRouterClient
from .client import _Client
from .config import logger
from .exception import ConflictError, InternalError, ServiceError
from .stream_type import StreamType

if TYPE_CHECKING:
    pass


async def _sandbox_logs_iterator(
    sandbox_id: str, file_descriptor: "api_pb2.FileDescriptor.ValueType", last_entry_id: str, client: _Client
) -> AsyncGenerator[tuple[bytes | None, str], None]:
    req = api_pb2.SandboxGetLogsRequest(
        sandbox_id=sandbox_id,
        file_descriptor=file_descriptor,
        timeout=55,
        last_entry_id=last_entry_id,
    )
    async for log_batch in client.stub.SandboxGetLogs.unary_stream(req):
        last_entry_id = log_batch.entry_id

        for message in log_batch.items:
            yield (message.data.encode("utf-8"), last_entry_id)
        if log_batch.eof:
            yield (None, last_entry_id)
            break


T = TypeVar("T", str, bytes)


class _StreamReaderThroughServer(Generic[T]):
    """A StreamReader implementation that reads sandbox logs from the server."""

    _stream: AsyncGenerator[T, None] | None

    def __init__(
        self,
        params: "_StreamReaderThroughServerParams",
        text: bool = True,
        by_line: bool = False,
    ) -> None:
        """mdmd:hidden"""
        self._file_descriptor = params.file_descriptor
        self._object_id = params.object_id
        self._client = params.client
        self._stream = None
        self._last_entry_id: str = ""
        self._line_buffer = b""

        # Sandbox logs are streamed to the client as strings, so StreamReaders reading
        # them must have text mode enabled.
        if not text:
            raise ValueError("Sandbox streams must have text mode enabled.")

        self._text = text
        self._by_line = by_line

        # Whether the reader received an EOF. Once EOF is True, it returns
        # an empty string for any subsequent reads (including async for)
        self.eof = False

    @property
    def file_descriptor(self) -> int:
        """Possible values are `1` for stdout and `2` for stderr."""
        return self._file_descriptor

    async def read(self) -> T:
        """Fetch the entire contents of the stream until EOF."""
        logger.debug(f"{self._object_id} StreamReader fd={self._file_descriptor} read starting")
        if self._text:
            buffer = io.StringIO()
            async for message in _decode_bytes_stream_to_str(self._get_logs()):
                buffer.write(message)
            logger.debug(f"{self._object_id} StreamReader fd={self._file_descriptor} read completed after EOF")
            return cast(T, buffer.getvalue())
        else:
            buffer = io.BytesIO()
            async for message in self._get_logs():
                buffer.write(message)
            logger.debug(f"{self._object_id} StreamReader fd={self._file_descriptor} read completed after EOF")
            return cast(T, buffer.getvalue())

    async def _get_logs(self, skip_empty_messages: bool = True) -> AsyncGenerator[bytes, None]:
        """Streams sandbox logs from the server to the reader.

        Logs returned by this method may contain partial or multiple lines at a time.

        When the stream receives an EOF, it yields None. Once an EOF is received,
        subsequent invocations will not yield logs.
        """
        if self.eof:
            return

        completed = False

        retries_remaining = 10
        while not completed:
            try:
                iterator = _sandbox_logs_iterator(
                    self._object_id, self._file_descriptor, self._last_entry_id, self._client
                )

                async for message, entry_id in iterator:
                    self._last_entry_id = entry_id
                    # Empty messages are sent when the process boots up. Don't yield them unless
                    # we're using the empty message to signal process liveness.
                    if skip_empty_messages and message == b"":
                        continue

                    if message is None:
                        completed = True
                        self.eof = True
                        return

                    yield message

            except (ServiceError, InternalError, StreamTerminatedError) as exc:
                if retries_remaining > 0:
                    retries_remaining -= 1
                    if isinstance(exc, (ServiceError, InternalError)):
                        await asyncio.sleep(1.0)
                        continue
                    elif isinstance(exc, StreamTerminatedError):
                        continue
                raise

    async def _get_logs_by_line(self) -> AsyncGenerator[bytes, None]:
        """Process logs from the server and yield complete lines only."""
        async for message in self._get_logs():
            assert isinstance(message, bytes)
            self._line_buffer += message
            while b"\n" in self._line_buffer:
                line, self._line_buffer = self._line_buffer.split(b"\n", 1)
                yield line + b"\n"

        if self._line_buffer:
            yield self._line_buffer
            self._line_buffer = b""

    def __aiter__(self) -> AsyncGenerator[T, None]:
        if not self._stream:
            if self._by_line:
                # TODO: This is quite odd - it does line buffering in binary mode
                # but we then always add the buffered text decoding on top of that.
                # feels a bit upside down...
                stream = self._get_logs_by_line()
            else:
                stream = self._get_logs()
            if self._text:
                stream = _decode_bytes_stream_to_str(stream)
            self._stream = cast(AsyncGenerator[T, None], stream)
        return self._stream

    async def aclose(self):
        """mdmd:hidden"""
        if self._stream:
            await self._stream.aclose()


async def _decode_bytes_stream_to_str(stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
    """Incrementally decode a bytes async generator as UTF-8 without breaking on chunk boundaries.

    This function uses a streaming UTF-8 decoder so that multi-byte characters split across
    chunks are handled correctly instead of raising ``UnicodeDecodeError``.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    async for item in stream:
        text = decoder.decode(item, final=False)
        if text:
            yield text

    # Flush any buffered partial character at end-of-stream
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail


async def _stream_by_line(stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    """Yield complete lines only (ending with \n), buffering partial lines until complete.

    When this generator returns, the underlying generator is closed.
    """
    line_buffer = b""
    try:
        async for message in stream:
            assert isinstance(message, bytes)
            line_buffer += message
            while b"\n" in line_buffer:
                line, line_buffer = line_buffer.split(b"\n", 1)
                yield line + b"\n"

        if line_buffer:
            yield line_buffer
    finally:
        await stream.aclose()


@dataclass(frozen=True)
class _StreamReaderThroughServerParams:
    """Parameters for a ``_StreamReader`` that reads sandbox logs through the server."""

    file_descriptor: "api_pb2.FileDescriptor.ValueType"
    object_id: str
    client: _Client


@dataclass(frozen=True)
class _StreamReaderThroughSandboxExecCommandRouterParams:
    """Parameters for a ``_StreamReader`` that reads sandbox-exec stdio
    directly from the worker via the task command router (``exec_stdio_read``)."""

    file_descriptor: "api_pb2.FileDescriptor.ValueType"
    task_id: str
    object_id: str
    command_router_client: TaskCommandRouterClient
    deadline: float | None


@dataclass(frozen=True)
class _StreamReaderThroughSandboxCommandRouterParams:
    """Parameters for a ``_StreamReader`` that reads a V2 sandbox's
    stdio directly from the worker via the task command router
    (``sandbox_stdio_read``)."""

    file_descriptor: "api_pb2.FileDescriptor.ValueType"
    sandbox_id: str
    # Lazily fetches ``(task_id, command_router_client)`` the first time the
    # stream is iterated. Captures the sandbox handle so we only mint a JWT
    # and open a connection to the worker when stdio is actually read.
    resolve_router: Callable[[], Awaitable[tuple[str, TaskCommandRouterClient]]]


_StreamReaderThroughCommandRouterParams = (
    _StreamReaderThroughSandboxExecCommandRouterParams | _StreamReaderThroughSandboxCommandRouterParams
)


async def _stdio_stream_from_sandbox_command_router(
    params: _StreamReaderThroughSandboxCommandRouterParams,
) -> AsyncGenerator[bytes, None]:
    """Stream raw bytes from a V2 sandbox's primary stdio via ``sandbox_stdio_read``."""
    task_id, command_router_client = await params.resolve_router()
    first_chunk = True
    async with aclosing(command_router_client.sandbox_stdio_read(task_id, params.file_descriptor)) as stream:
        async for item in stream:
            if len(item.data) == 0:
                raise ValueError("Received empty message streaming stdio from sandbox.")
            if first_chunk:
                first_chunk = False
                if item.starting_offset > 0:
                    logger.warning(
                        f"V2 sandbox {params.sandbox_id} stdio: dropped first "
                        f"{item.starting_offset} bytes; only the most recent portion "
                        f"of output is retained."
                    )
            yield item.data


async def _stdio_stream_from_sandbox_exec_command_router(
    params: _StreamReaderThroughSandboxExecCommandRouterParams,
) -> AsyncGenerator[bytes, None]:
    """Stream raw bytes from a V2 sandbox-exec'd process via ``exec_stdio_read``."""
    async with aclosing(
        params.command_router_client.exec_stdio_read(
            params.task_id, params.object_id, params.file_descriptor, params.deadline
        )
    ) as stream:
        try:
            async for item in stream:
                if len(item.data) == 0:
                    raise ValueError("Received empty message streaming stdio from sandbox.")
                yield item.data
        except ExecTimeoutError:
            logger.debug(f"Deadline exceeded while streaming stdio for exec {params.object_id}")
            # TODO(saltzm): This is a weird API, but customers currently may rely on it. We
            # should probably raise this error rather than just ending the stream.
            return


def _stdio_stream_from_command_router(
    params: _StreamReaderThroughCommandRouterParams,
) -> AsyncGenerator[bytes, None]:
    """Dispatch between the V2-sandbox primary stdio and the V2-sandbox-exec
    stdio streams, both of which yield raw bytes."""
    if isinstance(params, _StreamReaderThroughSandboxCommandRouterParams):
        return _stdio_stream_from_sandbox_command_router(params)
    return _stdio_stream_from_sandbox_exec_command_router(params)


class _BytesStreamReaderThroughCommandRouter:
    """StreamReader that yields raw bytes from the router-backed stdio source
    (either V2 sandbox top-level stdio or V2 sandbox-exec stdio)."""

    def __init__(self, params: _StreamReaderThroughCommandRouterParams) -> None:
        self._params = params

    @property
    def file_descriptor(self) -> int:
        return self._params.file_descriptor

    async def read(self) -> bytes:
        buffer = io.BytesIO()
        async for part in self:
            buffer.write(part)
        return buffer.getvalue()

    def __aiter__(self) -> AsyncGenerator[bytes, None]:
        return _stdio_stream_from_command_router(self._params)

    async def _print_all(self, output_stream: TextIO) -> None:
        async for part in self:
            output_stream.buffer.write(part)
            output_stream.buffer.flush()


class _TextStreamReaderThroughCommandRouter:
    """StreamReader that yields UTF-8-decoded text from the router-backed
    stdio source."""

    def __init__(self, params: _StreamReaderThroughCommandRouterParams, by_line: bool) -> None:
        self._params = params
        self._by_line = by_line

    @property
    def file_descriptor(self) -> int:
        return self._params.file_descriptor

    async def read(self) -> str:
        buffer = io.StringIO()
        async for part in self:
            buffer.write(part)
        return buffer.getvalue()

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        async with aclosing(_stdio_stream_from_command_router(self._params)) as bytes_stream:
            if self._by_line:
                stream = _decode_bytes_stream_to_str(_stream_by_line(bytes_stream))
            else:
                stream = _decode_bytes_stream_to_str(bytes_stream)

            async with aclosing(stream):
                async for part in stream:
                    yield part

    async def _print_all(self, output_stream: TextIO) -> None:
        async with aclosing(self.__aiter__()) as stream:
            async for part in stream:
                output_stream.write(part)


class _StdoutPrintingStreamReaderThroughCommandRouter(Generic[T]):
    """
    StreamReader implementation for StreamType.STDOUT when using the task command router.

    This mirrors the behavior from the server-backed implementation: the stream is printed to
    the local stdout immediately and is not readable via StreamReader methods.
    """

    _reader: _TextStreamReaderThroughCommandRouter | _BytesStreamReaderThroughCommandRouter

    def __init__(
        self,
        reader: _TextStreamReaderThroughCommandRouter | _BytesStreamReaderThroughCommandRouter,
    ) -> None:
        self._reader = reader
        self._task: asyncio.Task[None] | None = None
        # Kick off a background task that reads from the underlying text stream and prints to stdout.
        self._start_printing_task()

    @property
    def file_descriptor(self) -> int:
        return self._reader.file_descriptor

    def _start_printing_task(self) -> None:
        async def _run():
            try:
                await self._reader._print_all(sys.stdout)
            except Exception as e:
                logger.exception(f"Error printing stream: {e}")

        self._task = asyncio.create_task(_run())

    async def read(self) -> T:
        raise InvalidError("Output can only be retrieved using the PIPE stream type.")

    def __aiter__(self) -> AsyncIterator[T]:
        raise InvalidError("Output can only be retrieved using the PIPE stream type.")

    async def __anext__(self) -> T:
        raise InvalidError("Output can only be retrieved using the PIPE stream type.")

    async def aclose(self):
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


class _DevnullStreamReader(Generic[T]):
    """StreamReader implementation for a stream configured with
    StreamType.DEVNULL. Throws an error if read or any other method is
    called.
    """

    def __init__(self, file_descriptor: "api_pb2.FileDescriptor.ValueType") -> None:
        self._file_descriptor = file_descriptor

    @property
    def file_descriptor(self) -> int:
        return self._file_descriptor

    async def read(self) -> T:
        raise ValueError("read is not supported for a stream configured with StreamType.DEVNULL")

    def __aiter__(self) -> AsyncIterator[T]:
        raise ValueError("__aiter__ is not supported for a stream configured with StreamType.DEVNULL")

    async def __anext__(self) -> T:
        raise ValueError("__anext__ is not supported for a stream configured with StreamType.DEVNULL")

    async def aclose(self):
        raise ValueError("aclose is not supported for a stream configured with StreamType.DEVNULL")


class _StreamReader(Generic[T]):
    """Retrieve logs from a stream (`stdout` or `stderr`).

    As an asynchronous iterable, the object supports the `for` and `async for`
    statements. Just loop over the object to read in chunks.
    """

    _impl: (
        _StreamReaderThroughServer
        | _DevnullStreamReader
        | _TextStreamReaderThroughCommandRouter
        | _BytesStreamReaderThroughCommandRouter
        | _StdoutPrintingStreamReaderThroughCommandRouter
    )
    _read_gen: AsyncGenerator[T, None] | None = None

    def __init__(
        self,
        params: _StreamReaderThroughServerParams | _StreamReaderThroughCommandRouterParams,
        *,
        stream_type: StreamType = StreamType.PIPE,
        text: bool = True,
        by_line: bool = False,
    ) -> None:
        """mdmd:hidden"""
        # we can remove this once we ensure no constructors use async code
        assert asyncio.get_running_loop() == synchronizer._get_loop(start=False)

        if by_line and not text:
            raise ValueError("line-buffering is only supported when text=True")

        if isinstance(params, _StreamReaderThroughCommandRouterParams):
            if stream_type == StreamType.DEVNULL:
                self._impl = _DevnullStreamReader(params.file_descriptor)
            else:
                assert stream_type == StreamType.PIPE or stream_type == StreamType.STDOUT
                if text:
                    reader = _TextStreamReaderThroughCommandRouter(params, by_line)
                else:
                    reader = _BytesStreamReaderThroughCommandRouter(params)

                if stream_type == StreamType.STDOUT:
                    self._impl = _StdoutPrintingStreamReaderThroughCommandRouter(reader)
                else:
                    self._impl = reader
        else:
            # Sandbox logs are read via the server.
            self._impl = _StreamReaderThroughServer(params, text, by_line)

    @property
    def file_descriptor(self) -> int:
        """Possible values are `1` for stdout and `2` for stderr."""
        return self._impl.file_descriptor

    async def read(self) -> T:
        """Fetch the entire contents of the stream until EOF."""
        return cast(T, await self._impl.read())

    def __aiter__(self) -> AsyncGenerator[T, None]:
        if not self._read_gen:
            self._read_gen = cast(AsyncGenerator[T, None], self._impl.__aiter__())
        return self._read_gen

    async def __anext__(self) -> T:
        """Deprecated: This exists for backwards compatibility and will be removed in a future version of Modal

        Only use next/anext on the return value of iter/aiter on the StreamReader object (treat streamreader as
        an iterable, not an iterator).
        """
        if not self._read_gen:
            self.__aiter__()  # initialize the read generator
        assert self._read_gen
        return await self._read_gen.__anext__()

    async def aclose(self):
        """mdmd:hidden"""
        if self._read_gen:
            await self._read_gen.aclose()
            self._read_gen = None


MAX_BUFFER_SIZE = 2 * 1024 * 1024
# Larger buffer limit for the exec path via the task command router.
# This applies only to task_exec_stdin_write; sandbox stdin via the server keeps MAX_BUFFER_SIZE.
TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE: int = 16 * 1024 * 1024


@dataclass(frozen=True)
class _StreamWriterThroughServerParams:
    """Parameters for a ``_StreamWriter`` that writes sandbox stdin through the server."""

    object_id: str
    client: _Client


@dataclass(frozen=True)
class _StreamWriterThroughCommandRouterSandboxExecParams:
    """Parameters for a ``_StreamWriter`` that writes the stdin of a process
    spawned via ``sb.exec(...)`` directly to the worker via the task command
    router."""

    task_id: str
    object_id: str
    command_router_client: TaskCommandRouterClient


@dataclass(frozen=True)
class _StreamWriterThroughCommandRouterSandboxParams:
    """Parameters for a ``_StreamWriter`` that writes a V2 sandbox entrypoint's
    stdin directly to the worker via the task command router.
    """

    # Lazily fetches ``(task_id, command_router_client)`` the first time the
    # writer drains. Captures the sandbox handle so we only mint a JWT and
    # open a connection to the worker when stdin is actually written.
    resolve_router: Callable[[], Awaitable[tuple[str, TaskCommandRouterClient]]]


class _StreamWriterThroughServer:
    """Provides an interface to buffer and write to a sandbox stream (`stdin`) via the server."""

    def __init__(self, params: _StreamWriterThroughServerParams) -> None:
        """mdmd:hidden"""
        self._index = 1
        self._object_id = params.object_id
        self._client = params.client
        self._is_closed = False
        self._buffer = bytearray()

    def _get_next_index(self) -> int:
        index = self._index
        self._index += 1
        return index

    def write(self, data: bytes | bytearray | memoryview | str) -> None:
        """Write data to the stream but does not send it immediately.

        This is non-blocking and queues the data to an internal buffer. Must be
        used along with the `drain()` method, which flushes the buffer.
        """
        if self._is_closed:
            raise ValueError("Stdin is closed. Cannot write to it.")
        if isinstance(data, (bytes, bytearray, memoryview, str)):
            if isinstance(data, str):
                data = data.encode("utf-8")
            if len(self._buffer) + len(data) > MAX_BUFFER_SIZE:
                raise BufferError("Buffer size exceed limit. Call drain to flush the buffer.")
            self._buffer.extend(data)
        else:
            raise TypeError(f"data argument must be a bytes-like object, not {type(data).__name__}")

    def write_eof(self) -> None:
        """Close the write end of the stream after the buffered data is drained.

        If the process was blocked on input, it will become unblocked after
        `write_eof()`. This method needs to be used along with the `drain()`
        method, which flushes the EOF to the process.
        """
        self._is_closed = True

    async def drain(self) -> None:
        """Flush the write buffer and send data to the running process.

        This is a flow control method that blocks until data is sent. It returns
        when it is appropriate to continue writing data to the stream.
        """
        data = bytes(self._buffer)
        self._buffer.clear()
        index = self._get_next_index()

        try:
            await self._client.stub.SandboxStdinWrite(
                api_pb2.SandboxStdinWriteRequest(
                    sandbox_id=self._object_id, index=index, eof=self._is_closed, input=data
                ),
            )
        except ConflictError as exc:
            raise ValueError(str(exc))


class _StreamWriterThroughCommandRouterBuffer(ABC):
    """Buffering/draining logic for stream writers that flush data
    to the task command router."""

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()
        self._is_closed: bool = False
        self._offset: int = 0

    @abstractmethod
    async def stdin_write(self, data: bytes, eof: bool) -> None:
        """Write the given chunk (with optional EOF) to the command router."""

    def write(self, data: bytes | bytearray | memoryview | str) -> None:
        if self._is_closed:
            raise ValueError("Stdin is closed. Cannot write to it.")
        if isinstance(data, (bytes, bytearray, memoryview, str)):
            if isinstance(data, str):
                data = data.encode("utf-8")
            if len(self._buffer) + len(data) > TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE:
                raise BufferError("Buffer size exceed limit. Call drain to flush the buffer.")
            self._buffer.extend(data)
        else:
            raise TypeError(f"data argument must be a bytes-like object, not {type(data).__name__}")

    def write_eof(self) -> None:
        self._is_closed = True

    async def drain(self) -> None:
        eof = self._is_closed
        # NB: There's no need to prevent writing eof twice, because the command router will ignore the second EOF.
        if self._buffer or eof:
            data = bytes(self._buffer)
            await self.stdin_write(data, eof)
            # Only clear the buffer after writing the data to the command router is successful.
            # This allows the client to retry drain() in the event of an exception (though
            # the underlying write call already retries on transient errors, so most users will
            # probably not do this).
            self._buffer.clear()
            self._offset += len(data)


class _StreamWriterThroughCommandRouterSandboxExec(_StreamWriterThroughCommandRouterBuffer):
    def __init__(self, params: _StreamWriterThroughCommandRouterSandboxExecParams) -> None:
        super().__init__()
        self._object_id = params.object_id
        self._command_router_client = params.command_router_client
        self._task_id = params.task_id

    async def stdin_write(self, data: bytes, eof: bool) -> None:
        await self._command_router_client.exec_stdin_write(
            task_id=self._task_id, exec_id=self._object_id, offset=self._offset, data=data, eof=eof
        )


class _StreamWriterThroughCommandRouterSandbox(_StreamWriterThroughCommandRouterBuffer):
    """Write a V2 sandbox entrypoint's stdin directly to the worker
    via the task command router."""

    def __init__(self, params: _StreamWriterThroughCommandRouterSandboxParams) -> None:
        super().__init__()
        self._resolve_router = params.resolve_router

    async def stdin_write(self, data: bytes, eof: bool) -> None:
        task_id, client = await self._resolve_router()
        await client.sandbox_stdin_write_v2(task_id=task_id, offset=self._offset, data=data, eof=eof)


class _StreamWriter:
    """Provides an interface to buffer and write logs to a sandbox or container process stream (`stdin`)."""

    def __init__(
        self,
        params: _StreamWriterThroughServerParams
        | _StreamWriterThroughCommandRouterSandboxExecParams
        | _StreamWriterThroughCommandRouterSandboxParams,
    ) -> None:
        """mdmd:hidden"""
        if isinstance(params, _StreamWriterThroughCommandRouterSandboxExecParams):
            self._impl = _StreamWriterThroughCommandRouterSandboxExec(params)
        elif isinstance(params, _StreamWriterThroughCommandRouterSandboxParams):
            self._impl = _StreamWriterThroughCommandRouterSandbox(params)
        else:
            self._impl = _StreamWriterThroughServer(params)

    def write(self, data: bytes | bytearray | memoryview | str) -> None:
        """Write data to the stream but does not send it immediately.

        This is non-blocking and queues the data to an internal buffer. Must be
        used along with the `drain()` method, which flushes the buffer.

        Examples:
            ```python fixture:sandbox
            proc = sandbox.exec(
                "bash",
                "-c",
                "while read line; do echo $line; done",
            )
            proc.stdin.write(b"foo\\n")
            proc.stdin.write(b"bar\\n")
            proc.stdin.write_eof()
            proc.stdin.drain()
            ```
        """
        self._impl.write(data)

    def write_eof(self) -> None:
        """Close the write end of the stream after the buffered data is drained.

        If the process was blocked on input, it will become unblocked after
        `write_eof()`. This method needs to be used along with the `drain()`
        method, which flushes the EOF to the process.
        """
        self._impl.write_eof()

    async def drain(self) -> None:
        """Flush the write buffer and send data to the running process.

        This is a flow control method that blocks until data is sent. It returns
        when it is appropriate to continue writing data to the stream.

        Examples:
            ```python notest
            writer.write(data)
            writer.drain()
            ```

            Async usage:

            ```python notest
            writer.write(data)  # not a blocking operation
            await writer.drain.aio()
            ```
        """
        await self._impl.drain()


StreamReader = synchronize_api(_StreamReader)
StreamWriter = synchronize_api(_StreamWriter)
