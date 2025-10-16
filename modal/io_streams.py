# Copyright Modal Labs 2022
import asyncio
import codecs
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
)

from grpclib import Status
from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal.exception import ClientClosed, ExecTimeoutError, InvalidError
from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors
from ._utils.task_command_router_client import TaskCommandRouterClient
from .client import _Client
from .config import logger
from .stream_type import StreamType

if TYPE_CHECKING:
    pass


async def _sandbox_logs_iterator(
    sandbox_id: str, file_descriptor: "api_pb2.FileDescriptor.ValueType", last_entry_id: str, client: _Client
) -> AsyncGenerator[tuple[Optional[bytes], str], None]:
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


async def _container_process_logs_iterator(
    process_id: str,
    file_descriptor: "api_pb2.FileDescriptor.ValueType",
    client: _Client,
    last_index: int,
    deadline: Optional[float] = None,
) -> AsyncGenerator[tuple[Optional[bytes], int], None]:
    req = api_pb2.ContainerExecGetOutputRequest(
        exec_id=process_id,
        timeout=55,
        file_descriptor=file_descriptor,
        get_raw_bytes=True,
        last_batch_index=last_index,
    )

    stream = client.stub.ContainerExecGetOutput.unary_stream(req)
    while True:
        # Check deadline before attempting to receive the next batch
        try:
            remaining = (deadline - time.monotonic()) if deadline else None
            batch = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
        except asyncio.TimeoutError:
            yield None, -1
            break
        except StopAsyncIteration:
            break
        if batch.HasField("exit_code"):
            yield None, batch.batch_index
            break
        for item in batch.items:
            yield item.message_bytes, batch.batch_index


T = TypeVar("T", str, bytes)


class _StreamReaderThroughServer(Generic[T]):
    """A StreamReader implementation that reads from the server."""

    _stream: Optional[AsyncGenerator[Optional[bytes], None]]

    def __init__(
        self,
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
        object_id: str,
        object_type: Literal["sandbox", "container_process"],
        client: _Client,
        stream_type: StreamType = StreamType.PIPE,
        text: bool = True,
        by_line: bool = False,
        deadline: Optional[float] = None,
    ) -> None:
        """mdmd:hidden"""
        self._file_descriptor = file_descriptor
        self._object_type = object_type
        self._object_id = object_id
        self._client = client
        self._stream = None
        self._last_entry_id: str = ""
        self._line_buffer = b""
        self._deadline = deadline

        # Sandbox logs are streamed to the client as strings, so StreamReaders reading
        # them must have text mode enabled.
        if object_type == "sandbox" and not text:
            raise ValueError("Sandbox streams must have text mode enabled.")

        self._text = text
        self._by_line = by_line

        # Whether the reader received an EOF. Once EOF is True, it returns
        # an empty string for any subsequent reads (including async for)
        self.eof = False

        if not isinstance(stream_type, StreamType):
            raise TypeError(f"stream_type must be of type StreamType, got {type(stream_type)}")

        # We only support piping sandbox logs because they're meant to be durable logs stored
        # on the user's application.
        if object_type == "sandbox" and stream_type != StreamType.PIPE:
            raise ValueError("Sandbox streams must be piped.")
        self._stream_type = stream_type

        if self._object_type == "container_process":
            # Container process streams need to be consumed as they are produced,
            # otherwise the process will block. Use a buffer to store the stream
            # until the client consumes it.
            self._container_process_buffer: list[Optional[bytes]] = []
            self._consume_container_process_task = asyncio.create_task(self._consume_container_process_stream())

    @property
    def file_descriptor(self) -> int:
        """Possible values are `1` for stdout and `2` for stderr."""
        return self._file_descriptor

    async def read(self) -> T:
        """Fetch the entire contents of the stream until EOF."""
        data_str = ""
        data_bytes = b""
        logger.debug(f"{self._object_id} StreamReader fd={self._file_descriptor} read starting")
        async for message in self._get_logs():
            if message is None:
                break
            if self._text:
                data_str += message.decode("utf-8")
            else:
                data_bytes += message

        logger.debug(f"{self._object_id} StreamReader fd={self._file_descriptor} read completed after EOF")
        if self._text:
            return cast(T, data_str)
        else:
            return cast(T, data_bytes)

    async def _consume_container_process_stream(self):
        """Consume the container process stream and store messages in the buffer."""
        if self._stream_type == StreamType.DEVNULL:
            return

        completed = False
        retries_remaining = 10
        last_index = 0
        while not completed:
            if self._deadline and time.monotonic() >= self._deadline:
                break
            try:
                iterator = _container_process_logs_iterator(
                    self._object_id, self._file_descriptor, self._client, last_index, self._deadline
                )
                async for message, batch_index in iterator:
                    if self._stream_type == StreamType.STDOUT and message:
                        print(message.decode("utf-8"), end="")
                    elif self._stream_type == StreamType.PIPE:
                        self._container_process_buffer.append(message)

                    if message is None:
                        completed = True
                        break
                    else:
                        last_index = batch_index

            except (GRPCError, StreamTerminatedError, ClientClosed) as exc:
                if retries_remaining > 0:
                    retries_remaining -= 1
                    if isinstance(exc, GRPCError):
                        if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                            await asyncio.sleep(1.0)
                            continue
                    elif isinstance(exc, StreamTerminatedError):
                        continue
                    elif isinstance(exc, ClientClosed):
                        # If the client was closed, the user has triggered a cleanup.
                        break
                logger.error(f"{self._object_id} stream read failure while consuming process output: {exc}")
                raise exc

    async def _stream_container_process(self) -> AsyncGenerator[tuple[Optional[bytes], str], None]:
        """Streams the container process buffer to the reader."""
        entry_id = 0
        if self._last_entry_id:
            entry_id = int(self._last_entry_id) + 1

        while True:
            if entry_id >= len(self._container_process_buffer):
                await asyncio.sleep(0.1)
                continue

            item = self._container_process_buffer[entry_id]

            yield (item, str(entry_id))
            if item is None:
                break

            entry_id += 1

    async def _get_logs(self, skip_empty_messages: bool = True) -> AsyncGenerator[Optional[bytes], None]:
        """Streams sandbox or process logs from the server to the reader.

        Logs returned by this method may contain partial or multiple lines at a time.

        When the stream receives an EOF, it yields None. Once an EOF is received,
        subsequent invocations will not yield logs.
        """
        if self._stream_type != StreamType.PIPE:
            raise InvalidError("Logs can only be retrieved using the PIPE stream type.")

        if self.eof:
            yield None
            return

        completed = False

        retries_remaining = 10
        while not completed:
            try:
                if self._object_type == "sandbox":
                    iterator = _sandbox_logs_iterator(
                        self._object_id, self._file_descriptor, self._last_entry_id, self._client
                    )
                else:
                    iterator = self._stream_container_process()

                async for message, entry_id in iterator:
                    self._last_entry_id = entry_id
                    # Empty messages are sent when the process boots up. Don't yield them unless
                    # we're using the empty message to signal process liveness.
                    if skip_empty_messages and message == b"":
                        continue

                    if message is None:
                        completed = True
                        self.eof = True
                    yield message

            except (GRPCError, StreamTerminatedError) as exc:
                if retries_remaining > 0:
                    retries_remaining -= 1
                    if isinstance(exc, GRPCError):
                        if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                            await asyncio.sleep(1.0)
                            continue
                    elif isinstance(exc, StreamTerminatedError):
                        continue
                raise

    async def _get_logs_by_line(self) -> AsyncGenerator[Optional[bytes], None]:
        """Process logs from the server and yield complete lines only."""
        async for message in self._get_logs():
            if message is None:
                if self._line_buffer:
                    yield self._line_buffer
                    self._line_buffer = b""
                yield None
            else:
                assert isinstance(message, bytes)
                self._line_buffer += message
                while b"\n" in self._line_buffer:
                    line, self._line_buffer = self._line_buffer.split(b"\n", 1)
                    yield line + b"\n"

    def _ensure_stream(self) -> AsyncGenerator[Optional[bytes], None]:
        if not self._stream:
            if self._by_line:
                self._stream = self._get_logs_by_line()
            else:
                self._stream = self._get_logs()
        return self._stream

    async def __anext__(self) -> T:
        """mdmd:hidden"""
        stream = self._ensure_stream()

        value = await stream.__anext__()

        # The stream yields None if it receives an EOF batch.
        if value is None:
            raise StopAsyncIteration

        if self._text:
            return cast(T, value.decode("utf-8"))
        else:
            return cast(T, value)

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
    """Yield complete lines only (ending with \n), buffering partial lines until complete."""
    line_buffer = b""
    async for message in stream:
        assert isinstance(message, bytes)
        line_buffer += message
        while b"\n" in line_buffer:
            line, line_buffer = line_buffer.split(b"\n", 1)
            yield line + b"\n"

    if line_buffer:
        yield line_buffer


@dataclass
class _StreamReaderThroughCommandRouterParams:
    file_descriptor: "api_pb2.FileDescriptor.ValueType"
    task_id: str
    object_id: str
    command_router_client: TaskCommandRouterClient
    deadline: Optional[float]


async def _stdio_stream_from_command_router(
    params: _StreamReaderThroughCommandRouterParams,
) -> AsyncGenerator[bytes, None]:
    """Stream raw bytes from the router client."""
    stream = params.command_router_client.exec_stdio_read(
        params.task_id, params.object_id, params.file_descriptor, params.deadline
    )
    try:
        async for item in stream:
            if len(item.data) == 0:
                # This is an error.
                raise ValueError("Received empty message streaming stdio from sandbox.")

            yield item.data
    except ExecTimeoutError:
        logger.debug(f"Deadline exceeded while streaming stdio for exec {params.object_id}")
        # TODO(saltzm): This is a weird API, but customers currently may rely on it. We
        # should probably raise this error rather than just ending the stream.
        return


class _BytesStreamReaderThroughCommandRouter(Generic[T]):
    """
    StreamReader implementation that will read directly from the worker that
    hosts the sandbox.

    This implementation is used for non-text streams.
    """

    def __init__(
        self,
        params: _StreamReaderThroughCommandRouterParams,
    ) -> None:
        self._params = params
        self._stream = None

    @property
    def file_descriptor(self) -> int:
        return self._params.file_descriptor

    async def read(self) -> T:
        data_bytes = b""
        async for part in self:
            data_bytes += cast(bytes, part)
        return cast(T, data_bytes)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if self._stream is None:
            self._stream = _stdio_stream_from_command_router(self._params)
        # This raises StopAsyncIteration if the stream is at EOF.
        return cast(T, await self._stream.__anext__())

    async def aclose(self):
        if self._stream:
            await self._stream.aclose()


class _TextStreamReaderThroughCommandRouter(Generic[T]):
    """
    StreamReader implementation that will read directly from the worker
    that hosts the sandbox.

    This implementation is used for text streams.
    """

    def __init__(
        self,
        params: _StreamReaderThroughCommandRouterParams,
        by_line: bool,
    ) -> None:
        self._params = params
        self._by_line = by_line
        self._stream = None

    @property
    def file_descriptor(self) -> int:
        return self._params.file_descriptor

    async def read(self) -> T:
        data_str = ""
        async for part in self:
            data_str += cast(str, part)
        return cast(T, data_str)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if self._stream is None:
            bytes_stream = _stdio_stream_from_command_router(self._params)
            if self._by_line:
                self._stream = _decode_bytes_stream_to_str(_stream_by_line(bytes_stream))
            else:
                self._stream = _decode_bytes_stream_to_str(bytes_stream)
        # This raises StopAsyncIteration if the stream is at EOF.
        return cast(T, await self._stream.__anext__())

    async def aclose(self):
        if self._stream:
            await self._stream.aclose()


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

    **Usage**

    ```python fixture:running_app
    from modal import Sandbox

    sandbox = Sandbox.create(
        "bash",
        "-c",
        "for i in $(seq 1 10); do echo foo; sleep 0.1; done",
        app=running_app,
    )
    for message in sandbox.stdout:
        print(f"Message: {message}")
    ```
    """

    def __init__(
        self,
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
        object_id: str,
        object_type: Literal["sandbox", "container_process"],
        client: _Client,
        stream_type: StreamType = StreamType.PIPE,
        text: bool = True,
        by_line: bool = False,
        deadline: Optional[float] = None,
        command_router_client: Optional[TaskCommandRouterClient] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """mdmd:hidden"""
        if by_line and not text:
            raise ValueError("line-buffering is only supported when text=True")

        if command_router_client is None:
            self._impl = _StreamReaderThroughServer(
                file_descriptor, object_id, object_type, client, stream_type, text, by_line, deadline
            )
        else:
            # The only reason task_id is optional is because StreamReader is
            # also used for sandbox logs, which don't have a task ID available
            # when the StreamReader is created.
            assert task_id is not None
            assert object_type == "container_process"
            if stream_type == StreamType.DEVNULL:
                self._impl = _DevnullStreamReader(file_descriptor)
            else:
                assert stream_type == StreamType.PIPE or stream_type == StreamType.STDOUT
                # TODO(saltzm): The original implementation of STDOUT StreamType in
                # _StreamReaderThroughServer prints to stdout immediately. This doesn't match
                # python subprocess.run, which uses None to print to stdout immediately, and uses
                # STDOUT as an argument to stderr to redirect stderr to the stdout stream. We should
                # implement the old behavior here before moving out of beta, but after that
                # we should consider changing the API to match python subprocess.run. I don't expect
                # many customers are using this in any case, so I think it's fine to leave this
                # unimplemented for now.
                if stream_type == StreamType.STDOUT:
                    raise NotImplementedError(
                        "Currently only the PIPE stream type is supported when using exec "
                        "through a task command router, which is currently in beta."
                    )
                params = _StreamReaderThroughCommandRouterParams(
                    file_descriptor, task_id, object_id, command_router_client, deadline
                )
                if text:
                    self._impl = _TextStreamReaderThroughCommandRouter(params, by_line)
                else:
                    self._impl = _BytesStreamReaderThroughCommandRouter(params)

    @property
    def file_descriptor(self) -> int:
        """Possible values are `1` for stdout and `2` for stderr."""
        return self._impl.file_descriptor

    async def read(self) -> T:
        """Fetch the entire contents of the stream until EOF.

        **Usage**

        ```python fixture:running_app
        from modal import Sandbox

        sandbox = Sandbox.create("echo", "hello", app=running_app)
        sandbox.wait()

        print(sandbox.stdout.read())
        ```
        """
        return await self._impl.read()

    # TODO(saltzm): I'd prefer to have the implementation classes only implement __aiter__
    # and have them return generator functions directly, but synchronicity doesn't let us
    # return self._impl.__aiter__() here because it won't properly wrap the implementation
    # classes.
    def __aiter__(self) -> AsyncIterator[T]:
        """mdmd:hidden"""
        return self

    async def __anext__(self) -> T:
        """mdmd:hidden"""
        return await self._impl.__anext__()

    async def aclose(self):
        """mdmd:hidden"""
        await self._impl.aclose()


MAX_BUFFER_SIZE = 2 * 1024 * 1024


class _StreamWriterThroughServer:
    """Provides an interface to buffer and write logs to a sandbox or container process stream (`stdin`)."""

    def __init__(self, object_id: str, object_type: Literal["sandbox", "container_process"], client: _Client) -> None:
        """mdmd:hidden"""
        self._index = 1
        self._object_id = object_id
        self._object_type = object_type
        self._client = client
        self._is_closed = False
        self._buffer = bytearray()

    def _get_next_index(self) -> int:
        index = self._index
        self._index += 1
        return index

    def write(self, data: Union[bytes, bytearray, memoryview, str]) -> None:
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
            if self._object_type == "sandbox":
                await retry_transient_errors(
                    self._client.stub.SandboxStdinWrite,
                    api_pb2.SandboxStdinWriteRequest(
                        sandbox_id=self._object_id, index=index, eof=self._is_closed, input=data
                    ),
                )
            else:
                await retry_transient_errors(
                    self._client.stub.ContainerExecPutInput,
                    api_pb2.ContainerExecPutInputRequest(
                        exec_id=self._object_id,
                        input=api_pb2.RuntimeInputMessage(message=data, message_index=index, eof=self._is_closed),
                    ),
                )
        except GRPCError as exc:
            if exc.status == Status.FAILED_PRECONDITION:
                raise ValueError(exc.message)
            else:
                raise exc


class _StreamWriterThroughCommandRouter:
    def __init__(
        self,
        object_id: str,
        command_router_client: TaskCommandRouterClient,
        task_id: str,
    ) -> None:
        self._object_id = object_id
        self._command_router_client = command_router_client
        self._task_id = task_id
        self._is_closed = False
        self._buffer = bytearray()
        self._offset = 0

    def write(self, data: Union[bytes, bytearray, memoryview, str]) -> None:
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
        self._is_closed = True

    async def drain(self) -> None:
        eof = self._is_closed
        # NB: There's no need to prevent writing eof twice, because the command router will ignore the second EOF.
        if self._buffer or eof:
            data = bytes(self._buffer)
            await self._command_router_client.exec_stdin_write(
                task_id=self._task_id, exec_id=self._object_id, offset=self._offset, data=data, eof=eof
            )
            # Only clear the buffer after writing the data to the command router is successful.
            # This allows the client to retry drain() in the event of an exception (though
            # exec_stdin_write already retries on transient errors, so most users will probably
            # not do this).
            self._buffer.clear()
            self._offset += len(data)


class _StreamWriter:
    """Provides an interface to buffer and write logs to a sandbox or container process stream (`stdin`)."""

    def __init__(
        self,
        object_id: str,
        object_type: Literal["sandbox", "container_process"],
        client: _Client,
        command_router_client: Optional[TaskCommandRouterClient] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """mdmd:hidden"""
        if command_router_client is None:
            self._impl = _StreamWriterThroughServer(object_id, object_type, client)
        else:
            assert task_id is not None
            assert object_type == "container_process"
            self._impl = _StreamWriterThroughCommandRouter(object_id, command_router_client, task_id=task_id)

    def write(self, data: Union[bytes, bytearray, memoryview, str]) -> None:
        """Write data to the stream but does not send it immediately.

        This is non-blocking and queues the data to an internal buffer. Must be
        used along with the `drain()` method, which flushes the buffer.

        **Usage**

        ```python fixture:running_app
        from modal import Sandbox

        sandbox = Sandbox.create(
            "bash",
            "-c",
            "while read line; do echo $line; done",
            app=running_app,
        )
        sandbox.stdin.write(b"foo\\n")
        sandbox.stdin.write(b"bar\\n")
        sandbox.stdin.write_eof()

        sandbox.stdin.drain()
        sandbox.wait()
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

        **Usage**

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
