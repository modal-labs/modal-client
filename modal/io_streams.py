# Copyright Modal Labs 2022
import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
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

from modal.exception import ClientClosed, InvalidError
from modal_proto import api_pb2, sandbox_router_pb2 as sr_pb2

from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors
from ._utils.sandbox_utils import SandboxRouterServiceClient
from .client import _Client
from .config import logger
from .stream_type import StreamType

if TYPE_CHECKING:
    pass


async def _decode_bytes_stream_to_str(stream: AsyncGenerator[Optional[bytes], None]) -> AsyncGenerator[str, None]:
    async for item in stream:
        yield item.decode("utf-8")


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

        # line-buffering is only supported when text=True
        if by_line and not text:
            raise ValueError("line-buffering is only supported when text=True")

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
        """Fetch the entire contents of the stream until EOF.

        **Usage**

        ```python fixture:running_app
        from modal import Sandbox

        sandbox = Sandbox.create("echo", "hello", app=running_app)
        sandbox.wait()

        print(sandbox.stdout.read())
        ```
        """
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

    def __aiter__(self) -> AsyncIterator[T]:
        """mdmd:hidden"""
        self._ensure_stream()
        return self

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


class _StreamReaderDirect(Generic[T]):
    """
    Placeholder StreamReader implementation that will read directly from the worker
    that hosts the sandbox.
    """

    def __init__(
        self,
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
        object_id: str,
        object_type: Literal["sandbox"],
        router_client: SandboxRouterServiceClient,
        # TODO(saltzm): We should probably just construct a different kind of dummy object
        # if the user has DEVNULL for the stream_type.
        stream_type: StreamType = StreamType.PIPE,
        text: bool = True,
        by_line: bool = False,
        deadline: Optional[float] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self._file_descriptor = file_descriptor
        self._object_type = object_type
        self._object_id = object_id
        self._router_client = router_client
        self._stream_type = stream_type
        self._text = text
        self._by_line = by_line
        self._deadline = deadline
        self.eof = False
        self._task_id = task_id or ""

    @property
    def file_descriptor(self) -> int:
        return self._file_descriptor

    async def read(self) -> T:
        if self._text:
            data_str = ""
            async for part in self:
                data_str += cast(str, part)
            self.eof = True
            return cast(T, data_str)
        else:
            data_bytes = b""
            async for part in self:
                data_bytes += cast(bytes, part)
            self.eof = True
            return cast(T, data_bytes)

    async def _get_stdio_stream(self) -> AsyncGenerator[Optional[bytes], None]:
        """Stream raw bytes from the router client, yielding None at EOF.

        This mirrors _get_logs() semantics for the through-server implementation.
        """
        if self._stream_type != StreamType.PIPE:
            raise InvalidError("Logs can only be retrieved using the PIPE stream type.")

        offset = 0
        # Select the appropriate stream based on the file descriptor and current offset
        if self._file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            stream = self._router_client.exec_stdout_read(self._task_id, self._object_id, offset=offset)
        else:
            stream = self._router_client.exec_stderr_read(self._task_id, self._object_id, offset=offset)

        async for item in stream:
            # TODO(saltzm): Figure out origin of "liveness" comment in other implementation.
            if len(item.data) == 0:
                # This is an error.
                raise ValueError("Received empty message streaming stdio from sandbox.")

            offset += len(item.data)
            yield item.data

    async def _get_stdio_stream_by_line(self) -> AsyncGenerator[Optional[bytes], None]:
        """Yield complete lines only (ending with \n), buffering partial lines until complete."""
        line_buffer = b""
        async for message in self._get_stdio_stream():
            assert isinstance(message, bytes)
            line_buffer += message
            while b"\n" in line_buffer:
                line, line_buffer = line_buffer.split(b"\n", 1)
                yield line + b"\n"

        if line_buffer:
            yield line_buffer

    def _ensure_stream(self) -> AsyncGenerator[Optional[bytes], None]:
        if self._stream:
            return self._stream
        if self._by_line:
            stream = self._get_stdio_stream_by_line()
        else:
            stream = self._get_stdio_stream()
        if self._text:
            self._stream = _decode_bytes_stream_to_str(stream)
        else:
            self._stream = stream

        return self._stream

    # TODO(saltzm): I sort of would prefer an API where you either do read() or as_stream() and as_stream() would
    # return a new stream object.
    # TODO(saltzm): Is it a problem I'm returning a new stream object every time?
    # def __aiter__(self) -> AsyncIterator[T]:
    #    if self._by_line:
    #        byte_stream = self._get_stdio_stream_by_line()
    #    else:
    #        byte_stream = self._get_stdio_stream()

    #    if self._text:
    #        return _decode_bytes_stream_to_str(byte_stream)
    #    else:
    #        return byte_stream

    def __aiter__(self) -> AsyncIterator[T]:
        """mdmd:hidden"""
        # TODO (saltzm): I don't know if we need this stream-caching behavior. I think now that we save
        # exec output and we can consume it more than once, this could return a new stream object every time.
        # I originally did this, and removed __anext__/aclose, but had trouble doing the same with
        # _StreamReaderThroughServer, so I'm doing this to match the behavior and allow _StreamReader to
        # implement __anext__ and aclose.
        self._ensure_stream()
        return self

    async def __anext__(self) -> T:
        """mdmd:hidden"""
        stream = self._ensure_stream()

        # This raises StopAsyncIteration if the stream is at EOF.
        value = await stream.__anext__()

        if self._text:
            return cast(T, value.decode("utf-8"))
        else:
            return cast(T, value)

    async def aclose(self):
        """mdmd:hidden"""
        if self._stream:
            await self._stream.aclose()


class _StreamReader(Generic[T]):
    """Delegating StreamReader that chooses implementation based on whether direct access
    is enabled for the sandbox."""

    def __init__(
        self,
        file_descriptor: "api_pb2.FileDescriptor.ValueType",
        object_id: str,
        object_type: Literal["sandbox", "container_process"],
        client: _Client,
        *,
        stream_type: StreamType = StreamType.PIPE,
        text: bool = True,
        by_line: bool = False,
        deadline: Optional[float] = None,
        router_client: Optional[SandboxRouterServiceClient] = None,
        task_id: Optional[str] = None,
    ) -> None:
        if router_client is None:
            self._impl = _StreamReaderThroughServer[T](
                file_descriptor,
                object_id,
                object_type,
                client,
                stream_type=stream_type,
                text=text,
                by_line=by_line,
                deadline=deadline,
            )
        else:
            self._impl = _StreamReaderDirect[T](
                file_descriptor,
                object_id,
                object_type,
                router_client,
                stream_type=stream_type,
                text=text,
                by_line=by_line,
                deadline=deadline,
                task_id=task_id,
            )

    @property
    def file_descriptor(self) -> int:
        return self._impl.file_descriptor

    async def read(self) -> T:
        return await self._impl.read()

    def __aiter__(self) -> AsyncIterator[T]:
        return self._impl.__aiter__()

    async def __anext__(self) -> T:
        return await self._impl.__anext__()

    async def aclose(self):
        await self._impl.aclose()


MAX_BUFFER_SIZE = 2 * 1024 * 1024


class _StreamWriterThroughServer:
    def __init__(self, object_id: str, object_type: Literal["sandbox", "container_process"], client: _Client) -> None:
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
        if self._is_closed:
            raise ValueError("Stdin is closed. Cannot write to it.")
        if isinstance(data, (bytes, bytearray, memoryview, str)):
            if isinstance(data, str):
                data = data.encode("utf-8")
            if len(self._buffer) + len(data) > MAX_BUFFER_SIZE:
                raise BufferError("Buffer size exceed limit. Call drain to clear the buffer.")
            self._buffer.extend(data)
        else:
            raise TypeError(f"data argument must be a bytes-like object, not {type(data).__name__}")

    def write_eof(self) -> None:
        self._is_closed = True

    async def drain(self) -> None:
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


class _StreamWriterDirect:
    def __init__(
        self,
        object_id: str,
        object_type: Literal["sandbox", "container_process"],
        router_client: SandboxRouterServiceClient,
        *,
        task_id: Optional[str] = None,
    ) -> None:
        self._object_id = object_id
        self._object_type = object_type
        self._router_client = router_client
        self._task_id = task_id or ""
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
                raise BufferError("Buffer size exceed limit. Call drain to clear the buffer.")
            self._buffer.extend(data)
        else:
            raise TypeError(f"data argument must be a bytes-like object, not {type(data).__name__}")

    def write_eof(self) -> None:
        self._is_closed = True

    async def drain(self) -> None:
        data = bytes(self._buffer)
        self._buffer.clear()
        start_offset = self._offset

        async def _gen():
            if data or self._is_closed:
                yield sr_pb2.SandboxExecStdinWriteRequest(
                    task_id=self._task_id,
                    exec_id=self._object_id,
                    offset=start_offset,
                    data=data,
                )

        await self._router_client.exec_stdin_write(_gen())
        self._offset += len(data)


class _StreamWriter:
    """Provides an interface to buffer and write logs to a sandbox or container process stream (`stdin`)."""

    def __init__(
        self,
        object_id: str,
        object_type: Literal["sandbox", "container_process"],
        client: _Client,
        *,
        router_client: Optional[SandboxRouterServiceClient] = None,
        task_id: Optional[str] = None,
    ) -> None:
        if router_client is None:
            self._impl = _StreamWriterThroughServer(object_id, object_type, client)
        else:
            self._impl = _StreamWriterDirect(object_id, object_type, router_client, task_id=task_id)

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
