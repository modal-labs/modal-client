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
from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors
from .client import _Client
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


class _StreamReaderV1(Generic[T]):
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
        async for message in self._get_logs():
            if message is None:
                break
            if self._text:
                data_str += message.decode("utf-8")
            else:
                data_bytes += message

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


class _StreamReaderV2(Generic[T]):
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
        tunnel_url: str = None,
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
        self._tunnel_url = tunnel_url

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
        async for message in self._get_logs():
            if message is None:
                break
            if self._text:
                data_str += message.decode("utf-8")
            else:
                data_bytes += message

        if self._text:
            return cast(T, data_str)
        else:
            return cast(T, data_bytes)

    async def _get_logs(self) -> AsyncGenerator[Optional[bytes], None]:
        # Support stdout and stderr via the sandbox daemon tunnel.
        if self._file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            stream_name = "stdout"
        elif self._file_descriptor == api_pb2.FILE_DESCRIPTOR_STDERR:
            stream_name = "stderr"
        else:
            # Unsupported descriptor for the tunnel-backed reader.
            yield None
            return

        import httpx

        read_url = f"{self._tunnel_url}/exec/{stream_name}/read"
        drain_url = f"{self._tunnel_url}/exec/{stream_name}/drain"

        # Tracks the next offset we need to request from the server.
        offset = 0

        # Queue for offsets that have been fully processed by the caller and need to be ack-ed via /drain.
        ack_queue: asyncio.Queue[Optional[int]] = asyncio.Queue()

        async def _send_acks(client: "httpx.AsyncClient") -> None:
            """Background coroutine that streams acknowledged offsets to the daemon.

            It is fault-tolerant: if the drain connection drops, it will reconnect
            and re-emit the latest offset so the server can continue draining.
            """

            latest_sent: int | None = None
            done = False

            while not done:

                async def _ack_gen():
                    nonlocal latest_sent, done
                    # First (re)send the most recent offset so the server doesn't miss it.
                    if latest_sent is not None:
                        yield f"{latest_sent}\n".encode()

                    while True:
                        off = await ack_queue.get()
                        if off is None:
                            done = True
                            yield f"{off}\n".encode()
                            return
                        latest_sent = off
                        yield f"{off}\n".encode()

                try:
                    async with client.stream("POST", drain_url, data=_ack_gen()) as resp:
                        # Wait until the server closes the connection or an error happens.
                        await resp.aread()
                except (httpx.HTTPError, httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError):
                    # Transient error – retry after short delay. Keep `latest_sent` so we can
                    # re-ack once we reconnect.
                    if done:
                        break
                    await asyncio.sleep(1.0)
                    continue
                # Normal EOF (exec finished): loop will exit because `done` should be True after sentinel.

        async with httpx.AsyncClient(http2=True, verify=False, timeout=None) as client:
            # Fire-and-forget task that keeps the drain connection open.
            drain_task = asyncio.create_task(_send_acks(client))

            # Continue attempting to read until the server signals EOF (empty response body) or an error occurs
            # after which we retry with the current offset.
            while True:
                try:
                    async with client.stream(
                        "POST",
                        read_url,
                        json={
                            "exec_id": self._object_id,
                            "timeout": 15,
                            "offset": offset,
                        },
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            if not chunk:
                                continue
                            offset += len(chunk)
                            # Send ack (non-blocking – worst case it waits until queue space available)
                            await ack_queue.put(offset)
                            yield chunk
                    # If the server cleanly closes the stream, we're done – send sentinel None.
                    break
                except (httpx.HTTPError, httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError):
                    # Transient network error – retry with the current offset after a short delay.
                    await asyncio.sleep(1.0)
                    continue

            # Signal the drain task to finish and wait for it.
            await ack_queue.put(None)
            await drain_task

        # Emit EOF sentinel for the outer reader helpers.
        yield None

    async def _get_logs_by_line(self) -> AsyncGenerator[Optional[bytes], None]:
        async for message in self._get_logs():
            if message is None:
                if self._line_buffer:
                    yield self._line_buffer
                    self._line_buffer = b""
                yield None
            else:
                self._line_buffer += message
                while b"\n" in self._line_buffer:
                    line, self._line_buffer = self._line_buffer.split(b"\n", 1)
                    yield line + b"\n"

    def _ensure_stream(self):
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

    _impl: _StreamReaderV1[T] | _StreamReaderV2[T]

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
        impl: str = "v1",  # TODO: Make this an enum, or swap some other way.
        tunnel_url: Optional[str] = None,
    ) -> None:
        """mdmd:hidden"""
        if impl == "v1":
            self._impl = _StreamReaderV1(
                file_descriptor, object_id, object_type, client, stream_type, text, by_line, deadline
            )
        elif impl == "v2":
            self._impl = _StreamReaderV2(
                file_descriptor,
                object_id,
                object_type,
                client,
                stream_type,
                text,
                by_line,
                deadline,
                tunnel_url,
            )
        else:
            raise InvalidError(f"Invalid implementation: {impl}")

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

    def __aiter__(self) -> AsyncIterator[T]:
        """mdmd:hidden"""
        return self._impl.__aiter__()

    async def __anext__(self) -> T:
        """mdmd:hidden"""
        return await self._impl.__anext__()

    async def aclose(self):
        """mdmd:hidden"""
        await self._impl.aclose()


MAX_BUFFER_SIZE = 2 * 1024 * 1024


class _StreamWriter:
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


StreamReader = synchronize_api(_StreamReader)
StreamWriter = synchronize_api(_StreamWriter)
