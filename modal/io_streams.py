# Copyright Modal Labs 2022
import asyncio
from typing import TYPE_CHECKING, AsyncGenerator, Literal, Optional, Tuple, Union

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
    sandbox_id: str, file_descriptor: int, last_entry_id: Optional[str], client: _Client
) -> AsyncGenerator[Tuple[Optional[str], str], None]:
    req = api_pb2.SandboxGetLogsRequest(
        sandbox_id=sandbox_id,
        file_descriptor=file_descriptor,
        timeout=55,
        last_entry_id=last_entry_id,
    )
    async for log_batch in client.stub.SandboxGetLogs.unary_stream(req):
        last_entry_id = log_batch.entry_id

        for message in log_batch.items:
            yield (message.data, last_entry_id)
        if log_batch.eof:
            yield (None, last_entry_id)
            break


async def _container_process_logs_iterator(
    process_id: str, file_descriptor: int, client: _Client
) -> AsyncGenerator[Optional[str], None]:
    req = api_pb2.ContainerExecGetOutputRequest(
        exec_id=process_id,
        timeout=55,
        file_descriptor=file_descriptor,
    )
    async for batch in client.stub.ContainerExecGetOutput.unary_stream(req):
        if batch.HasField("exit_code"):
            yield None
            break
        for item in batch.items:
            yield item.message


class _StreamReader:
    """Provides an interface to buffer and fetch logs from a stream (`stdout` or `stderr`).

    As an asynchronous iterable, the object supports the async for statement.

    **Usage**

    ```python
    from modal import Sandbox

    sandbox = Sandbox.create(
        "bash",
        "-c",
        "for i in $(seq 1 10); do echo foo; sleep 0.1; done",
        app=app,
    )
    for message in sandbox.stdout:
        print(f"Message: {message}")
    ```
    """

    def __init__(
        self,
        file_descriptor: int,
        object_id: str,
        object_type: Literal["sandbox", "container_process"],
        client: _Client,
        stream_type: StreamType = StreamType.PIPE,
        by_line: bool = False,  # if True, streamed logs are further processed into complete lines.
    ) -> None:
        """mdmd:hidden"""

        self._file_descriptor = file_descriptor
        self._object_type = object_type
        self._object_id = object_id
        self._client = client
        self._stream = None
        self._last_entry_id = None
        self._line_buffer = ""
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
            self._container_process_buffer = []
            self._consume_container_process_task = asyncio.create_task(self._consume_container_process_stream())

    @property
    def file_descriptor(self):
        return self._file_descriptor

    async def read(self) -> str:
        """Fetch and return contents of the entire stream. If EOF was received,
        return an empty string.

        **Usage**

        ```python
        from modal import Sandbox

        sandbox = Sandbox.create("echo", "hello", app=app)
        sandbox.wait()

        print(sandbox.stdout.read())
        ```

        """
        data = ""
        # TODO: maybe combine this with get_app_logs_loop
        async for message in self._get_logs_by_line():
            if message is None:
                break
            data += message

        return data

    async def _consume_container_process_stream(self):
        """
        Consumes the container process stream and stores the messages in the buffer.
        """
        if self._stream_type == StreamType.DEVNULL:
            return

        completed = False
        retries_remaining = 10
        while not completed:
            try:
                iterator = _container_process_logs_iterator(self._object_id, self._file_descriptor, self._client)

                async for message in iterator:
                    if self._stream_type == StreamType.STDOUT and message:
                        print(message, end="")
                    elif self._stream_type == StreamType.PIPE:
                        self._container_process_buffer.append(message)
                    if message is None:
                        completed = True
                        break

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

    async def _stream_container_process(self) -> AsyncGenerator[Tuple[Optional[str], str], None]:
        """mdmd:hidden
        Streams the container process buffer to the reader.
        """
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

    async def _get_logs(self) -> AsyncGenerator[Optional[str], None]:
        """mdmd:hidden
        Streams sandbox or process logs from the server to the reader.

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
                    yield message
                    if message is None:
                        completed = True
                        self.eof = True

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

    async def _get_logs_by_line(self) -> AsyncGenerator[Optional[str], None]:
        """mdmd:hidden
        Processes logs from the server and yields complete lines only.
        """
        async for message in self._get_logs():
            if message is None:
                if self._line_buffer:
                    yield self._line_buffer
                    self._line_buffer = ""
                yield None
            else:
                self._line_buffer += message
                while "\n" in self._line_buffer:
                    line, self._line_buffer = self._line_buffer.split("\n", 1)
                    yield line + "\n"

    def __aiter__(self):
        """mdmd:hidden"""
        if not self._stream:
            if self._by_line:
                self._stream = self._get_logs_by_line()
            else:
                self._stream = self._get_logs()
        return self

    async def __anext__(self):
        """mdmd:hidden"""
        value = await self._stream.__anext__()

        # The stream yields None if it receives an EOF batch.
        if value is None:
            raise StopAsyncIteration

        return value


MAX_BUFFER_SIZE = 2 * 1024 * 1024


class _StreamWriter:
    """Provides an interface to buffer and write logs to a sandbox or container process stream (`stdin`)."""

    def __init__(self, object_id: str, object_type: Literal["sandbox", "container_process"], client: _Client):
        self._index = 1
        self._object_id = object_id
        self._object_type = object_type
        self._client = client
        self._is_closed = False
        self._buffer = bytearray()

    def get_next_index(self):
        """mdmd:hidden"""
        index = self._index
        self._index += 1
        return index

    def write(self, data: Union[bytes, bytearray, memoryview, str]):
        """
        Writes data to stream's internal buffer, but does not drain/flush the write.

        This method needs to be used along with the `drain()` method which flushes the buffer.

        **Usage**

        ```python
        from modal import Sandbox

        sandbox = Sandbox.create(
            "bash",
            "-c",
            "while read line; do echo $line; done",
            app=app,
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

    def write_eof(self):
        """
        Closes the write end of the stream after the buffered write data is drained.
        If the process was blocked on input, it will become unblocked after `write_eof()`.

        This method needs to be used along with the `drain()` method which flushes the EOF to the process.
        """
        self._is_closed = True

    async def drain(self):
        """
        Flushes the write buffer to the running process. Flushes the EOF if the writer is closed.
        """
        data = bytes(self._buffer)
        self._buffer.clear()
        index = self.get_next_index()

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
