# Copyright Modal Labs 2024
import io
from typing import AsyncIterator, Optional, Union, cast

from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from .client import _Client
from .exception import FilesystemExecutionError


# The Sandbox file handling API is designed to mimic Python's io.FileIO
# See https://github.com/python/cpython/blob/main/Lib/_pyio.py#L1459
# Unlike io.FileIO, it also implements some higher level APIs, like `delete_bytes` and `write_replace_bytes`.
class _FileIO:
    _binary = False
    _readable = False
    _writable = False
    _appended = False
    _closed = True

    _task_id: Optional[str] = None
    _file_descriptor: Optional[str] = None
    _client: Optional[_Client] = None

    def _validate_mode(self, mode: str) -> None:
        if not any(char in mode for char in "rwax"):
            raise ValueError(f"Invalid file mode: {mode}")

        self._readable = "r" in mode or "+" in mode
        self._writable = "w" in mode or "a" in mode or "x" in mode or "+" in mode
        self._appended = "a" in mode
        self._binary = "b" in mode

        valid_chars = set("rwaxb+")
        if any(char not in valid_chars for char in mode):
            raise ValueError(f"Invalid file mode: {mode}")

        seen_chars = set()
        for char in mode:
            if char in seen_chars:
                raise ValueError(f"Invalid file mode: {mode}")
            seen_chars.add(char)

    def _handle_error(self, error: api_pb2.SystemErrorMessage) -> None:
        raise FilesystemExecutionError(error.error_message)

    async def _consume_output(self, exec_id: str) -> AsyncIterator[Optional[str]]:
        req = api_pb2.ContainerFilesystemExecGetOutputRequest(
            exec_id=exec_id,
            timeout=55,
            last_batch_index=0,
        )
        async for batch in self._client.stub.ContainerFilesystemExecGetOutput.unary_stream(req):
            if batch.eof:
                yield None
                break
            if batch.HasField("error"):
                self._handle_error(batch.error)
            for message in batch.output:
                yield message

    async def _wait(self, exec_id: str) -> Union[bytes, str]:
        output = ""
        async for data in self._consume_output(exec_id):
            if data is None:
                break
            output += data
        if self._binary:
            return output.encode("utf-8")
        return output

    def _validate_type(self, data: Union[bytes, str]) -> None:
        if self._binary and isinstance(data, str):
            raise TypeError("Expected bytes when in binary mode")
        if not self._binary and isinstance(data, bytes):
            raise TypeError("Expected str when in text mode")

    async def _open_file(self, path: str, mode: str) -> None:
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_open_request=api_pb2.ContainerFileOpenRequest(path=path, mode=mode),
                task_id=self._task_id,
            )
        )
        if not resp.HasField("file_descriptor"):
            raise FilesystemExecutionError("Failed to open file")
        self._file_descriptor = resp.file_descriptor
        await self._wait(resp.exec_id)

    @classmethod
    async def create(cls, path: str, mode: str, client: _Client, task_id: str) -> "_FileIO":
        self = cls.__new__(cls)
        self._validate_mode(mode)
        self._client = client
        self._task_id = task_id
        await self._open_file(path, mode)
        self._closed = False
        return self

    async def read(self, n: Union[int, None] = None) -> Union[bytes, str]:
        """Read n bytes from the current position, or the entire remaining file if n is None."""
        self._check_closed()
        self._check_readable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_read_request=api_pb2.ContainerFileReadRequest(file_descriptor=self._file_descriptor, n=n),
                task_id=self._task_id,
            )
        )
        return await self._wait(resp.exec_id)

    async def readline(self) -> Union[bytes, str]:
        """Read a single line from the current position."""
        self._check_closed()
        self._check_readable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_read_line_request=api_pb2.ContainerFileReadLineRequest(file_descriptor=self._file_descriptor),
                task_id=self._task_id,
            )
        )
        return await self._wait(resp.exec_id)

    async def readlines(self) -> Union[list[bytes], list[str]]:
        """Read all lines from the current position."""
        self._check_closed()
        self._check_readable()
        data = await self.read()
        if self._binary:
            return cast(bytes, data).split(b"\n")
        return cast(str, data).split("\n")

    async def write(self, data: Union[bytes, str]) -> None:
        """Write data to the current position.

        Writes may not appear until the entire buffer is flushed, which
        can be done manually with `flush()` or automatically when the file is
        closed.
        """
        self._check_closed()
        self._check_writable()
        self._validate_type(data)
        if isinstance(data, str):
            data = data.encode("utf-8")
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_write_request=api_pb2.ContainerFileWriteRequest(file_descriptor=self._file_descriptor, data=data),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def flush(self) -> None:
        """Flush the buffer to disk."""
        self._check_closed()
        self._check_writable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_flush_request=api_pb2.ContainerFileFlushRequest(file_descriptor=self._file_descriptor),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    def _get_whence(self, whence: int) -> api_pb2.SeekWhence.ValueType:
        if whence == 0:
            return api_pb2.SeekWhence.SEEK_SET
        elif whence == 1:
            return api_pb2.SeekWhence.SEEK_CUR
        elif whence == 2:
            return api_pb2.SeekWhence.SEEK_END
        else:
            raise ValueError(f"Invalid whence value: {whence}")

    async def seek(self, offset: int, whence: int = 0) -> None:
        """Move to a new position in the file.

        `whence` defaults to 0 (absolute file positioning); other values are 1
        (relative to the current position) and 2 (relative to the file's end).
        """
        self._check_closed()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_seek_request=api_pb2.ContainerFileSeekRequest(
                    file_descriptor=self._file_descriptor,
                    offset=offset,
                    whence=self._get_whence(whence),
                ),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def delete_bytes(
        self, start_inclusive: Union[int, None] = None, end_exclusive: Union[int, None] = None
    ) -> None:
        """Delete a range of bytes from the file.

        `start_inclusive` and `end_exclusive` are byte offsets. If either is
        None, the start or end of the file is used, respectively.

        Resets the file pointer to the start of the file.
        """
        self._check_closed()
        self._check_writable()
        if start_inclusive is not None and end_exclusive is not None:
            if start_inclusive >= end_exclusive:
                raise ValueError("start_inclusive must be less than end_exclusive")
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_delete_bytes_request=api_pb2.ContainerFileDeleteBytesRequest(
                    file_descriptor=self._file_descriptor,
                    start_inclusive=start_inclusive,
                    end_exclusive=end_exclusive,
                ),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def write_replace_bytes(
        self, data: bytes, start_inclusive: Union[int, None] = None, end_exclusive: Union[int, None] = None
    ) -> None:
        """Replace a range of bytes in the file with new data.

        `start_inclusive` and `end_exclusive` are byte offsets. If either is
        None, the start or end of the file is used, respectively.

        Resets the file pointer to the start of the file.
        """
        self._check_closed()
        self._check_writable()
        if start_inclusive is not None and end_exclusive is not None:
            if start_inclusive >= end_exclusive:
                raise ValueError("start_inclusive must be less than end_exclusive")
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_write_replace_bytes_request=api_pb2.ContainerFileWriteReplaceBytesRequest(
                    file_descriptor=self._file_descriptor,
                    data=data,
                    start_inclusive=start_inclusive,
                    end_exclusive=end_exclusive,
                ),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def _close(self) -> None:
        # Buffer is flushed by the runner on close
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_close_request=api_pb2.ContainerFileCloseRequest(file_descriptor=self._file_descriptor),
                task_id=self._task_id,
            )
        )
        self._closed = True
        await self._wait(resp.exec_id)

    async def close(self) -> None:
        """Flush the buffer and close the file."""
        await self._close()

    # also validated in the runner, but checked in the client to catch errors early
    def _check_writable(self) -> None:
        if not self._writable:
            raise io.UnsupportedOperation("not writeable")

    # also validated in the runner, but checked in the client to catch errors early
    def _check_readable(self) -> None:
        if not self._readable:
            raise io.UnsupportedOperation("not readable")

    # also validated in the runner, but checked in the client to catch errors early
    def _check_closed(self) -> None:
        if self._closed:
            raise ValueError("I/O operation on closed file")

    def __enter__(self) -> "_FileIO":
        self._check_closed()
        return self

    async def __exit__(self, exc_type, exc_value, traceback) -> None:
        await self._close()


FileIO = synchronize_api(_FileIO)
