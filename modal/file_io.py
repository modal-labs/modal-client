# Copyright Modal Labs 2024
import asyncio
import io
from typing import AsyncIterator, Generic, Literal, Optional, Sequence, TypeVar, Union

from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal._utils.grpc_utils import retry_transient_errors
from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES
from .client import _Client
from .exception import FilesystemExecutionError

OpenTextModeUpdating = Literal["r+", "+r", "w+", "+w", "a+", "+a", "x+", "+x"]
OpenTextModeWriting = Literal["w", "a", "x"]
OpenTextModeReading = Literal["r"]
OpenTextMode = Union[OpenTextModeUpdating, OpenTextModeWriting, OpenTextModeReading]
OpenBinaryModeUpdating = Literal[
    "rb+",
    "r+b",
    "+rb",
    "br+",
    "b+r",
    "+br",
    "wb+",
    "w+b",
    "+wb",
    "bw+",
    "b+w",
    "+bw",
    "ab+",
    "a+b",
    "+ab",
    "ba+",
    "b+a",
    "+ba",
    "xb+",
    "x+b",
    "+xb",
    "bx+",
    "b+x",
    "+bx",
]
OpenBinaryModeWriting = Literal["wb", "bw", "ab", "ba", "xb", "bx"]
OpenBinaryModeReading = Literal["rb", "br"]
OpenBinaryMode = Union[OpenBinaryModeUpdating, OpenBinaryModeReading, OpenBinaryModeWriting]

LARGE_FILE_SIZE_LIMIT = 16 * 1024 * 1024  # 16 MiB
READ_FILE_SIZE_LIMIT = 100 * 1024 * 1024  # 100 MiB

ERROR_MAPPING = {
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_UNSPECIFIED: FilesystemExecutionError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_PERM: PermissionError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_NOENT: FileNotFoundError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_IO: IOError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_NXIO: IOError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_NOMEM: MemoryError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_ACCES: PermissionError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_EXIST: FileExistsError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_NOTDIR: NotADirectoryError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_ISDIR: IsADirectoryError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_INVAL: OSError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_MFILE: OSError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_FBIG: OSError,
    api_pb2.SystemErrorCode.SYSTEM_ERROR_CODE_NOSPC: OSError,
}

T = TypeVar("T", str, bytes)


# The Sandbox file handling API is designed to mimic Python's io.FileIO
# See https://github.com/python/cpython/blob/main/Lib/_pyio.py#L1459
# Unlike io.FileIO, it also implements some higher level APIs, like `delete_bytes` and `overwrite_bytes`.
class _FileIO(Generic[T]):
    """FileIO handle for the Sandbox filesystem API.

    The API is designed to mimic Python's io.FileIO.

    **Usage**

    ```python
    import modal

    app = modal.App.lookup("my-app", create_if_missing=True)

    sb = modal.Sandbox.create(app=app)
    f = sb.open("/tmp/foo.txt", "w")
    f.write("hello")
    f.close()
    ```
    """

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

        mode_count = sum(1 for c in mode if c in "rwax")
        if mode_count > 1:
            raise ValueError("must have exactly one of create/read/write/append mode")

        seen_chars = set()
        for char in mode:
            if char in seen_chars:
                raise ValueError(f"Invalid file mode: {mode}")
            seen_chars.add(char)

    def _handle_error(self, error: api_pb2.SystemErrorMessage) -> None:
        error_class = ERROR_MAPPING.get(error.error_code, FilesystemExecutionError)
        raise error_class(error.error_message)

    async def _consume_output(self, exec_id: str) -> AsyncIterator[Optional[bytes]]:
        req = api_pb2.ContainerFilesystemExecGetOutputRequest(
            exec_id=exec_id,
            timeout=55,
        )
        async for batch in self._client.stub.ContainerFilesystemExecGetOutput.unary_stream(req):
            if batch.eof:
                yield None
                break
            if batch.HasField("error"):
                self._handle_error(batch.error)
            for message in batch.output:
                yield message

    async def _wait(self, exec_id: str) -> bytes:
        # The logic here is similar to how output is read from `exec`
        output = b""
        completed = False
        retries_remaining = 10
        while not completed:
            try:
                async for data in self._consume_output(exec_id):
                    if data is None:
                        completed = True
                        break
                    output += data
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
    async def create(
        cls, path: str, mode: Union[OpenTextMode, OpenBinaryMode], client: _Client, task_id: str
    ) -> "_FileIO":
        """Create a new FileIO handle."""
        self = cls.__new__(cls)
        self._client = client
        self._task_id = task_id
        self._validate_mode(mode)
        await self._open_file(path, mode)
        self._closed = False
        return self

    async def _make_request(
        self, request: api_pb2.ContainerFilesystemExecRequest
    ) -> api_pb2.ContainerFilesystemExecResponse:
        return await retry_transient_errors(self._client.stub.ContainerFilesystemExec, request)

    async def _make_read_request(self, n: Optional[int]) -> bytes:
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_read_request=api_pb2.ContainerFileReadRequest(file_descriptor=self._file_descriptor, n=n),
                task_id=self._task_id,
            )
        )
        return await self._wait(resp.exec_id)

    async def read(self, n: Optional[int] = None) -> T:
        """Read n bytes from the current position, or the entire remaining file if n is None."""
        self._check_closed()
        self._check_readable()
        if n is not None and n > READ_FILE_SIZE_LIMIT:
            raise ValueError("Read request payload exceeds 100 MiB limit")
        output = await self._make_read_request(n)
        if self._binary:
            return output
        return output.decode("utf-8")

    async def readline(self) -> T:
        """Read a single line from the current position."""
        self._check_closed()
        self._check_readable()
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_read_line_request=api_pb2.ContainerFileReadLineRequest(file_descriptor=self._file_descriptor),
                task_id=self._task_id,
            )
        )
        output = await self._wait(resp.exec_id)
        if self._binary:
            return output
        return output.decode("utf-8")

    async def readlines(self) -> Sequence[T]:
        """Read all lines from the current position."""
        self._check_closed()
        self._check_readable()
        output = await self._make_read_request(None)
        if self._binary:
            lines_bytes = output.split(b"\n")
            return [line + b"\n" for line in lines_bytes[:-1]] + ([lines_bytes[-1]] if lines_bytes[-1] else [])
        else:
            lines = output.decode("utf-8").split("\n")
            return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

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
        if len(data) > LARGE_FILE_SIZE_LIMIT:
            raise ValueError("Write request payload exceeds 16 MiB limit")
        resp = await self._make_request(
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
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_flush_request=api_pb2.ContainerFileFlushRequest(file_descriptor=self._file_descriptor),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    def _get_whence(self, whence: int):
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
        resp = await self._make_request(
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

    async def delete_bytes(self, start: Optional[int] = None, end: Optional[int] = None) -> None:
        """Delete a range of bytes from the file.

        `start` and `end` are byte offsets. `start` is inclusive, `end` is exclusive.
        If either is None, the start or end of the file is used, respectively.

        Resets the file pointer to the start of the file.
        """
        self._check_closed()
        if start is not None and end is not None:
            if start >= end:
                raise ValueError("start must be less than end")
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_delete_bytes_request=api_pb2.ContainerFileDeleteBytesRequest(
                    file_descriptor=self._file_descriptor,
                    start_inclusive=start,
                    end_exclusive=end,
                ),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def overwrite_bytes(self, data: bytes, start: Optional[int] = None, end: Optional[int] = None) -> None:
        """Overwrite a range of bytes in the file with new data. The length of the data does not
        have to be the same as the length of the range being overwritten.

        `start` and `end` are byte offsets. `start` is inclusive, `end` is exclusive.
        If either is None, the start or end of the file is used, respectively.

        Resets the file pointer to the start of the file.
        """
        self._check_closed()
        if start is not None and end is not None:
            if start >= end:
                raise ValueError("start must be less than end")
        if len(data) > LARGE_FILE_SIZE_LIMIT:
            raise ValueError("Write request payload exceeds 16MB limit")
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_write_replace_bytes_request=api_pb2.ContainerFileWriteReplaceBytesRequest(
                    file_descriptor=self._file_descriptor,
                    data=data,
                    start_inclusive=start,
                    end_exclusive=end,
                ),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def listdir(self, path: str) -> list[str]:
        """List the contents of the provided directory."""
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_ls_request=api_pb2.ContainerFileLsRequest(path=path),
                task_id=self._task_id,
            )
        )
        return resp.file_ls_response.paths

    async def mkdir(self, path: str) -> None:
        """Create a new directory."""
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_mkdir_request=api_pb2.ContainerFileMkdirRequest(path=path),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def rmdir(self, dirpath: str) -> None:
        """Remove a directory."""
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_rm_request=api_pb2.ContainerFileRmRequest(path=dirpath, recursive=True),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def rm(self, filepath: str) -> None:
        """Remove a file."""
        resp = await self._make_request(
            api_pb2.ContainerFilesystemExecRequest(
                file_rm_request=api_pb2.ContainerFileRmRequest(path=filepath, recursive=False),
                task_id=self._task_id,
            )
        )
        await self._wait(resp.exec_id)

    async def _close(self) -> None:
        # Buffer is flushed by the runner on close
        resp = await self._make_request(
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
