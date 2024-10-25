# Copyright Modal Labs 2024
from typing import AsyncIterator, Optional, Tuple

from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from .client import _Client
from .exception import FilesystemExecutionError, UnsupportedOperation


# The Sandbox file handling API is designed to mimic Python's io.FileIO
# See https://github.com/python/cpython/blob/main/Lib/_pyio.py#L1459
# Unlike io.FileIO, it also implements some higher level APIs, like `delete_bytes` and `write_replace_bytes`.
# TODO: add delete to the entire file
# TODO: add list_dir to the Sandbox
class _FileIO:
    _readable = False
    _writable = False
    _appended = False
    _closed = True

    _task_id: Optional[str] = None
    _file_descriptor: Optional[str] = None

    def _validate_mode(self, mode: str) -> None:
        valid_modes = ["r", "w", "a", "x", "r+", "w+", "a+", "x+"]
        if mode not in valid_modes:
            raise ValueError(f"Unsupported file mode: {mode}")
        self._readable = any(m in mode for m in ["r", "r+", "w+", "a+", "x+"])
        self._writable = any(m in mode for m in ["w", "w+", "a", "a+", "x", "x+"])
        self._appended = "a" in mode

    async def _consume_output(self, exec_id: str, file_descriptor: int) -> AsyncIterator[Tuple[Optional[str], int]]:
        req = api_pb2.ContainerExecGetOutputRequest(
            exec_id=exec_id,
            timeout=55,
            last_batch_index=0,
            file_descriptor=file_descriptor,
        )
        async for batch in self._client.stub.ContainerExecGetOutput.unary_stream(req):
            if batch.HasField("exit_code"):
                yield (None, batch.exit_code)
                break
            for item in batch.items:
                yield (item.message, batch.batch_index)

    async def _wait(self, exec_id: str) -> str:
        stdout = ""
        stderr = ""
        # The filesystem API shouldn't involve any long-running processes,
        # so it should be safe to consume output/err separately here.
        errored = False
        async for data, exit_code in self._consume_output(exec_id, api_pb2.FILE_DESCRIPTOR_STDOUT):
            if data is None:
                errored = exit_code != 0
                break
            stdout += data
        async for data, exit_code in self._consume_output(exec_id, api_pb2.FILE_DESCRIPTOR_STDERR):
            if data is None:
                errored = exit_code != 0
                break
            stderr += data
        if errored:
            raise FilesystemExecutionError(stderr)
        return stdout

    async def _open_file(self, path: str, mode: str) -> None:
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_open_request=api_pb2.ContainerFileOpenRequest(path=path, mode=mode, task_id=self._task_id)
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

    async def read(self, n: int | None = None) -> bytes:
        """Read n bytes from the current position, or the entire remaining file if n is None."""
        self._check_readable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_read_request=api_pb2.ContainerFileReadRequest(
                    task_id=self._task_id, file_descriptor=self._file_descriptor, n=n
                )
            )
        )
        return await self._wait(resp.exec_id)

    async def readline(self) -> bytes:
        """Read a single line from the current position."""
        self._check_readable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_read_line_request=api_pb2.ContainerFileReadLineRequest(
                    task_id=self._task_id, file_descriptor=self._file_descriptor
                )
            )
        )
        return await self._wait(resp.exec_id)

    async def readlines(self) -> list[bytes]:
        """Read all lines from the current position."""
        self._check_readable()
        return await self.read().split(b"\n")

    async def write(self, data: bytes) -> None:
        """Write data to the current position.

        NOTE: Writes may not appear until the entire buffer is flushed, which
        can be done manually with `flush()` or automatically when the file is
        closed.
        """
        self._check_writable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_write_request=api_pb2.ContainerFileWriteRequest(
                    task_id=self._task_id, file_descriptor=self._file_descriptor, data=data
                )
            )
        )
        await self._wait(resp.exec_id)

    async def flush(self) -> None:
        """Flush the buffer to disk."""
        self._check_writable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_flush_request=api_pb2.ContainerFileFlushRequest(
                    task_id=self._task_id, file_descriptor=self._file_descriptor
                )
            )
        )
        await self._wait(resp.exec_id)

    def _get_whence(self, whence: int) -> api_pb2.SeekWhence:
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
        self._check_seekable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_seek_request=api_pb2.ContainerFileSeekRequest(
                    task_id=self._task_id,
                    file_descriptor=self._file_descriptor,
                    offset=offset,
                    whence=self._get_whence(whence),
                )
            )
        )
        await self._wait(resp.exec_id)

    async def delete_bytes(self, start_inclusive: int | None = None, end_exclusive: int | None = None) -> None:
        """Delete a range of bytes from the file.

        `start_inclusive` and `end_exclusive` are byte offsets. If either is
        None, the start or end of the file is used, respectively.

        Resets the file pointer to the start of the file.
        """
        self._check_seekable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_delete_bytes_request=api_pb2.ContainerFileDeleteBytesRequest(
                    task_id=self._task_id,
                    file_descriptor=self._file_descriptor,
                    start_inclusive=start_inclusive,
                    end_exclusive=end_exclusive,
                )
            )
        )
        await self._wait(resp.exec_id)

    async def write_replace_bytes(
        self, data: bytes, start_inclusive: int | None = None, end_exclusive: int | None = None
    ) -> None:
        """Replace a range of bytes in the file with new data.

        `start_inclusive` and `end_exclusive` are byte offsets. If either is
        None, the start or end of the file is used, respectively.

        Resets the file pointer to the start of the file.
        """
        self._check_seekable()
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_write_replace_bytes_request=api_pb2.ContainerFileWriteReplaceBytesRequest(
                    task_id=self._task_id,
                    file_descriptor=self._file_descriptor,
                    data=data,
                    start_inclusive=start_inclusive,
                    end_exclusive=end_exclusive,
                )
            )
        )
        await self._wait(resp.exec_id)

    async def _close(self) -> None:
        # Buffer is flushed by the runner on close
        resp = await self._client.stub.ContainerFilesystemExec(
            api_pb2.ContainerFilesystemExecRequest(
                file_close_request=api_pb2.ContainerFileCloseRequest(
                    task_id=self._task_id, file_descriptor=self._file_descriptor
                )
            )
        )
        await self._wait(resp.exec_id)
        self._closed = True

    async def close(self) -> None:
        """Flush the buffer and close the file."""
        await self._close()

    # This is also validated in the runner, but we check in the client to catch errors early
    def _check_writable(self) -> None:
        if not self._writable:
            raise UnsupportedOperation("File is not writable. Add 'w' to the file mode to make it writable.")

    # This is also validated in the runner, but we check in the client to catch errors early
    def _check_readable(self) -> None:
        if not self._readable:
            raise UnsupportedOperation("File is not readable. Add 'r' to the file mode to make it readable.")

    # This is also validated in the runner, but we check in the client to catch errors early
    def _check_closed(self) -> None:
        if self._closed:
            raise ValueError("Cannot perform I/O on a closed file")

    def __enter__(self) -> "_FileIO":
        self._check_closed()
        return self

    async def __exit__(self, exc_type, exc_value, traceback) -> None:
        await self._close()


FileIO = synchronize_api(_FileIO)
