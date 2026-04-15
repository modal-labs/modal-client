# Copyright Modal Labs 2026
import asyncio
import os
import random
import string
import time
import weakref
from pathlib import Path
from typing import Union, cast

from ._utils.async_utils import synchronize_api
from ._utils.logger import logger
from ._utils.sandbox_fs_utils import (
    make_make_directory_command,
    make_read_file_command,
    make_remove_command,
    make_write_file_command,
    raise_make_directory_error,
    raise_read_file_error,
    raise_remove_error,
    raise_write_file_error,
    translate_exec_errors,
    validate_absolute_remote_path,
)
from .io_streams import TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE

_SANDBOX_FS_TOOLS_PATH = "/__modal/.bin/modal-sandbox-fs-tools"

_BYTES_PER_MIB = 1024 * 1024


def _log_throughput(op: str, size_bytes: int, dur_s: float) -> None:
    size_mib = size_bytes / _BYTES_PER_MIB
    throughput_mib_s = size_mib / dur_s
    logger.debug(f"sandbox {op}: {size_mib:.2f} MiB ({throughput_mib_s:.2f} MiB/s, total {dur_s:.2f}s)")


class _SandboxFilesystem:
    """mdmd:namespace
    Namespace for Sandbox filesystem APIs."""

    def __init__(self, sandbox):
        """mdmd:hidden"""
        from modal.sandbox import _Sandbox

        # inv type-stubs does not work with importing _Sandbox with TYPE_CHECKING,
        # so we'll use `cast` as a workaround.
        # Use a weakref proxy to avoid circular references between Sandbox and SandboxFilesystem.
        self._sandbox = cast(_Sandbox, weakref.proxy(sandbox))

    async def read_bytes(self, remote_path: str) -> bytes:
        """Read a file from the Sandbox and return its contents as bytes.

        `remote_path` must be an absolute path to a file in the Sandbox.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
        - `SandboxFilesystemPermissionError`: read permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_bytes(b"Hello, world!\\n", "/tmp/hello.bin")
        contents = sandbox.filesystem.read_bytes("/tmp/hello.bin")
        print(contents.decode("utf-8"))
        ```
        """
        validate_absolute_remote_path(remote_path, "read_bytes")

        t0 = time.monotonic()
        with translate_exec_errors("read_bytes", remote_path):
            process = await self._sandbox.exec(_SANDBOX_FS_TOOLS_PATH, make_read_file_command(remote_path), text=False)
            stdout, stderr, returncode = await asyncio.gather(
                process.stdout.read(), process.stderr.read(), process.wait()
            )

        if returncode != 0:
            raise_read_file_error(returncode, stderr, remote_path)

        dur_s = max(time.monotonic() - t0, 0.001)
        _log_throughput(f"read_bytes {remote_path}", len(stdout), dur_s)
        return stdout

    async def read_text(self, remote_path: str) -> str:
        """Read a file from the Sandbox and return its contents as a UTF-8 string.

        `remote_path` must be an absolute path to a file in the Sandbox.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
        - `SandboxFilesystemPermissionError`: read permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_text("Hello, world!\\n", "/tmp/hello.txt")
        contents = sandbox.filesystem.read_text("/tmp/hello.txt")
        print(contents)
        ```
        """
        validate_absolute_remote_path(remote_path, "read_text")

        t0 = time.monotonic()
        with translate_exec_errors("read_text", remote_path):
            process = await self._sandbox.exec(_SANDBOX_FS_TOOLS_PATH, make_read_file_command(remote_path))
            stdout, stderr, returncode = await asyncio.gather(
                process.stdout.read(), process.stderr.read(), process.wait()
            )

        if returncode != 0:
            raise_read_file_error(returncode, stderr, remote_path)

        dur_s = max(time.monotonic() - t0, 0.001)
        # len(stdout) is character count, not byte count — close enough for
        # debug logging and avoids re-encoding the entire string.
        _log_throughput(f"read_text {remote_path}", len(stdout), dur_s)
        return stdout

    async def copy_to_local(self, remote_path: str, local_path: Union[str, os.PathLike]) -> None:
        """Copy a file from the Sandbox to a local path.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `local_path` are created if needed.
        The local file is overwritten if it already exists.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the remote path does not exist.
        - `SandboxFilesystemIsADirectoryError`: the remote path points to a directory.
        - `SandboxFilesystemPermissionError`: read permission is denied in the Sandbox.
        - `SandboxFilesystemError`: the command fails for any other reason.
        - `IsADirectoryError`: `local_path` points to a directory.
        - `NotADirectoryError`: a component of the `local_path` parent is not a directory.
        - `PermissionError`: writing `local_path` is not permitted.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_text("Hello, world!\\n", "/tmp/hello.txt")
        sandbox.filesystem.copy_to_local("/tmp/hello.txt", "/tmp/local-hello.txt")
        ```
        """
        validate_absolute_remote_path(remote_path, "copy_to_local")
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same directory as the target so that
        # replace() is atomic. This avoids clobbering an existing local file
        # when the remote read fails.
        suffix = "".join(random.choices(string.ascii_letters + string.digits, k=6))
        tmp_path = local_path_obj.parent / f".modal-sandbox-fs-tmp-{suffix}"
        t0 = time.monotonic()
        try:
            with translate_exec_errors("copy_to_local", remote_path):
                process = await self._sandbox.exec(
                    _SANDBOX_FS_TOOLS_PATH, make_read_file_command(remote_path), text=False
                )

                async def stream_stdout_to_file() -> int:
                    total = 0
                    with open(tmp_path, "wb") as file_obj:
                        async for chunk in process.stdout:
                            file_obj.write(chunk)
                            total += len(chunk)
                    return total

                total_bytes, stderr, returncode = await asyncio.gather(
                    stream_stdout_to_file(), process.stderr.read(), process.wait()
                )

            if returncode != 0:
                raise_read_file_error(returncode, stderr, remote_path)

            dur_s = max(time.monotonic() - t0, 0.001)
            _log_throughput(f"copy_to_local {remote_path} -> {local_path}", total_bytes, dur_s)

            # Rename the temporary file into its target location.
            tmp_path.replace(local_path_obj)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    async def write_bytes(self, data: Union[bytes, bytearray, memoryview], remote_path: str) -> None:
        """Write binary content to a file in the Sandbox.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `remote_path` are created if needed.
        The remote file is overwritten if it already exists.

        **Raises**

        - `SandboxFilesystemNotADirectoryError`: a parent path component is not a directory.
        - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
        - `SandboxFilesystemPermissionError`: write permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_bytes(b"Hello, world!\\n", "/tmp/hello.bin")
        ```
        """
        validate_absolute_remote_path(remote_path, "write_bytes")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")

        t0 = time.monotonic()
        with translate_exec_errors("write_bytes", remote_path):
            process = await self._sandbox.exec(_SANDBOX_FS_TOOLS_PATH, make_write_file_command(remote_path))
            for offset in range(0, max(len(data), 1), TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE):
                process.stdin.write(data[offset : offset + TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE])
                await process.stdin.drain()
            process.stdin.write_eof()
            await process.stdin.drain()
            stderr, returncode = await asyncio.gather(process.stderr.read(), process.wait())

        if returncode != 0:
            raise_write_file_error(returncode, stderr, remote_path)

        dur_s = max(time.monotonic() - t0, 0.001)
        _log_throughput(f"write_bytes {remote_path}", len(data), dur_s)

    async def write_text(self, data: str, remote_path: str) -> None:
        """Write UTF-8 text to a file in the Sandbox.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `remote_path` are created if needed.
        The remote file is overwritten if it already exists.

        **Raises**

        - `SandboxFilesystemNotADirectoryError`: a parent path component is not a directory.
        - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
        - `SandboxFilesystemPermissionError`: write permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_text("Hello, world!\\n", "/tmp/hello.txt")
        ```
        """
        validate_absolute_remote_path(remote_path, "write_text")
        if not isinstance(data, str):
            raise TypeError("data must be str")
        await self.write_bytes(data.encode("utf-8"), remote_path)

    async def copy_from_local(self, local_path: Union[str, os.PathLike], remote_path: str) -> None:
        """Copy a local file into the Sandbox.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `remote_path` are created if needed.
        The remote file is overwritten if it already exists.

        **Raises**

        - `SandboxFilesystemNotADirectoryError`: a parent path component of `remote_path` is not a directory.
        - `SandboxFilesystemIsADirectoryError`: `remote_path` points to a directory.
        - `SandboxFilesystemPermissionError`: write permission is denied in the Sandbox.
        - `SandboxFilesystemError`: the command fails for any other reason.
        - `FileNotFoundError`: `local_path` does not exist.
        - `IsADirectoryError`: `local_path` is a directory.
        - `PermissionError`: reading `local_path` is not permitted.

        **Usage**

        ```python fixture:sandbox
        import tempfile
        from pathlib import Path

        local_path = Path(tempfile.mktemp())
        local_path.write_text("Hello, world!\\n")
        sandbox.filesystem.copy_from_local(local_path, "/tmp/hello.txt")
        ```
        """
        validate_absolute_remote_path(remote_path, "copy_from_local")

        t0 = time.monotonic()
        total_bytes = 0
        with open(local_path, "rb") as file_obj:
            with translate_exec_errors("copy_from_local", remote_path):
                process = await self._sandbox.exec(_SANDBOX_FS_TOOLS_PATH, make_write_file_command(remote_path))
                while True:
                    # TODO(saltzm): If this fails, the ContainerProcess will remain alive indefinitely since
                    # stdin will remain open. Unfortunately we can't just call write_eof either, since that
                    # would lead to a partially written file being persisted. We should catch exceptions
                    # from this and kill the ContainerProcess when we have a way to do this.
                    chunk = file_obj.read(TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    process.stdin.write(chunk)
                    await process.stdin.drain()
                process.stdin.write_eof()
                await process.stdin.drain()
                stderr, returncode = await asyncio.gather(process.stderr.read(), process.wait())

            if returncode != 0:
                raise_write_file_error(returncode, stderr, remote_path)

        dur_s = max(time.monotonic() - t0, 0.001)
        _log_throughput(f"copy_from_local {local_path} -> {remote_path}", total_bytes, dur_s)

    async def remove(self, remote_path: str, *, recursive: bool = False) -> None:
        """Remove a file or directory in the Sandbox.

        `remote_path` must be an absolute path in the Sandbox.

        When `remote_path` is a directory and `recursive` is `False` (the
        default), removes it only if it is empty. When `recursive` is `True`,
        removes the directory and all its contents.

        Recursive directory removal is not supported on all mounts.
        In particular, `CloudBucketMount` does not support it. An
        `InvalidError` is raised in that case.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemDirectoryNotEmptyError`: `recursive` is `False` and the directory is not empty.
        - `SandboxFilesystemPermissionError`: removal is not permitted.
        - `InvalidError`: the operation is not supported by the mount.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        To remove a file:

        ```python fixture:sandbox
        sandbox.filesystem.write_bytes(b"Hello, world!\\n", "/tmp/hello.bin")
        sandbox.filesystem.remove("/tmp/hello.bin")
        ```

        To remove a directory and all its contents:

        ```python fixture:sandbox
        sandbox.filesystem.make_directory("/tmp/mydir/subdir")
        sandbox.filesystem.remove("/tmp/mydir", recursive=True)
        ```
        """
        validate_absolute_remote_path(remote_path, "remove")

        with translate_exec_errors("remove", remote_path):
            process = await self._sandbox.exec(_SANDBOX_FS_TOOLS_PATH, make_remove_command(remote_path, recursive))
            stderr, returncode = await asyncio.gather(process.stderr.read(), process.wait())

        if returncode != 0:
            raise_remove_error(returncode, stderr, remote_path)

    async def make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        """Create a new directory in the Sandbox.

        `remote_path` must be an absolute path in the Sandbox.

        When `create_parents` is `True` (the default), any missing parent directories are created and the call is
        idempotent (succeeds silently if the directory already exists). When `create_parents` is `False`, the
        immediate parent directory must already exist and the path must not already exist.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the parent directory does not exist and `create_parents` is `False`.
        - `SandboxFilesystemPathAlreadyExistsError`: the path already exists.
        - `SandboxFilesystemNotADirectoryError`: a path component is not a directory.
        - `SandboxFilesystemPermissionError`: creation is not permitted.
        - `InvalidError`: the operation is not supported by the mount.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.make_directory("/tmp/a/b/c")
        ```
        """
        validate_absolute_remote_path(remote_path, "make_directory")

        with translate_exec_errors("make_directory", remote_path):
            process = await self._sandbox.exec(
                _SANDBOX_FS_TOOLS_PATH, make_make_directory_command(remote_path, create_parents)
            )
            stderr, returncode = await asyncio.gather(process.stderr.read(), process.wait())

        if returncode != 0:
            raise_make_directory_error(returncode, stderr, remote_path)


SandboxFilesystem = synchronize_api(_SandboxFilesystem)
