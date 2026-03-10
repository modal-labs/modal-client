# Copyright Modal Labs 2026
import asyncio
import os
import random
import string
import weakref
from contextlib import suppress
from pathlib import Path
from typing import Union, cast

from ._utils.async_utils import synchronize_api
from ._utils.sandbox_fs_utils import (
    make_read_file_command,
    make_write_file_command,
    raise_read_file_error,
    raise_write_file_error,
    translate_exec_errors,
    validate_absolute_remote_path,
)
from .io_streams import TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE

_SANDBOX_FS_TOOLS_PATH = "/__modal/.bin/modal-sandbox-fs-tools"


class _SandboxFilesystem:
    """Namespace for Sandbox filesystem APIs."""

    def __init__(self, sandbox):
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
        """
        validate_absolute_remote_path(remote_path, "read_bytes")

        with translate_exec_errors("read_bytes", remote_path):
            process = await self._sandbox.exec(_SANDBOX_FS_TOOLS_PATH, make_read_file_command(remote_path), text=False)
            stdout, stderr, returncode = await asyncio.gather(
                process.stdout.read(), process.stderr.read(), process.wait()
            )

        if returncode != 0:
            raise_read_file_error(returncode, stderr, remote_path)

        return stdout

    async def read_text(self, remote_path: str) -> str:
        """Read a file from the Sandbox and return its contents as a UTF-8 string.

        `remote_path` must be an absolute path to a file in the Sandbox.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemIsADirectoryError`: the path points to a directory.
        - `SandboxFilesystemPermissionError`: read permission is denied.
        - `SandboxFilesystemError`: the command fails for any other reason.
        """
        validate_absolute_remote_path(remote_path, "read_text")

        with translate_exec_errors("read_text", remote_path):
            process = await self._sandbox.exec(_SANDBOX_FS_TOOLS_PATH, make_read_file_command(remote_path))
            stdout, stderr, returncode = await asyncio.gather(
                process.stdout.read(), process.stderr.read(), process.wait()
            )

        if returncode != 0:
            raise_read_file_error(returncode, stderr, remote_path)

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
        """
        validate_absolute_remote_path(remote_path, "copy_to_local")
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same directory as the target so that
        # replace() is atomic. This avoids clobbering an existing local file
        # when the remote read fails.
        suffix = "".join(random.choices(string.ascii_letters + string.digits, k=6))
        tmp_path = local_path_obj.parent / f".modal-sandbox-fs-tmp-{suffix}"
        try:
            with translate_exec_errors("copy_to_local", remote_path):
                process = await self._sandbox.exec(
                    _SANDBOX_FS_TOOLS_PATH, make_read_file_command(remote_path), text=False
                )

                async def stream_stdout_to_file():
                    with open(tmp_path, "wb") as file_obj:
                        async for chunk in process.stdout:
                            file_obj.write(chunk)

                _, stderr, returncode = await asyncio.gather(
                    stream_stdout_to_file(), process.stderr.read(), process.wait()
                )

            if returncode != 0:
                raise_read_file_error(returncode, stderr, remote_path)

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
        """
        validate_absolute_remote_path(remote_path, "write_bytes")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")

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
        """
        validate_absolute_remote_path(remote_path, "copy_from_local")

        with open(local_path, "rb") as file_obj:
            with translate_exec_errors("copy_from_local", remote_path):
                process = await self._sandbox.exec(_SANDBOX_FS_TOOLS_PATH, make_write_file_command(remote_path))
                while True:
                    try:
                        chunk = file_obj.read(TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE)
                    except Exception:
                        # Best-effort EOF so the container process doesn't hang.
                        with suppress(Exception):
                            process.stdin.write_eof()
                            await process.stdin.drain()
                        raise
                    if not chunk:
                        break
                    process.stdin.write(chunk)
                    await process.stdin.drain()
                process.stdin.write_eof()
                await process.stdin.drain()
                stderr, returncode = await asyncio.gather(process.stderr.read(), process.wait())

            if returncode != 0:
                raise_write_file_error(returncode, stderr, remote_path)


SandboxFilesystem = synchronize_api(_SandboxFilesystem)
