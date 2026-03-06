# Copyright Modal Labs 2026
import asyncio
import os
import random
import string
import weakref
from pathlib import Path
from typing import Union, cast

from ._utils.async_utils import synchronize_api
from ._utils.sandbox_fs_utils import (
    make_read_file_command,
    raise_read_file_error,
    translate_exec_errors,
    validate_absolute_remote_path,
)

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


SandboxFilesystem = synchronize_api(_SandboxFilesystem)
