# Copyright Modal Labs 2026
import asyncio
import json
import os
import random
import string
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Optional, Union, cast

if TYPE_CHECKING:
    import modal.sandbox

from ._utils.async_utils import synchronize_api
from ._utils.logger import logger
from ._utils.sandbox_fs_utils import (
    make_list_files_command,
    make_make_directory_command,
    make_read_file_command,
    make_remove_command,
    make_stat_command,
    make_watch_command,
    make_write_file_command,
    raise_list_files_error,
    raise_make_directory_error,
    raise_read_file_error,
    raise_remove_error,
    raise_stat_error,
    raise_watch_error,
    raise_write_file_error,
    translate_exec_errors,
    validate_absolute_remote_path,
)
from .exception import ConflictError
from .io_streams import TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE
from .types import FileInfo, FileType, FileWatchEvent, FileWatchEventType

_SANDBOX_FS_TOOLS_PATH = "/__modal/.bin/modal-sandbox-fs-tools"

_BYTES_PER_MIB = 1024 * 1024


def _log_throughput(op: str, size_bytes: int, dur_s: float) -> None:
    size_mib = size_bytes / _BYTES_PER_MIB
    throughput_mib_s = size_mib / dur_s
    logger.debug(f"sandbox {op}: {size_mib:.2f} MiB ({throughput_mib_s:.2f} MiB/s, total {dur_s:.2f}s)")


# Rust rename variants that all collapse to Python FileWatchEventType.Modify.
_RUST_RENAME_VARIANTS = ("Rename", "RenameFrom", "RenameTo")


def _expand_watch_filter(filter: list[FileWatchEventType]) -> list[str]:
    """Expand a Python filter list into modal-sandbox-fs-tools event type strings.

    FileWatchEventType.Modify covers fs tool's Rename/RenameFrom/RenameTo variants,
    so those must be included when the caller filters for Modify events.
    """
    result: list[str] = []
    for event_type in filter:
        result.append(event_type.value)
        if event_type == FileWatchEventType.Modify:
            result.extend(_RUST_RENAME_VARIANTS)
    return result


class _SandboxFilesystem:
    """mdmd:namespace
    Namespace for Sandbox filesystem APIs."""

    _container: Union["modal.sandbox._Sandbox", "modal.sandbox._SidecarContainer"]

    def __init__(self, container: Union["modal.sandbox._Sandbox", "modal.sandbox._SidecarContainer"]) -> None:
        """mdmd:hidden"""
        from modal.sandbox import _Sandbox, _SidecarContainer

        # Use a weakref proxy to avoid circular references between Sandbox/SidecarContainer and SandboxFilesystem.
        self._container = cast(_Sandbox | _SidecarContainer, weakref.proxy(container))

    async def copy_from_local(self, local_path: str | os.PathLike, remote_path: str) -> None:
        """Copy a local file into the Sandbox.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `remote_path` are created if needed.
        The remote file is overwritten if it already exists.

        Args:
            local_path: Path to the file on the local machine.
            remote_path: Absolute path to the file in the Sandbox.

        Raises:
            SandboxFilesystemNotADirectoryError: A parent path component of ``remote_path`` is not a directory.
            SandboxFilesystemIsADirectoryError: ``remote_path`` points to a directory.
            SandboxFilesystemPermissionError: Write permission is denied in the Sandbox.
            SandboxFilesystemError: The command fails for any other reason.
            FileNotFoundError: ``local_path`` does not exist.
            IsADirectoryError: ``local_path`` is a directory.
            PermissionError: Reading ``local_path`` is not permitted.

        Examples:
            ```python fixture:sandbox fixture:tmpdir
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
                process = await self._container.exec(_SANDBOX_FS_TOOLS_PATH, make_write_file_command(remote_path))
                try:
                    while True:
                        logger.debug(
                            f"sandbox copy_from_local('{local_path}', '{remote_path}'): reading "
                            f"{TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE} bytes from {local_path} at offset {total_bytes}"
                        )
                        # TODO(saltzm): If this fails, the ContainerProcess will remain alive indefinitely since
                        # stdin will remain open. Unfortunately we can't just call write_eof either, since that
                        # would lead to a partially written file being persisted. We should catch exceptions
                        # from this and kill the ContainerProcess when we have a way to do this.
                        chunk = file_obj.read(TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE)
                        if not chunk:
                            logger.debug(
                                f"sandbox copy_from_local('{local_path}', '{remote_path}'): read no data from "
                                f"{local_path}, finished reading file"
                            )
                            break
                        total_bytes += len(chunk)
                        logger.debug(
                            f"sandbox copy_from_local('{local_path}', '{remote_path}'): writing {len(chunk)} bytes "
                            f"from {local_path} to remote {remote_path}"
                        )
                        process.stdin.write(chunk)
                        await process.stdin.drain()
                        logger.debug(
                            f"sandbox copy_from_local('{local_path}', '{remote_path}'): finished writing "
                            f"{len(chunk)} bytes from {local_path} to remote {remote_path}"
                        )
                    logger.debug(
                        f"sandbox copy_from_local('{local_path}', '{remote_path}'): writing eof to remote {remote_path}"
                    )
                    process.stdin.write_eof()
                    await process.stdin.drain()
                    logger.debug(
                        f"sandbox copy_from_local('{local_path}', '{remote_path}'): finished writing eof to remote "
                        f"{remote_path}"
                    )
                # When the FS tools binary exits early on an error, the worker
                # reports the dropped stdin write as ConflictError.
                except ConflictError:
                    pass

                async def read_stderr():
                    stderr = await process.stderr.read()
                    logger.debug(f"sandbox copy_from_local('{local_path}', '{remote_path}'): finished reading stderr")
                    return stderr

                async def wait_for_process_completion():
                    returncode = await process.wait()
                    logger.debug(
                        f"sandbox copy_from_local('{local_path}', '{remote_path}'): finished waiting for process "
                        "completion"
                    )
                    return returncode

                stderr, returncode = await asyncio.gather(read_stderr(), wait_for_process_completion())

            if returncode != 0:
                raise_write_file_error(returncode, stderr, remote_path)

        dur_s = max(time.monotonic() - t0, 0.001)
        _log_throughput(f"copy_from_local('{local_path}', '{remote_path}')", total_bytes, dur_s)

    async def copy_to_local(self, remote_path: str, local_path: str | os.PathLike) -> None:
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

        ```python fixture:sandbox fixture:tmpdir
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
                process = await self._container.exec(
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

    async def list_files(self, remote_path: str) -> list[FileInfo]:
        """List files and directories in a Sandbox directory.

        Args:
            remote_path: Absolute path to the directory in the Sandbox.

        Returns:
            A list of `FileInfo` objects describing each entry.

        Raises:
            SandboxFilesystemNotFoundError: The path does not exist.
            SandboxFilesystemNotADirectoryError: The path is not a directory.
            SandboxFilesystemPermissionError: Read permission is denied.
            SandboxFilesystemError: The command fails for any other reason.

        Examples:
            ```python fixture:sandbox
            entries = sandbox.filesystem.list_files("/tmp")
            for entry in entries:
                print(entry.name, entry.type, entry.size)
            ```
        """
        validate_absolute_remote_path(remote_path, "list_files")

        t0 = time.monotonic()
        with translate_exec_errors("list_files", remote_path):
            process = await self._container.exec(_SANDBOX_FS_TOOLS_PATH, make_list_files_command(remote_path))
            stdout, stderr, returncode = await asyncio.gather(
                process.stdout.read(), process.stderr.read(), process.wait()
            )

        if returncode != 0:
            raise_list_files_error(returncode, stderr, remote_path)

        entries_data = json.loads(stdout)
        result = [
            FileInfo(
                name=entry["name"],
                path=entry["path"],
                type=FileType(entry["type"]),
                size=entry["size"],
                mode=entry["mode"],
                permissions=entry["permissions"],
                owner=entry["owner"],
                group=entry["group"],
                modified_time=entry["modified_time"],
                symlink_target=entry.get("symlink_target"),
            )
            for entry in entries_data
        ]

        dur_s = max(time.monotonic() - t0, 0.001)
        logger.debug(f"sandbox list_files {remote_path}: {len(result)} entries ({dur_s:.2f}s)")
        return result

    async def make_directory(self, remote_path: str, *, create_parents: bool = True) -> None:
        """Create a new directory in the Sandbox.

        `remote_path` must be an absolute path in the Sandbox.

        When `create_parents` is `True` (the default), any missing parent directories are created and the call is
        idempotent (succeeds silently if the directory already exists). When `create_parents` is `False`, the
        immediate parent directory must already exist and the path must not already exist.

        Args:
            remote_path: Absolute path of the directory to create in the Sandbox.
            create_parents: When ``True``, create missing parents and succeed if the directory already exists.

        Raises:
            SandboxFilesystemNotFoundError: The parent directory does not exist and ``create_parents`` is false.
            SandboxFilesystemPathAlreadyExistsError: The path already exists.
            SandboxFilesystemNotADirectoryError: A path component is not a directory.
            SandboxFilesystemPermissionError: Creation is not permitted.
            InvalidError: The operation is not supported by the mount.
            SandboxFilesystemError: The command fails for any other reason.

        Examples:
            ```python fixture:sandbox
            sandbox.filesystem.make_directory("/tmp/a/b/c")
            ```
        """
        validate_absolute_remote_path(remote_path, "make_directory")

        with translate_exec_errors("make_directory", remote_path):
            process = await self._container.exec(
                _SANDBOX_FS_TOOLS_PATH, make_make_directory_command(remote_path, create_parents)
            )
            stderr, returncode = await asyncio.gather(process.stderr.read(), process.wait())

        if returncode != 0:
            raise_make_directory_error(returncode, stderr, remote_path)

    async def read_bytes(self, remote_path: str) -> bytes:
        """Read a file from the Sandbox and return its contents as bytes.

        `remote_path` must be an absolute path to a file in the Sandbox.

        Args:
            remote_path: Absolute path to the file in the Sandbox.

        Returns:
            Raw bytes read from the file.

        Raises:
            SandboxFilesystemNotFoundError: The path does not exist.
            SandboxFilesystemIsADirectoryError: The path points to a directory.
            SandboxFilesystemPermissionError: Read permission is denied.
            SandboxFilesystemError: The command fails for any other reason.

        Examples:
            ```python fixture:sandbox
            sandbox.filesystem.write_bytes(b"Hello, world!\\n", "/tmp/hello.bin")
            contents = sandbox.filesystem.read_bytes("/tmp/hello.bin")
            print(contents.decode("utf-8"))
            ```
        """
        validate_absolute_remote_path(remote_path, "read_bytes")

        t0 = time.monotonic()
        with translate_exec_errors("read_bytes", remote_path):
            process = await self._container.exec(
                _SANDBOX_FS_TOOLS_PATH, make_read_file_command(remote_path), text=False
            )
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

        Args:
            remote_path: Absolute path to the file in the Sandbox.

        Returns:
            File contents decoded as UTF-8.

        Raises:
            SandboxFilesystemNotFoundError: The path does not exist.
            SandboxFilesystemIsADirectoryError: The path points to a directory.
            SandboxFilesystemPermissionError: Read permission is denied.
            SandboxFilesystemError: The command fails for any other reason.

        Examples:
            ```python fixture:sandbox
            sandbox.filesystem.write_text("Hello, world!\\n", "/tmp/hello.txt")
            contents = sandbox.filesystem.read_text("/tmp/hello.txt")
            print(contents)
            ```
        """
        validate_absolute_remote_path(remote_path, "read_text")

        t0 = time.monotonic()
        with translate_exec_errors("read_text", remote_path):
            process = await self._container.exec(_SANDBOX_FS_TOOLS_PATH, make_read_file_command(remote_path))
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

    async def remove(self, remote_path: str, *, recursive: bool = False) -> None:
        """Remove a file or directory in the Sandbox.

        When `remote_path` is a directory and `recursive` is `False` (the
        default), removes it only if it is empty. When `recursive` is `True`,
        removes the directory and all its contents.

        Recursive directory removal is not supported on all mounts.
        In particular, `CloudBucketMount` does not support it. An
        `InvalidError` is raised in that case.

        Args:
            remote_path: Absolute path to the file in the Sandbox.
            recursive: When ``True``, remove the directory and all its contents.

        Raises:
            SandboxFilesystemNotFoundError: The remote path does not exist.
            SandboxFilesystemDirectoryNotEmptyError: `recursive` is `False` and the directory is not empty.
            SandboxFilesystemPermissionError: Read permission is denied in the Sandbox.
            InvalidError: The operation is not supported by the mount.
            SandboxFilesystemError: The command fails for any other reason.

        Examples:
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
            process = await self._container.exec(_SANDBOX_FS_TOOLS_PATH, make_remove_command(remote_path, recursive))
            stderr, returncode = await asyncio.gather(process.stderr.read(), process.wait())

        if returncode != 0:
            raise_remove_error(returncode, stderr, remote_path)

    async def stat(self, remote_path: str) -> FileInfo:
        """Return metadata for a single file, directory, or symlink in the Sandbox.

        `remote_path` must be an absolute path in the Sandbox. If `remote_path` is a symlink, the returned
        `FileInfo` object describes the symlink, not the target it points to.

        **Raises**

        - `SandboxFilesystemNotFoundError`: the path does not exist.
        - `SandboxFilesystemNotADirectoryError`: a non-leaf component of the path is not a directory.
        - `SandboxFilesystemPermissionError`: a component of the path is not searchable.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python fixture:sandbox
        sandbox.filesystem.write_text("Hello, world!\\n", "/tmp/hello.txt")
        info = sandbox.filesystem.stat("/tmp/hello.txt")
        print(info.size, info.permissions, info.modified_time)
        ```
        """
        validate_absolute_remote_path(remote_path, "stat")

        t0 = time.monotonic()
        with translate_exec_errors("stat", remote_path):
            process = await self._container.exec(_SANDBOX_FS_TOOLS_PATH, make_stat_command(remote_path))
            stdout, stderr, returncode = await asyncio.gather(
                process.stdout.read(), process.stderr.read(), process.wait()
            )

        if returncode != 0:
            raise_stat_error(returncode, stderr, remote_path)

        entry = json.loads(stdout)
        result = FileInfo(
            name=entry["name"],
            path=entry["path"],
            type=FileType(entry["type"]),
            size=entry["size"],
            mode=entry["mode"],
            permissions=entry["permissions"],
            owner=entry["owner"],
            group=entry["group"],
            modified_time=entry["modified_time"],
            symlink_target=entry.get("symlink_target"),
        )

        dur_s = max(time.monotonic() - t0, 0.001)
        logger.debug(f"sandbox stat {remote_path}: ({dur_s:.2f}s)")
        return result

    async def watch(
        self,
        remote_path: str,
        *,
        filter: Optional[list[FileWatchEventType]] = None,
        recursive: bool = False,
        timeout: Optional[int] = None,
    ) -> AsyncIterator[FileWatchEvent]:
        """Watch a path in the Sandbox for filesystem changes.

        `remote_path` must be an absolute path in the Sandbox. If it points
        to a file, events for that file are reported. If it points to a
        directory, events for entries directly inside it are reported. Set
        `recursive=True` to also receive events for all nested subdirectories.
        If `remote_path` is a symlink, it is followed and events reference
        paths under the resolved target.

        Yields `FileWatchEvent` objects as changes occur, until either
        `timeout` seconds elapse, the iterator is closed, or the Sandbox
        is terminated.

        Optionally restrict the kinds of events emitted to those included
        in `filter`. The default filter `None` permits all event types.

        `timeout` is in seconds. `None` means watch indefinitely. When
        `timeout` elapses, the iterator stops without raising an exception.

        **Raises**

        - `SandboxFilesystemNotFoundError`: `remote_path` does not exist.
        - `SandboxFilesystemPermissionError`: watch access is denied.
        - `InvalidError`: the filesystem at `remote_path` does not support
          watching.
        - `SandboxFilesystemError`: the command fails for any other reason.

        **Usage**

        ```python notest
        for event in sandbox.filesystem.watch(
            "/tmp/foo",
            recursive=True,
            filter=[FileWatchEventType.Create],
            timeout=60,
        ):
            if any(p.endswith(".done") for p in event.paths):
                break
        ```
        """
        validate_absolute_remote_path(remote_path, "watch")

        with translate_exec_errors("watch", remote_path):
            process = await self._container.exec(
                _SANDBOX_FS_TOOLS_PATH,
                make_watch_command(
                    remote_path,
                    recursive=recursive,
                    filter=_expand_watch_filter(filter) if filter is not None else None,
                    timeout=timeout,
                ),
                bufsize=1,  # Read process.stdout one JSON event per line.
            )

            try:
                async for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        paths = data.get("paths")
                        if not paths:
                            continue
                        raw_type = data["event_type"]
                        # Collapse all Rename variants from fs tools binary to Modify.
                        if raw_type in _RUST_RENAME_VARIANTS:
                            raw_type = "Modify"
                        yield FileWatchEvent(
                            type=FileWatchEventType(raw_type),
                            paths=paths,
                        )
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Skip invalid events.
                        pass
            finally:
                # Close stdin first so the fs tools binary detects EOF and exits, before any other await.
                try:
                    process.stdin.write_eof()
                    await process.stdin.drain()
                except Exception:
                    pass
                stderr, returncode = await asyncio.gather(process.stderr.read(), process.wait())

        if returncode != 0:
            raise_watch_error(returncode, stderr, remote_path)

    async def write_bytes(self, data: bytes | bytearray | memoryview, remote_path: str) -> None:
        """Write binary content to a file in the Sandbox.

        `remote_path` must be an absolute path to a file in the Sandbox.
        Parent directories for `remote_path` are created if needed.
        The remote file is overwritten if it already exists.

        Args:
            data: Bytes to write.
            remote_path: Absolute path to the file in the Sandbox.

        Raises:
            TypeError: ``data`` is not bytes-like.
            SandboxFilesystemNotADirectoryError: A parent path component is not a directory.
            SandboxFilesystemIsADirectoryError: ``remote_path`` points to a directory.
            SandboxFilesystemPermissionError: Write permission is denied.
            SandboxFilesystemError: The command fails for any other reason.

        Examples:
            ```python fixture:sandbox
            sandbox.filesystem.write_bytes(b"Hello, world!\\n", "/tmp/hello.bin")
            ```
        """
        validate_absolute_remote_path(remote_path, "write_bytes")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")

        t0 = time.monotonic()
        with translate_exec_errors("write_bytes", remote_path):
            process = await self._container.exec(_SANDBOX_FS_TOOLS_PATH, make_write_file_command(remote_path))
            try:
                for offset in range(0, max(len(data), 1), TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE):
                    process.stdin.write(data[offset : offset + TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE])
                    await process.stdin.drain()
                process.stdin.write_eof()
                await process.stdin.drain()
            # When the FS tools binary exits early on an error, the worker
            # reports the dropped stdin write as ConflictError.
            except ConflictError:
                pass
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

        Args:
            data: Text to write (encoded as UTF-8).
            remote_path: Absolute path to the file in the Sandbox.

        Raises:
            TypeError: ``data`` is not a string.
            SandboxFilesystemNotADirectoryError: A parent path component is not a directory.
            SandboxFilesystemIsADirectoryError: ``remote_path`` points to a directory.
            SandboxFilesystemPermissionError: Write permission is denied.
            SandboxFilesystemError: The command fails for any other reason.

        Examples:
            ```python fixture:sandbox
            sandbox.filesystem.write_text("Hello, world!\\n", "/tmp/hello.txt")
            ```
        """
        validate_absolute_remote_path(remote_path, "write_text")
        if not isinstance(data, str):
            raise TypeError("data must be str")
        await self.write_bytes(data.encode("utf-8"), remote_path)


SandboxFilesystem = synchronize_api(_SandboxFilesystem)
