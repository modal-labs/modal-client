# Copyright Modal Labs 2025
"""Local file server for FUSE-based directory mounting.

This module implements a file server that runs on the local machine and
handles filesystem requests from the FUSE daemon running in the container.
"""

from __future__ import annotations

import asyncio
import errno
import os
from pathlib import Path
from typing import Any, Callable, Optional

from .protocol import (
    FuseOp,
    FuseRequest,
    FuseResponse,
    error_response,
    lstat_to_dict,
    success_response,
)


class LocalFileServer:
    """A file server that handles FUSE requests for a local directory.

    This server receives requests from the FUSE daemon running in the container
    and responds with local filesystem data.
    """

    def __init__(self, local_path: Path, read_only: bool = True):
        """Initialize the file server.

        Args:
            local_path: The local directory to serve.
            read_only: If True, reject any write operations.
        """
        self.local_path = local_path.resolve()
        self.read_only = read_only
        self._open_files: dict[int, Any] = {}  # fh -> file object
        self._next_fh = 1

        if not self.local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {self.local_path}")
        if not self.local_path.is_dir():
            raise NotADirectoryError(f"Local path is not a directory: {self.local_path}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a FUSE path to a local path, with security checks."""
        # Remove leading slash and resolve
        if path.startswith("/"):
            path = path[1:]

        if not path:
            return self.local_path

        local = (self.local_path / path).resolve()

        # Security check: ensure we don't escape the local_path
        try:
            local.relative_to(self.local_path)
        except ValueError:
            raise PermissionError(f"Path escapes mount root: {path}")

        return local

    def handle_request(self, request: FuseRequest) -> FuseResponse:
        """Handle a FUSE request and return a response."""
        try:
            match request.op:
                case FuseOp.INIT:
                    return self._handle_init(request)
                case FuseOp.PING:
                    return self._handle_ping(request)
                case FuseOp.GETATTR:
                    return self._handle_getattr(request)
                case FuseOp.READDIR:
                    return self._handle_readdir(request)
                case FuseOp.READ:
                    return self._handle_read(request)
                case FuseOp.READLINK:
                    return self._handle_readlink(request)
                case FuseOp.STATFS:
                    return self._handle_statfs(request)
                case FuseOp.OPEN:
                    return self._handle_open(request)
                case FuseOp.RELEASE:
                    return self._handle_release(request)
                case _:
                    return error_response(request.request_id, errno.ENOSYS)
        except FileNotFoundError:
            return error_response(request.request_id, errno.ENOENT)
        except PermissionError:
            return error_response(request.request_id, errno.EACCES)
        except IsADirectoryError:
            return error_response(request.request_id, errno.EISDIR)
        except NotADirectoryError:
            return error_response(request.request_id, errno.ENOTDIR)
        except OSError as e:
            return error_response(request.request_id, e.errno or errno.EIO)
        except Exception:
            return error_response(request.request_id, errno.EIO)

    def _handle_init(self, request: FuseRequest) -> FuseResponse:
        """Handle initialization request."""
        return success_response(
            request.request_id,
            attrs={"version": "1.0", "read_only": self.read_only, "root": str(self.local_path)},
        )

    def _handle_ping(self, request: FuseRequest) -> FuseResponse:
        """Handle ping request."""
        return success_response(request.request_id)

    def _handle_getattr(self, request: FuseRequest) -> FuseResponse:
        """Handle getattr request."""
        local_path = self._resolve_path(request.path)
        attrs = lstat_to_dict(local_path)
        return success_response(request.request_id, attrs=attrs)

    def _handle_readdir(self, request: FuseRequest) -> FuseResponse:
        """Handle readdir request."""
        local_path = self._resolve_path(request.path)

        if not local_path.is_dir():
            return error_response(request.request_id, errno.ENOTDIR)

        entries = [".", ".."]
        for entry in local_path.iterdir():
            entries.append(entry.name)

        return success_response(request.request_id, entries=entries)

    def _handle_read(self, request: FuseRequest) -> FuseResponse:
        """Handle read request."""
        local_path = self._resolve_path(request.path)

        if local_path.is_dir():
            return error_response(request.request_id, errno.EISDIR)

        with open(local_path, "rb") as f:
            f.seek(request.offset)
            data = f.read(request.size)

        return success_response(request.request_id, data=data)

    def _handle_readlink(self, request: FuseRequest) -> FuseResponse:
        """Handle readlink request."""
        local_path = self._resolve_path(request.path)

        if not local_path.is_symlink():
            return error_response(request.request_id, errno.EINVAL)

        target = os.readlink(local_path)
        return success_response(request.request_id, link_target=str(target))

    def _handle_statfs(self, request: FuseRequest) -> FuseResponse:
        """Handle statfs request."""
        local_path = self._resolve_path(request.path)
        st = os.statvfs(local_path)

        statfs_data = {
            "f_bsize": st.f_bsize,
            "f_frsize": st.f_frsize,
            "f_blocks": st.f_blocks,
            "f_bfree": st.f_bfree,
            "f_bavail": st.f_bavail,
            "f_files": st.f_files,
            "f_ffree": st.f_ffree,
            "f_favail": st.f_favail,
            "f_flag": st.f_flag,
            "f_namemax": st.f_namemax,
        }

        return success_response(request.request_id, statfs=statfs_data)

    def _handle_open(self, request: FuseRequest) -> FuseResponse:
        """Handle open request."""
        local_path = self._resolve_path(request.path)

        # For read-only mode, just verify the file exists and is readable
        if not local_path.exists():
            return error_response(request.request_id, errno.ENOENT)

        if local_path.is_dir():
            return error_response(request.request_id, errno.EISDIR)

        # Check read permission
        if not os.access(local_path, os.R_OK):
            return error_response(request.request_id, errno.EACCES)

        # Assign a file handle
        fh = self._next_fh
        self._next_fh += 1
        self._open_files[fh] = str(local_path)

        return success_response(request.request_id, fh=fh)

    def _handle_release(self, request: FuseRequest) -> FuseResponse:
        """Handle release request."""
        if request.fh in self._open_files:
            del self._open_files[request.fh]
        return success_response(request.request_id)


class AsyncLocalFileServer(LocalFileServer):
    """Async version of LocalFileServer for use with asyncio."""

    async def handle_request_async(self, request: FuseRequest) -> FuseResponse:
        """Handle a FUSE request asynchronously."""
        # Run the synchronous handler in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.handle_request, request)

    async def run_stdio_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """Run the file server loop, reading from reader and writing to writer.

        This is used when the file server communicates via stdin/stdout with
        the FUSE daemon.
        """
        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                request = FuseRequest.from_json(line.decode("utf-8"))
                response = await self.handle_request_async(request)

                response_line = response.to_json() + "\n"
                writer.write(response_line.encode("utf-8"))
                await writer.drain()

            except asyncio.CancelledError:
                break
            except Exception as e:
                if on_error:
                    on_error(e)
                # Try to continue on errors
                continue
