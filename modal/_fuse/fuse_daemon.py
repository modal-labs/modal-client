# Copyright Modal Labs 2025
"""FUSE daemon for mounting local directories in Modal containers.

This script runs inside the Modal container and implements a FUSE filesystem
that forwards all filesystem operations to the local file server via stdin/stdout.

The daemon requires fusepy to be installed in the container.
"""

from __future__ import annotations

import argparse
import base64
import errno
import json
import os
import sys
import threading

import fuse

# Protocol constants matching protocol.py
FUSE_OP_GETATTR = "getattr"
FUSE_OP_READDIR = "readdir"
FUSE_OP_READ = "read"
FUSE_OP_READLINK = "readlink"
FUSE_OP_STATFS = "statfs"
FUSE_OP_OPEN = "open"
FUSE_OP_RELEASE = "release"
FUSE_OP_INIT = "init"
FUSE_OP_PING = "ping"


class FuseClient:
    """Client for communicating with the local file server."""

    def __init__(self, input_stream, output_stream):
        self._input = input_stream
        self._output = output_stream
        self._lock = threading.Lock()
        self._request_id = 0

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_request(self, op: str, path: str, **kwargs) -> dict:
        """Send a request and wait for response."""
        request_id = self._next_request_id()
        request = {
            "op": op,
            "path": path,
            "request_id": request_id,
            **kwargs,
        }

        with self._lock:
            # Send request
            line = json.dumps(request) + "\n"
            self._output.write(line)
            self._output.flush()

            # Read response
            response_line = self._input.readline()
            if not response_line:
                return {"error": errno.EIO}

            response = json.loads(response_line)
            return response

    def getattr(self, path: str) -> dict:
        return self._send_request(FUSE_OP_GETATTR, path)

    def readdir(self, path: str) -> dict:
        return self._send_request(FUSE_OP_READDIR, path)

    def read(self, path: str, size: int, offset: int, fh: int) -> dict:
        return self._send_request(FUSE_OP_READ, path, size=size, offset=offset, fh=fh)

    def readlink(self, path: str) -> dict:
        return self._send_request(FUSE_OP_READLINK, path)

    def statfs(self, path: str) -> dict:
        return self._send_request(FUSE_OP_STATFS, path)

    def open(self, path: str, flags: int) -> dict:
        return self._send_request(FUSE_OP_OPEN, path, flags=flags)

    def release(self, path: str, fh: int) -> dict:
        return self._send_request(FUSE_OP_RELEASE, path, fh=fh)

    def init(self) -> dict:
        """Initialize connection with file server."""
        return self._send_request(FUSE_OP_INIT, "/")

    def ping(self) -> dict:
        """Ping the file server."""
        return self._send_request(FUSE_OP_PING, "/")


class FuseOperations:
    """FUSE operations that forward to the file server."""

    def __init__(self, client: FuseClient):
        self.client = client
        self._open_files: dict[int, str] = {}
        self._next_fh = 1

    def getattr(self, path: str, fi=None) -> dict:
        resp = self.client.getattr(path)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        return resp.get("attrs", {})

    def readdir(self, path: str, fh: int) -> list:
        resp = self.client.readdir(path)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        entries = resp.get("entries", [])
        # Always include . and ..
        if "." not in entries:
            entries = [".", ".."] + entries
        return entries

    def read(self, path: str, size: int, offset: int, fh: int) -> bytes:
        resp = self.client.read(path, size, offset, fh)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        data = resp.get("data", "")
        if data:
            return base64.b64decode(data)
        return b""

    def readlink(self, path: str) -> str:
        resp = self.client.readlink(path)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        return resp.get("link_target", "")

    def statfs(self, path: str) -> dict:
        resp = self.client.statfs(path)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        return resp.get("statfs", {})

    def open(self, path: str, flags: int) -> int:
        # For read-only, just validate the file exists
        resp = self.client.open(path, flags)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        fh = self._next_fh
        self._next_fh += 1
        self._open_files[fh] = path
        return fh

    def release(self, path: str, fh: int) -> int:
        if fh in self._open_files:
            del self._open_files[fh]
        return 0


class ModalFuse(fuse.Operations):
    """FUSE filesystem implementation that wraps FuseOperations."""

    def __init__(self, ops: FuseOperations):
        self.ops = ops

    def getattr(self, path, fh=None):
        try:
            return self.ops.getattr(path)
        except OSError as e:
            raise fuse.FuseOSError(e.errno)

    def readdir(self, path, fh):
        try:
            return self.ops.readdir(path, fh)
        except OSError as e:
            raise fuse.FuseOSError(e.errno)

    def read(self, path, size, offset, fh):
        try:
            return self.ops.read(path, size, offset, fh)
        except OSError as e:
            raise fuse.FuseOSError(e.errno)

    def readlink(self, path):
        try:
            return self.ops.readlink(path)
        except OSError as e:
            raise fuse.FuseOSError(e.errno)

    def statfs(self, path):
        try:
            return self.ops.statfs(path)
        except OSError as e:
            raise fuse.FuseOSError(e.errno)

    def open(self, path, flags):
        try:
            return self.ops.open(path, flags)
        except OSError as e:
            raise fuse.FuseOSError(e.errno)

    def release(self, path, fh):
        return self.ops.release(path, fh)


def main():
    parser = argparse.ArgumentParser(description="Modal FUSE daemon for local directory mounting")
    parser.add_argument("--mount-point", required=True, help="Path where the filesystem will be mounted")
    parser.add_argument("--foreground", action="store_true", default=True, help="Run in foreground")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    mount_point = args.mount_point

    # Create mount point if it doesn't exist
    os.makedirs(mount_point, exist_ok=True)

    # Set up communication with file server
    input_stream = sys.stdin
    output_stream = sys.stdout

    # Redirect our stdout to stderr so FUSE output doesn't interfere
    sys.stdout = sys.stderr

    client = FuseClient(input_stream, output_stream)

    # Initialize connection
    print("[modal-fuse] Initializing connection to file server...", file=sys.stderr)  # noqa: T201
    resp = client.init()
    if resp.get("error"):
        print(f"[modal-fuse] Failed to initialize: {resp}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    print("[modal-fuse] Connected to file server", file=sys.stderr)  # noqa: T201

    # Create operations handler
    operations = FuseOperations(client)

    # Run FUSE loop
    print(f"[modal-fuse] Starting FUSE filesystem at {mount_point}", file=sys.stderr)  # noqa: T201
    try:
        fuse.FUSE(
            ModalFuse(operations),
            mount_point,
            foreground=args.foreground,
            ro=True,
            allow_other=False,
            nothreads=False,
        )
    except Exception as e:
        print(f"[modal-fuse] Error: {e}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    main()
