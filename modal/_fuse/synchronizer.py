# Copyright Modal Labs 2025
"""FUSE mount manager for Modal Shell.

This module provides the FuseMountManager class which orchestrates the
lifecycle of FUSE-based local directory mounts in Modal containers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Optional

from modal.config import logger

from .file_server import AsyncLocalFileServer
from .protocol import FuseRequest

if TYPE_CHECKING:
    from modal.container_process import _ContainerProcess
    from modal.sandbox import _Sandbox


# The FUSE daemon script that runs in the container
# We embed it here so we can inject it without needing a mount
FUSE_DAEMON_SCRIPT = """
# Modal FUSE daemon - injected by Modal SDK
import sys
import os
import json
import base64
import errno
import threading
import argparse
import fuse


class FuseClient:
    def __init__(self, input_stream, output_stream):
        self._input = input_stream
        self._output = output_stream
        self._lock = threading.Lock()
        self._request_id = 0

    def _next_request_id(self):
        self._request_id += 1
        return self._request_id

    def _send_request(self, op, path, **kwargs):
        request_id = self._next_request_id()
        request = {"op": op, "path": path, "request_id": request_id, **kwargs}
        with self._lock:
            line = json.dumps(request) + "\\n"
            self._output.write(line)
            self._output.flush()
            response_line = self._input.readline()
            if not response_line:
                return {"error": errno.EIO}
            return json.loads(response_line)

    def getattr(self, path):
        return self._send_request("getattr", path)

    def readdir(self, path):
        return self._send_request("readdir", path)

    def read(self, path, size, offset, fh):
        return self._send_request("read", path, size=size, offset=offset, fh=fh)

    def readlink(self, path):
        return self._send_request("readlink", path)

    def statfs(self, path):
        return self._send_request("statfs", path)

    def open(self, path, flags):
        return self._send_request("open", path, flags=flags)

    def release(self, path, fh):
        return self._send_request("release", path, fh=fh)

    def init(self):
        return self._send_request("init", "/")


class FuseOperations:
    def __init__(self, client):
        self.client = client
        self._open_files = {}
        self._next_fh = 1

    def getattr(self, path, fh=None):
        resp = self.client.getattr(path)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        return resp.get("attrs", {})

    def readdir(self, path, fh):
        resp = self.client.readdir(path)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        entries = resp.get("entries", [])
        if "." not in entries:
            entries = [".", ".."] + entries
        return entries

    def read(self, path, size, offset, fh):
        resp = self.client.read(path, size, offset, fh)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        data = resp.get("data", "")
        if data:
            return base64.b64decode(data)
        return b""

    def readlink(self, path):
        resp = self.client.readlink(path)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        return resp.get("link_target", "")

    def statfs(self, path):
        resp = self.client.statfs(path)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        return resp.get("statfs", {})

    def open(self, path, flags):
        resp = self.client.open(path, flags)
        if resp.get("error"):
            raise OSError(resp["error"], os.strerror(resp["error"]))
        fh = self._next_fh
        self._next_fh += 1
        self._open_files[fh] = path
        return fh

    def release(self, path, fh):
        if fh in self._open_files:
            del self._open_files[fh]
        return 0


class ModalFuse(fuse.Operations):
    def __init__(self, ops):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount-point", required=True)
    args = parser.parse_args()

    mount_point = args.mount_point
    os.makedirs(mount_point, exist_ok=True)

    input_stream = sys.stdin
    output_stream = sys.stdout
    sys.stdout = sys.stderr

    client = FuseClient(input_stream, output_stream)

    print(f"[modal-fuse] Initializing...", file=sys.stderr)
    resp = client.init()
    if resp.get("error"):
        print(f"[modal-fuse] Init failed: {resp}", file=sys.stderr)
        sys.exit(1)

    print(f"[modal-fuse] Connected, starting FUSE at {mount_point}", file=sys.stderr)
    ops = FuseOperations(client)

    fuse.FUSE(ModalFuse(ops), mount_point, foreground=True, ro=True, nothreads=False)


if __name__ == "__main__":
    main()
"""

# Path to the virtual environment for FUSE dependencies
FUSE_VENV_PATH = "/__modal_fuse_venv"


class FuseMountManager:
    """Manages FUSE-based local directory mounts for Modal containers.

    This class handles:
    - Injecting the FUSE daemon script into the container
    - Starting the local file server
    - Bridging communication between the file server and FUSE daemon
    - Cleanup when the mount is no longer needed
    """

    def __init__(
        self,
        sandbox: "_Sandbox",
        local_path: Path,
        remote_path: PurePosixPath,
        read_only: bool = True,
    ):
        """Initialize the mount manager.

        Args:
            sandbox: The Modal sandbox to mount into.
            local_path: Local directory to mount.
            remote_path: Path in the container where the directory will be mounted.
            read_only: If True, the mount is read-only.
        """
        self.sandbox = sandbox
        self.local_path = Path(local_path).resolve()
        self.remote_path = remote_path
        self.read_only = read_only

        self._file_server: Optional[AsyncLocalFileServer] = None
        self._fuse_process: Optional[_ContainerProcess[Any]] = None
        self._bridge_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._running = False

    async def start(self) -> None:
        """Start the FUSE mount.

        This will:
        1. Install FUSE dependencies (fuse3 and fusepy)
        2. Start the local file server
        3. Inject and start the FUSE daemon in the container
        4. Bridge communication between them
        """
        if self._running:
            return

        logger.debug(f"Starting FUSE mount: {self.local_path} -> {self.remote_path}")

        # Create file server
        self._file_server = AsyncLocalFileServer(self.local_path, read_only=self.read_only)

        # Set up FUSE dependencies in the container
        await self._setup_fuse_environment()

        # Inject the FUSE daemon script into the container
        await self._inject_fuse_daemon()

        # Start the FUSE daemon process using the venv Python
        self._fuse_process = await self.sandbox._exec(
            f"{FUSE_VENV_PATH}/bin/python",
            "/__modal_fuse_daemon.py",
            "--mount-point",
            str(self.remote_path),
            text=False,
        )

        # Start the stderr logger task (for debugging)
        self._stderr_task = asyncio.create_task(self._run_stderr_logger())

        # Start the bridge task
        self._bridge_task = asyncio.create_task(self._run_bridge())
        self._running = True

        # Wait a moment for the FUSE daemon to initialize
        await asyncio.sleep(0.5)

        logger.debug(f"FUSE mount started: {self.local_path} -> {self.remote_path}")

    async def _setup_fuse_environment(self) -> None:
        """Install fuse3 and set up a venv with fusepy."""
        logger.debug("Setting up FUSE environment in container...")

        # Install fuse3 using apt-get
        install_fuse = await self.sandbox._exec(
            "/bin/bash",
            "-c",
            "apt-get update -qq && apt-get install -y -qq fuse3 > /dev/null 2>&1",
        )
        await install_fuse.wait()

        # Create a virtual environment and install fusepy
        setup_venv = await self.sandbox._exec(
            "/bin/bash",
            "-c",
            f"python3 -m venv {FUSE_VENV_PATH} && {FUSE_VENV_PATH}/bin/pip install -q fusepy",
        )
        await setup_venv.wait()

        logger.debug("FUSE environment setup complete")

    async def _inject_fuse_daemon(self) -> None:
        """Inject the FUSE daemon script into the container."""
        # Write the daemon script to a known location
        script_path = "/__modal_fuse_daemon.py"

        # Use the sandbox file API to write the script
        f = await self.sandbox.open(script_path, "w")
        await f.write(FUSE_DAEMON_SCRIPT)
        await f.close()

        logger.debug(f"Injected FUSE daemon script at {script_path}")

    async def _run_stderr_logger(self) -> None:
        """Read and log stderr from the FUSE daemon for debugging."""
        if self._fuse_process is None:
            return

        try:
            async for chunk in self._fuse_process.stderr:
                if chunk:
                    # Log each line from stderr
                    for line in chunk.strip().split(b"\n"):
                        if line:
                            logger.debug(f"[FUSE daemon] {line.decode('utf-8', errors='replace')}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Error reading FUSE daemon stderr: {e}")

    async def _run_bridge(self) -> None:
        """Bridge communication between the file server and FUSE daemon."""
        if self._file_server is None or self._fuse_process is None:
            return

        # Buffer for incomplete lines
        buffer = b""

        try:
            # Iterate over stdout chunks as they arrive
            async for chunk in self._fuse_process.stdout:
                if not chunk:
                    continue

                # Add chunk to buffer and process complete lines
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue

                    try:
                        request = FuseRequest.from_json(line.decode("utf-8"))
                    except Exception as e:
                        logger.warning(f"Failed to parse FUSE request: {e}, line: {line[:100]}")
                        continue

                    # Handle the request
                    response = await self._file_server.handle_request_async(request)

                    # Send response to FUSE daemon stdin
                    response_line = response.to_json() + "\n"
                    self._fuse_process.stdin.write(response_line.encode("utf-8"))
                    await self._fuse_process.stdin.drain()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Error in FUSE bridge: {e}")
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the FUSE mount."""
        if not self._running:
            return

        self._running = False

        # Cancel the bridge task
        if self._bridge_task:
            self._bridge_task.cancel()
            try:
                await self._bridge_task
            except asyncio.CancelledError:
                pass

        # Cancel the stderr logger task
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        # Terminate the FUSE daemon
        # The FUSE daemon should exit gracefully when stdin is closed
        if self._fuse_process:
            try:
                # Close stdin to signal shutdown
                self._fuse_process.stdin.write_eof()
            except Exception:
                pass

        logger.debug(f"FUSE mount stopped: {self.local_path} -> {self.remote_path}")

    async def __aenter__(self) -> "FuseMountManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


async def setup_fuse_mounts(
    sandbox: "_Sandbox",
    mounts: list[tuple[Path, PurePosixPath]],
) -> list[FuseMountManager]:
    """Set up multiple FUSE mounts for a sandbox.

    Args:
        sandbox: The Modal sandbox.
        mounts: List of (local_path, remote_path) tuples.

    Returns:
        List of FuseMountManager instances.
    """
    managers = []
    for local_path, remote_path in mounts:
        manager = FuseMountManager(sandbox, local_path, remote_path)
        await manager.start()
        managers.append(manager)
    return managers


async def teardown_fuse_mounts(managers: list[FuseMountManager]) -> None:
    """Tear down FUSE mounts."""
    for manager in managers:
        await manager.stop()
