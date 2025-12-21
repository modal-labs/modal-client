# Copyright Modal Labs 2025
"""FUSE-based local directory mounting for Modal Shell.

This package provides functionality to mount local directories into Modal
containers using FUSE (Filesystem in Userspace). This allows real-time
synchronization of local files into the container, unlike static mounts
which are uploaded as snapshots.

Usage:
    The main entry point is FuseMountManager, which handles the lifecycle
    of a FUSE mount between a local directory and a Modal container.

Example:
    async with FuseMountManager(sandbox, local_path, remote_path) as mount:
        # The local directory is now accessible at remote_path in the container
        pass
"""

from .file_server import AsyncLocalFileServer, LocalFileServer
from .synchronizer import FuseMountManager, setup_fuse_mounts, teardown_fuse_mounts

__all__ = [
    "LocalFileServer",
    "AsyncLocalFileServer",
    "FuseMountManager",
    "setup_fuse_mounts",
    "teardown_fuse_mounts",
]
