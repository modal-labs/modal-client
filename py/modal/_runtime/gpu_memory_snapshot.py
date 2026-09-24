# Copyright Modal Labs 2022
#
# NOTE: Do not modify this file. GPU memory checkpoint/restore
# (`cuda-checkpoint`) is driven by gVisor, Modal's container runtime. If you
# feel the need to modify this file, please reach out to Modal support.

import logging
import os

logger = logging.getLogger("modal-client")

# Kept in sync with the runtime's GPU_SNAPSHOT_RESTORE_FAILED_EXIT_CODE.
SNAPSHOT_RESTORE_FAILED_EXIT_CODE: int = 222

# Reading this file blocks until restore has completed, including the GPU
# memory restore. The read yields "restore" (this is a restored instance),
# "resume" (the checkpointed instance is running again), or "error".
RUNTIME_CHECKPOINT_WAIT_PATH: str = "/proc/gvisor/checkpoint"


class CudaCheckpointException(Exception):
    """Exception raised for CUDA checkpoint operations."""


class CudaCheckpointSession:
    """The container runtime checkpoints and restores GPU memory itself, and
    this session only waits for it."""

    def __init__(self) -> None:
        self.cuda_processes: list = []
        self._runtime_wait_fd: int = -1

    def checkpoint(self) -> None:
        # Open the wait handle *before* the snapshot is taken. This is important
        # because the open registers interest in the *next* checkpoint.
        try:
            self._runtime_wait_fd = os.open(RUNTIME_CHECKPOINT_WAIT_PATH, os.O_RDONLY)
        except OSError as exc:
            raise CudaCheckpointException(f"Failed to open {RUNTIME_CHECKPOINT_WAIT_PATH}: {exc}") from exc

    def restore(self) -> None:
        if self._runtime_wait_fd < 0:
            return
        try:
            result = os.read(self._runtime_wait_fd, 16).decode("utf-8", errors="replace").strip()
        except OSError as exc:
            raise CudaCheckpointException(f"Failed to read {RUNTIME_CHECKPOINT_WAIT_PATH}: {exc}") from exc
        finally:
            try:
                os.close(self._runtime_wait_fd)
            except OSError:
                pass
            self._runtime_wait_fd = -1
        if result not in ("restore", "resume"):
            raise CudaCheckpointException(f"Container runtime GPU memory restore failed: {result!r}")
        logger.debug(f"Container runtime GPU memory {result} succeeded.")
