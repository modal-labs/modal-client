# Copyright Modal Labs 2022
#
# This module provides a simple interface for creating GPU memory snapshots,
# provising a convenient interface to `cuda-checkpoint` [1]. This is intended
# to be used in conjunction with memory snapshots.
#
# [1] https://github.com/NVIDIA/cuda-checkpoint

import os
import subprocess
import time
from enum import Enum

from modal.config import config, logger

CUDA_CHECKPOINT_PATH: str = config.get("cuda_checkpoint_path")


class CudaCheckpointState(Enum):
    """State representation from the CUDA API: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc96cdda177a2b8c296144567cbea4f23"""

    RUNNING = "running"
    LOCKED = "locked"
    CHECKPOINTED = "checkpointed"
    FAILED = "failed"


class CudaCheckpointException(Exception):
    pass


def toggle():
    """Toggle CUDA checkpoint state for current process, moving GPU memory to the
    CPU and back depending on the current process state when called."""
    pid = get_own_pid()
    logger.debug(f"Toggling CUDA checkpoint state for PID {pid}")

    try:
        subprocess.run(
            [
                CUDA_CHECKPOINT_PATH,
                "--toggle",
                "--pid",
                str(pid),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug("Successfully toggled CUDA checkpoint state")

    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to toggle CUDA checkpoint state: {e.stderr}")
        raise CudaCheckpointException(e.stderr)


def get_state() -> CudaCheckpointState:
    """Get current CUDA checkpoint state for this process."""
    pid = get_own_pid()

    try:
        result = subprocess.run(
            [CUDA_CHECKPOINT_PATH, "--get-state", "--pid", str(pid)], check=True, capture_output=True, text=True
        )

        # Parse output to get state
        state_str = result.stdout.strip().lower()
        return CudaCheckpointState(state_str)

    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to get CUDA checkpoint state: {e.stderr}")
        raise CudaCheckpointException(e.stderr)


def wait_for_state(target_state: CudaCheckpointState, timeout_secs: float = 5.0):
    """Wait for CUDA checkpoint to reach a specific state."""
    logger.debug(f"Waiting for CUDA checkpoint state {target_state.value}")
    start_time = time.monotonic()

    while True:
        current_state = get_state()

        if current_state == target_state:
            logger.debug(f"Target state {target_state.value} reached")
            break

        if current_state == CudaCheckpointState.FAILED:
            raise CudaCheckpointException(f"CUDA process state is {current_state}")

        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_secs:
            raise CudaCheckpointException(f"Timeout after {elapsed:.2f}s waiting for state {target_state.value}")

        time.sleep(0.1)


def get_own_pid():
    """Returns the Process ID (PID) of the current Python process
    using only the standard library.
    """
    return os.getpid()
