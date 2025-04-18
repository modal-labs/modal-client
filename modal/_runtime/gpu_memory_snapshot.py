# Copyright Modal Labs 2022
#
# This module provides a simple interface for creating GPU memory snapshots,
# provising a convenient interface to `cuda-checkpoint` [1]. This is intended
# to be used in conjunction with memory snapshots.
#
# [1] https://github.com/NVIDIA/cuda-checkpoint

import subprocess
import time
from enum import Enum

from modal.config import config, logger

CUDA_CHECKPOINT_PATH: str = config.get("cuda_checkpoint_path")
CUDA_PIDS: list[int] = []  # list of PIDs with active CUDA sessions


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
    global CUDA_PIDS
    CUDA_PIDS = get_cuda_pids()

    for pid in CUDA_PIDS:
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


def get_cuda_states() -> list[CudaCheckpointState]:
    """Get current CUDA checkpoint state for this process."""
    global CUDA_PIDS

    logger.debug(f"Tracking the checkpointing state of {len(CUDA_PIDS)} CUDA sesisons")

    pid_states: list[CudaCheckpointState] = []
    for pid in CUDA_PIDS:
        try:
            result = subprocess.run(
                [CUDA_CHECKPOINT_PATH, "--get-state", "--pid", str(pid)], check=True, capture_output=True, text=True
            )

            # Parse output to get state
            state_str = result.stdout.strip().lower()
            pid_states.append(CudaCheckpointState(state_str))

        except subprocess.CalledProcessError as e:
            logger.debug(f"Failed to get CUDA checkpoint state: {e.stderr}")
            raise CudaCheckpointException(e.stderr)

    return pid_states


def wait_for_state(target_state: CudaCheckpointState, timeout_secs: float = 5.0):
    """Wait for CUDA checkpoint to reach a specific state."""
    logger.debug(f"Waiting for CUDA checkpoint state {target_state.value}")
    start_time = time.monotonic()

    while True:
        cuda_states = get_cuda_states()
        if all(cuda_state == target_state for cuda_state in cuda_states):
            logger.debug(f"All CUDA sessions reached {target_state.value}")
            break

        if any(cuda_state == CudaCheckpointState.FAILED for cuda_state in cuda_states):
            raise CudaCheckpointException("CUDA session in failed state")

        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_secs:
            raise CudaCheckpointException(f"Timeout after {elapsed:.2f}s waiting for state {target_state.value}")

        time.sleep(0.1)


def get_cuda_pids() -> list[int]:
    """Returns the PIDs of processes that have an active CUDA session."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=True
    )

    pids = [int(pid.strip()) for pid in result.stdout.strip().split("\n") if pid.strip()]
    return pids
