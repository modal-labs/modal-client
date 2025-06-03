# Copyright Modal Labs 2022
#
# This module provides a simple interface for creating GPU memory snapshots,
# provising a convenient interface to `cuda-checkpoint` [1]. This is intended
# to be used in conjunction with memory snapshots.
#
# [1] https://github.com/NVIDIA/cuda-checkpoint

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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


@dataclass
class CudaCheckpointProcess:
    """Contains a reference to a PID with active CUDA session. This also provides
    methods for checkpointing and restoring GPU memory."""

    pid: int
    state: CudaCheckpointState

    def toggle(self, target_state: CudaCheckpointState, timeout_secs: float = 5 * 60.0):
        """Toggle CUDA checkpoint state for current process, moving GPU memory to the
        CPU and back depending on the current process state when called."""
        logger.debug(f"PID: {self.pid} Toggling CUDA checkpoint state to {target_state.value}")

        start_time = time.monotonic()

        while self._should_continue_toggle(target_state, start_time, timeout_secs):
            self._execute_toggle_command()
            time.sleep(0.1)

        logger.debug(f"PID: {self.pid} Target state {target_state.value} reached")

    def _should_continue_toggle(
        self, target_state: CudaCheckpointState, start_time: float, timeout_secs: float
    ) -> bool:
        """Check if toggle operation should continue based on current state and timeout."""
        self.refresh_state()

        if self.state == target_state:
            return False

        if self.state == CudaCheckpointState.FAILED:
            raise CudaCheckpointException(f"PID: {self.pid} CUDA process state is {self.state}")

        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_secs:
            raise CudaCheckpointException(
                f"PID: {self.pid} Timeout after {elapsed:.2f}s waiting for state {target_state.value}. "
                f"Current state: {self.state}"
            )

        return True

    def _execute_toggle_command(self):
        """Execute the cuda-checkpoint toggle command."""
        try:
            subprocess.run(
                [CUDA_CHECKPOINT_PATH, "--toggle", "--pid", str(self.pid)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug(f"PID: {self.pid} Successfully toggled CUDA checkpoint state")
        except subprocess.CalledProcessError as e:
            logger.debug(f"PID: {self.pid} Failed to toggle CUDA checkpoint state: {e.stderr}")
            raise CudaCheckpointException(e.stderr)

    def refresh_state(self) -> None:
        """Refreshes the current CUDA checkpoint state for this process."""
        try:
            result = subprocess.run(
                [CUDA_CHECKPOINT_PATH, "--get-state", "--pid", str(self.pid)],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )

            state_str = result.stdout.strip().lower()
            self.state = CudaCheckpointState(state_str)

        except subprocess.CalledProcessError as e:
            logger.debug(f"PID: {self.pid} Failed to get CUDA checkpoint state: {e.stderr}")
            raise CudaCheckpointException(e.stderr)


class CudaCheckpointSession:
    """Manages the checkpointing state of processes with active CUDA sessions."""

    def __init__(self):
        self.cuda_processes = self._get_cuda_pids()
        logger.debug(f"PIDs with CUDA sessions: {[c.pid for c in self.cuda_processes]}")

    def _get_cuda_pids(self) -> list[CudaCheckpointProcess]:
        """Iterates over all PIDs and identifies the ones that have running
        CUDA sessions."""
        cuda_pids: list[CudaCheckpointProcess] = []

        # Get all active process IDs from /proc directory
        proc_dir = Path("/proc")
        if not proc_dir.exists():
            raise CudaCheckpointException(
                "OS does not have /proc path rendering it incompatible with GPU memory snapshots."
            )

        for entry in proc_dir.iterdir():
            if not entry.name.isdigit():
                continue

            pid = int(entry.name)
            try:
                # Call cuda-checkpoint to check if this PID has a CUDA session
                result = subprocess.run(
                    [CUDA_CHECKPOINT_PATH, "--get-state", "--pid", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # If the command succeeds (return code 0), this PID has a CUDA session
                if result.returncode == 0:
                    state_str = result.stdout.strip().lower()
                    state = CudaCheckpointState(state_str)

                    cuda_checkpoint_process = CudaCheckpointProcess(pid=pid, state=state)
                    cuda_pids.append(cuda_checkpoint_process)

            # Command failed, which is expected for PIDs without CUDA sessions
            except subprocess.CalledProcessError:
                continue

            # Raise other exceptions
            except subprocess.TimeoutExpired:
                raise CudaCheckpointException(f"Failed to get CUDA state for PID {pid}")
            except Exception as e:
                raise CudaCheckpointException(e)

        # Sort PIDs for ordered checkpointing
        cuda_pids.sort(key=lambda x: x.pid)
        return cuda_pids

    def checkpoint(self) -> None:
        # Validate all states first
        for proc in self.cuda_processes:
            if proc.state != CudaCheckpointState.RUNNING:
                raise CudaCheckpointException(f"CUDA session not in {CudaCheckpointState.RUNNING} state.")

        # Moving state from GPU to CPU can take several seconds per CUDA session.
        # Make a parallel call per CUDA session.
        start = time.perf_counter()

        def checkpoint_impl(proc: CudaCheckpointProcess):
            proc.toggle(CudaCheckpointState.CHECKPOINTED)

        with ThreadPoolExecutor() as executor:
            list(executor.map(checkpoint_impl, self.cuda_processes))

        elapsed = time.perf_counter() - start
        logger.debug(f"Checkpointing CUDA sessions took => {elapsed:.3f}s")

    def restore(self) -> None:
        # Validate all states first
        for proc in self.cuda_processes:
            if proc.state != CudaCheckpointState.CHECKPOINTED:
                raise CudaCheckpointException(f"CUDA session not in {CudaCheckpointState.CHECKPOINTED} state.")

        # See checkpoint() for rationale about parallelism.
        start = time.perf_counter()

        def restore_process(proc: CudaCheckpointProcess):
            proc.toggle(CudaCheckpointState.RUNNING)

        with ThreadPoolExecutor() as executor:
            list(executor.map(restore_process, self.cuda_processes))

        elapsed = time.perf_counter() - start
        logger.debug(f"Restoring CUDA sessions took => {elapsed:.3f}s")
