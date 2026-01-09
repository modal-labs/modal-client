# Copyright Modal Labs 2025
#
# This module provides a simple interface for creating GPU memory snapshots,
# providing a convenient interface to `cuda-checkpoint` [1]. This is intended
# to be used in conjunction with memory snapshots.
#
# [1] https://github.com/NVIDIA/cuda-checkpoint

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from modal.config import config, logger

CUDA_CHECKPOINT_PATH: str = config.get("cuda_checkpoint_path")

# Maximum total duration for each individual `cuda-checkpoint` invocation.
CUDA_CHECKPOINT_TIMEOUT: float = 3 * 60.0

# Default timeout for the lock operation in milliseconds.
# The lock action waits internally for CUDA work to complete.
CUDA_CHECKPOINT_LOCK_TIMEOUT_MS: int = int(CUDA_CHECKPOINT_TIMEOUT * 1000)


class CudaCheckpointState(Enum):
    """State representation from the CUDA API [1].

    [1] https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html"""

    RUNNING = "running"
    LOCKED = "locked"
    CHECKPOINTED = "checkpointed"
    FAILED = "failed"


class CudaCheckpointException(Exception):
    """Exception raised for CUDA checkpoint operations."""

    pass


@dataclass
class CudaCheckpointProcess:
    """Contains a reference to a PID with active CUDA session. This also provides
    methods for checkpointing and restoring GPU memory."""

    pid: int
    state: CudaCheckpointState

    def checkpoint(self) -> None:
        """Checkpoint CUDA state for this process using direct action commands.

        This moves GPU memory to CPU by executing:
        1. lock (with timeout) - blocks CUDA API calls and waits for pending work
        2. checkpoint - copies device memory to host and releases GPU resources

        The process transitions: RUNNING -> LOCKED -> CHECKPOINTED
        """
        logger.debug(f"PID: {self.pid} Starting checkpoint operation")
        start_time = time.perf_counter()

        try:
            # Step 1: Lock the process (waits for CUDA work to complete)
            self._execute_action("lock", timeout_ms=CUDA_CHECKPOINT_LOCK_TIMEOUT_MS)

            # Step 2: Checkpoint (copy GPU memory to CPU)
            self._execute_action("checkpoint")

            self.state = CudaCheckpointState.CHECKPOINTED
            elapsed = time.perf_counter() - start_time
            logger.debug(f"PID: {self.pid} Checkpoint completed in {elapsed:.3f}s")

        except CudaCheckpointException:
            # Try to get current state for better error reporting
            try:
                self.refresh_state()
            except CudaCheckpointException:
                pass
            raise

    def restore(self) -> None:
        """Restore CUDA state for this process using direct action commands.

        This moves memory back from CPU to GPU by executing:
        1. restore - re-acquires GPUs and copies memory back to device
        2. unlock - allows CUDA API calls to proceed

        The process transitions: CHECKPOINTED -> LOCKED -> RUNNING
        """
        logger.debug(f"PID: {self.pid} Starting restore operation")
        start_time = time.perf_counter()

        try:
            # Step 1: Restore (copy CPU memory back to GPU)
            self._execute_action("restore")

            # Step 2: Unlock (allow CUDA API calls)
            self._execute_action("unlock")

            self.state = CudaCheckpointState.RUNNING
            elapsed = time.perf_counter() - start_time
            logger.debug(f"PID: {self.pid} Restore completed in {elapsed:.3f}s")

        except CudaCheckpointException:
            # Try to get current state for better error reporting
            try:
                self.refresh_state()
            except CudaCheckpointException:
                pass
            raise

    def _execute_action(self, action: str, timeout_ms: Optional[int] = None) -> None:
        """Execute a cuda-checkpoint action command."""
        cmd = [CUDA_CHECKPOINT_PATH, "--action", action, "--pid", str(self.pid)]

        if timeout_ms is not None:
            if action != "lock":
                raise ValueError(f"timeout_ms is only valid for 'lock' action, not '{action}'")
            cmd.extend(["--timeout", str(timeout_ms)])

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=CUDA_CHECKPOINT_TIMEOUT,
            )
            logger.debug(f"PID: {self.pid} Action '{action}' succeeded")
            if result.stdout.strip():
                logger.debug(f"PID: {self.pid} stdout: {result.stdout.strip()}")

        except subprocess.CalledProcessError as e:
            error_msg = f"PID: {self.pid} Action '{action}' failed: {e.stderr.strip()}"
            logger.debug(error_msg)
            raise CudaCheckpointException(error_msg)

        except subprocess.TimeoutExpired:
            error_msg = f"PID: {self.pid} Action '{action}' timed out after {CUDA_CHECKPOINT_TIMEOUT}s"
            logger.debug(error_msg)
            raise CudaCheckpointException(error_msg)

    def refresh_state(self) -> None:
        """Refreshes the current CUDA checkpoint state for this process."""
        try:
            result = subprocess.run(
                [CUDA_CHECKPOINT_PATH, "--get-state", "--pid", str(self.pid)],
                check=True,
                capture_output=True,
                text=True,
                timeout=CUDA_CHECKPOINT_TIMEOUT,
            )

            state_str = result.stdout.strip().lower()
            self.state = CudaCheckpointState(state_str)

        except subprocess.CalledProcessError as e:
            error_msg = f"PID: {self.pid} Failed to get CUDA checkpoint state: {e.stderr}"
            logger.debug(error_msg)
            raise CudaCheckpointException(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"PID: {self.pid} Get state command timed out"
            logger.debug(error_msg)
            raise CudaCheckpointException(error_msg)


class CudaCheckpointSession:
    """Manages the checkpointing state of processes with active CUDA sessions."""

    def __init__(self):
        self.cuda_processes = self._get_cuda_pids()
        if self.cuda_processes:
            logger.debug(
                f"Found {len(self.cuda_processes)} PID(s) with CUDA sessions: {[c.pid for c in self.cuda_processes]}"
            )
        else:
            logger.debug("No CUDA sessions found.")

    def _get_cuda_pids(self) -> List[CudaCheckpointProcess]:
        """Iterates over all PIDs and identifies the ones that have running
        CUDA sessions."""
        cuda_pids: List[CudaCheckpointProcess] = []

        # Get all active process IDs from /proc directory
        proc_dir = Path("/proc")
        if not proc_dir.exists():
            raise CudaCheckpointException(
                "OS does not have /proc path rendering it incompatible with GPU memory snapshots."
            )

        # Get all numeric directories (PIDs) from /proc
        pid_dirs = [entry for entry in proc_dir.iterdir() if entry.name.isdigit()]

        # Use ThreadPoolExecutor to check PIDs in parallel for better performance
        with ThreadPoolExecutor(max_workers=min(50, len(pid_dirs))) as executor:
            future_to_pid = {
                executor.submit(self._check_cuda_session, int(entry.name)): int(entry.name) for entry in pid_dirs
            }

            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    cuda_process = future.result()
                    if cuda_process:
                        cuda_pids.append(cuda_process)
                except Exception as e:
                    logger.debug(f"Error checking PID {pid}: {e}")

        # Sort PIDs for ordered checkpointing
        cuda_pids.sort(key=lambda x: x.pid)
        return cuda_pids

    def _check_cuda_session(self, pid: int) -> Optional[CudaCheckpointProcess]:
        """Check if a specific PID has a CUDA session."""
        try:
            result = subprocess.run(
                [CUDA_CHECKPOINT_PATH, "--get-state", "--pid", str(pid)],
                capture_output=True,
                text=True,
                timeout=CUDA_CHECKPOINT_TIMEOUT,
            )

            # If the command succeeds (return code 0), this PID has a CUDA session
            if result.returncode == 0:
                state_str = result.stdout.strip().lower()
                state = CudaCheckpointState(state_str)
                return CudaCheckpointProcess(pid=pid, state=state)

        except subprocess.CalledProcessError:
            # Command failed, which is expected for PIDs without CUDA sessions
            pass
        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout checking CUDA state for PID {pid}")
        except Exception as e:
            logger.debug(f"Error checking PID {pid}: {e}")

        return None

    def checkpoint(self) -> None:
        """Checkpoint all CUDA processes, moving GPU memory to CPU."""
        if not self.cuda_processes:
            logger.debug("No CUDA processes to checkpoint.")
            return

        # Validate all states first
        for proc in self.cuda_processes:
            proc.refresh_state()
            if proc.state != CudaCheckpointState.RUNNING:
                raise CudaCheckpointException(
                    f"PID {proc.pid}: CUDA session not in {CudaCheckpointState.RUNNING.value} state. "
                    f"Current state: {proc.state.value}"
                )

        # Checkpoint all processes in parallel
        start = time.perf_counter()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(proc.checkpoint) for proc in self.cuda_processes]

            # Wait for all futures and collect any exceptions
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    exceptions.append(e)

            if exceptions:
                raise CudaCheckpointException(
                    f"Failed to checkpoint {len(exceptions)} processes: {'; '.join(str(e) for e in exceptions)}"
                )

        elapsed = time.perf_counter() - start
        logger.debug(f"Checkpointing {len(self.cuda_processes)} CUDA sessions took => {elapsed:.3f}s")

    def restore(self) -> None:
        """Restore all CUDA processes, moving memory back from CPU to GPU."""
        if not self.cuda_processes:
            logger.debug("No CUDA sessions to restore.")
            return

        # Restore all processes in parallel
        start = time.perf_counter()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(proc.restore) for proc in self.cuda_processes]

            # Wait for all futures and collect any exceptions
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    exceptions.append(e)

            if exceptions:
                raise CudaCheckpointException(
                    f"Failed to restore {len(exceptions)} processes: {'; '.join(str(e) for e in exceptions)}"
                )

        elapsed = time.perf_counter() - start
        logger.debug(f"Restoring {len(self.cuda_processes)} CUDA session(s) took => {elapsed:.3f}s")

    def get_process_count(self) -> int:
        """Get the number of CUDA processes managed by this session."""
        return len(self.cuda_processes)

    def get_process_states(self) -> List[tuple[int, CudaCheckpointState]]:
        """Get current states of all managed processes."""
        states = []
        for proc in self.cuda_processes:
            proc.refresh_state()
            states.append((proc.pid, proc.state))
        return states
