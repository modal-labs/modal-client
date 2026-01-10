# Copyright Modal Labs 2025
#
# This module provides a simple interface for creating GPU memory snapshots,
# using the CUDA Driver API for checkpoint/restore operations. This is intended
# to be used in conjunction with memory snapshots.
#
# CUDA Checkpoint API: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html

import ctypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable

from modal.config import logger

# Maximum total duration for lock operation in milliseconds.
CUDA_CHECKPOINT_LOCK_TIMEOUT_MS: int = 3 * 60 * 1000


class CUresult(IntEnum):
    """CUDA Driver API error codes."""

    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_NOT_INITIALIZED = 3
    CUDA_ERROR_ILLEGAL_STATE = 401
    CUDA_ERROR_NOT_SUPPORTED = 801
    CUDA_ERROR_NOT_READY = 600


class CUprocessState(IntEnum):
    """CUDA process checkpoint state from the CUDA Driver API.

    https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
    """

    CU_PROCESS_STATE_RUNNING = 0
    CU_PROCESS_STATE_LOCKED = 1
    CU_PROCESS_STATE_CHECKPOINTED = 2
    CU_PROCESS_STATE_FAILED = 3


class CudaCheckpointException(Exception):
    """Exception raised for CUDA checkpoint operations."""

    pass


class CudaDriverError(CudaCheckpointException):
    """Exception raised when a CUDA Driver API call fails."""

    function_name: str
    result: int

    def __init__(self, function_name: str, result: int):
        self.function_name = function_name
        self.result = result
        try:
            error_name = CUresult(result).name
        except ValueError:
            error_name = f"UNKNOWN_ERROR({result})"
        super().__init__(f"{function_name} failed with {error_name}")


class CudaDriver:
    """Wrapper for CUDA Driver API checkpoint functions using ctypes."""

    _instance: "CudaDriver | None" = None
    _libcuda: ctypes.CDLL
    _cuCheckpointProcessGetState: Callable[..., int]
    _cuCheckpointProcessLock: Callable[..., int]
    _cuCheckpointProcessCheckpoint: Callable[..., int]
    _cuCheckpointProcessRestore: Callable[..., int]
    _cuCheckpointProcessUnlock: Callable[..., int]

    def __init__(self) -> None:
        """Load libcuda and set up function signatures."""
        try:
            self._libcuda = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise CudaCheckpointException(f"Failed to load libcuda.so.1: {e}")

        # cuCheckpointProcessGetState(int pid, CUprocessState* state)
        self._cuCheckpointProcessGetState = self._libcuda.cuCheckpointProcessGetState
        self._cuCheckpointProcessGetState.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._cuCheckpointProcessGetState.restype = ctypes.c_int

        # cuCheckpointProcessLock(int pid, CUcheckpointLockArgs* args)
        self._cuCheckpointProcessLock = self._libcuda.cuCheckpointProcessLock
        self._cuCheckpointProcessLock.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self._cuCheckpointProcessLock.restype = ctypes.c_int

        # cuCheckpointProcessCheckpoint(int pid, CUcheckpointCheckpointArgs* args)
        self._cuCheckpointProcessCheckpoint = self._libcuda.cuCheckpointProcessCheckpoint
        self._cuCheckpointProcessCheckpoint.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self._cuCheckpointProcessCheckpoint.restype = ctypes.c_int

        # cuCheckpointProcessRestore(int pid, CUcheckpointRestoreArgs* args)
        self._cuCheckpointProcessRestore = self._libcuda.cuCheckpointProcessRestore
        self._cuCheckpointProcessRestore.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self._cuCheckpointProcessRestore.restype = ctypes.c_int

        # cuCheckpointProcessUnlock(int pid, CUcheckpointUnlockArgs* args)
        self._cuCheckpointProcessUnlock = self._libcuda.cuCheckpointProcessUnlock
        self._cuCheckpointProcessUnlock.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self._cuCheckpointProcessUnlock.restype = ctypes.c_int

    def __new__(cls) -> "CudaDriver":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_process_state(self, pid: int) -> CUprocessState:
        """Get the checkpoint state of a CUDA process."""
        state = ctypes.c_int()
        result: int = self._cuCheckpointProcessGetState(pid, ctypes.byref(state))
        if result != CUresult.CUDA_SUCCESS:
            raise CudaDriverError("cuCheckpointProcessGetState", result)
        return CUprocessState(state.value)

    def lock_process(self, pid: int, timeout_ms: int | None = None) -> None:
        """Lock a CUDA process, blocking further CUDA API calls."""
        if timeout_ms is not None:
            # CUcheckpointLockArgs structure from CUDA Driver API:
            # - reserved0: unsigned int, must be zero
            # - reserved1: cuuint64_t[7], must be zeroed
            # - timeoutMs: unsigned int, timeout in milliseconds
            class CUcheckpointLockArgs(ctypes.Structure):
                _fields_ = [
                    ("reserved0", ctypes.c_uint),
                    ("reserved1", ctypes.c_uint64 * 7),
                    ("timeoutMs", ctypes.c_uint),
                ]

            args = CUcheckpointLockArgs(reserved0=0, reserved1=(ctypes.c_uint64 * 7)(), timeoutMs=timeout_ms)
            result: int = self._cuCheckpointProcessLock(pid, ctypes.byref(args))
        else:
            result = self._cuCheckpointProcessLock(pid, None)

        if result != CUresult.CUDA_SUCCESS:
            raise CudaDriverError("cuCheckpointProcessLock", result)

    def checkpoint_process(self, pid: int) -> None:
        """Checkpoint a locked CUDA process, copying GPU memory to CPU."""
        result: int = self._cuCheckpointProcessCheckpoint(pid, None)
        if result != CUresult.CUDA_SUCCESS:
            raise CudaDriverError("cuCheckpointProcessCheckpoint", result)

    def restore_process(self, pid: int) -> None:
        """Restore a checkpointed CUDA process, copying memory back to GPU."""
        result: int = self._cuCheckpointProcessRestore(pid, None)
        if result != CUresult.CUDA_SUCCESS:
            raise CudaDriverError("cuCheckpointProcessRestore", result)

    def unlock_process(self, pid: int) -> None:
        """Unlock a CUDA process, allowing CUDA API calls to proceed."""
        result: int = self._cuCheckpointProcessUnlock(pid, None)
        if result != CUresult.CUDA_SUCCESS:
            raise CudaDriverError("cuCheckpointProcessUnlock", result)


def _cu_process_state_to_str(state: CUprocessState) -> str:
    """Convert CUprocessState to a human-readable string."""
    mapping = {
        CUprocessState.CU_PROCESS_STATE_RUNNING: "running",
        CUprocessState.CU_PROCESS_STATE_LOCKED: "locked",
        CUprocessState.CU_PROCESS_STATE_CHECKPOINTED: "checkpointed",
        CUprocessState.CU_PROCESS_STATE_FAILED: "failed",
    }
    return mapping.get(state, f"unknown({state})")


@dataclass
class CudaCheckpointProcess:
    """Contains a reference to a PID with active CUDA session. This also provides
    methods for checkpointing and restoring GPU memory."""

    pid: int
    state: CUprocessState
    _driver: CudaDriver = field(repr=False)

    def checkpoint(self) -> None:
        """Checkpoint CUDA state for this process.

        This moves GPU memory to CPU by executing:
        1. lock (with timeout) - blocks CUDA API calls and waits for pending work
        2. checkpoint - copies device memory to host and releases GPU resources

        The process transitions: RUNNING -> LOCKED -> CHECKPOINTED
        """
        logger.debug(f"PID: {self.pid} Starting checkpoint operation")
        start_time = time.perf_counter()

        try:
            # Step 1: Lock the process (waits for CUDA work to complete)
            self._driver.lock_process(self.pid, timeout_ms=CUDA_CHECKPOINT_LOCK_TIMEOUT_MS)
            logger.debug(f"PID: {self.pid} Lock succeeded")

            # Step 2: Checkpoint (copy GPU memory to CPU)
            self._driver.checkpoint_process(self.pid)
            logger.debug(f"PID: {self.pid} Checkpoint succeeded")

            self.state = CUprocessState.CU_PROCESS_STATE_CHECKPOINTED
            elapsed = time.perf_counter() - start_time
            logger.debug(f"PID: {self.pid} Checkpoint completed in {elapsed:.3f}s")

        except CudaDriverError as e:
            logger.debug(f"PID: {self.pid} Checkpoint failed: {e}")
            self._try_refresh_state()
            raise CudaCheckpointException(f"PID: {self.pid} Checkpoint failed: {e}")

    def restore(self) -> None:
        """Restore CUDA state for this process.

        This moves memory back from CPU to GPU by executing:
        1. restore - re-acquires GPUs and copies memory back to device
        2. unlock - allows CUDA API calls to proceed

        The process transitions: CHECKPOINTED -> LOCKED -> RUNNING
        """
        logger.debug(f"PID: {self.pid} Starting restore operation")
        start_time = time.perf_counter()

        try:
            # Step 1: Restore (copy CPU memory back to GPU)
            self._driver.restore_process(self.pid)
            logger.debug(f"PID: {self.pid} Restore succeeded")

            # Step 2: Unlock (allow CUDA API calls)
            self._driver.unlock_process(self.pid)
            logger.debug(f"PID: {self.pid} Unlock succeeded")

            self.state = CUprocessState.CU_PROCESS_STATE_RUNNING
            elapsed = time.perf_counter() - start_time
            logger.debug(f"PID: {self.pid} Restore completed in {elapsed:.3f}s")

        except CudaDriverError as e:
            logger.debug(f"PID: {self.pid} Restore failed: {e}")
            self._try_refresh_state()
            raise CudaCheckpointException(f"PID: {self.pid} Restore failed: {e}")

    def refresh_state(self) -> None:
        """Refreshes the current CUDA checkpoint state for this process."""
        try:
            self.state = self._driver.get_process_state(self.pid)
        except CudaDriverError as e:
            raise CudaCheckpointException(f"PID: {self.pid} Failed to get state: {e}")

    def _try_refresh_state(self) -> None:
        """Try to refresh state, ignoring errors."""
        try:
            self.refresh_state()
        except CudaCheckpointException:
            pass


class CudaCheckpointSession:
    """Manages the checkpointing state of processes with active CUDA sessions."""

    _driver: CudaDriver
    cuda_processes: list[CudaCheckpointProcess]

    def __init__(self):
        self._driver = CudaDriver()
        self.cuda_processes = self._get_cuda_pids()
        if self.cuda_processes:
            logger.debug(
                f"Found {len(self.cuda_processes)} PID(s) with CUDA sessions: {[c.pid for c in self.cuda_processes]}"
            )
        else:
            logger.debug("No CUDA sessions found.")

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

    def _check_cuda_session(self, pid: int) -> CudaCheckpointProcess | None:
        """Check if a specific PID has a CUDA session."""
        try:
            state = self._driver.get_process_state(pid)
            return CudaCheckpointProcess(pid=pid, state=state, _driver=self._driver)
        except CudaDriverError:
            # API call failed, which is expected for PIDs without CUDA sessions
            return None
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
            if proc.state != CUprocessState.CU_PROCESS_STATE_RUNNING:
                raise CudaCheckpointException(
                    f"PID {proc.pid}: CUDA session not in running state. "
                    + f"Current state: {_cu_process_state_to_str(proc.state)}"
                )

        # Checkpoint all processes in parallel
        start = time.perf_counter()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(proc.checkpoint) for proc in self.cuda_processes]

            # Wait for all futures and collect any exceptions
            exceptions: list[Exception] = []
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
            exceptions: list[Exception] = []
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

    def get_process_states(self) -> list[tuple[int, CUprocessState]]:
        """Get current states of all managed processes."""
        states: list[tuple[int, CUprocessState]] = []
        for proc in self.cuda_processes:
            proc.refresh_state()
            states.append((proc.pid, proc.state))
        return states
