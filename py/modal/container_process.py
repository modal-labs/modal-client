# Copyright Modal Labs 2024
import asyncio
import platform
from typing import Generic, TypeVar

from modal_proto import api_pb2

from ._utils.async_utils import TaskContext, synchronize_api
from ._utils.shell_utils import stream_from_stdin, write_to_fd
from ._utils.task_command_router_client import TaskCommandRouterClient
from .client import _Client
from .config import logger
from .exception import ExecTimeoutError, InteractiveTimeoutError, InvalidError
from .io_streams import (
    _StreamReader,
    _StreamReaderThroughSandboxExecCommandRouterParams,
    _StreamWriter,
    _StreamWriterThroughCommandRouterSandboxExecParams,
)
from .stream_type import StreamType

T = TypeVar("T", str, bytes)


async def _iter_stream_as_bytes(stream: _StreamReader[T]):
    """Yield raw bytes from a StreamReader regardless of text mode/backend."""
    async for part in stream:
        if isinstance(part, str):
            yield part.encode("utf-8")
        else:
            yield part


class _ContainerProcess(Generic[T]):
    """Represents a running process in a container.

    Container processes communicate via direct communication with
    the Modal worker where the container is running.
    """

    def __init__(
        self,
        process_id: str,
        task_id: str,
        client: _Client,
        command_router_client: TaskCommandRouterClient,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        exec_deadline: float | None = None,
        text: bool = True,
        by_line: bool = False,
    ) -> None:
        self._client = client
        self._command_router_client = command_router_client
        self._process_id = process_id
        self._exec_deadline = exec_deadline
        self._text = text
        self._by_line = by_line
        self._task_id = task_id
        self._stdout = _StreamReader[T](
            _StreamReaderThroughSandboxExecCommandRouterParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                task_id=self._task_id,
                object_id=process_id,
                command_router_client=self._command_router_client,
                deadline=exec_deadline,
            ),
            stream_type=stdout,
            text=text,
            by_line=by_line,
        )
        self._stderr = _StreamReader[T](
            _StreamReaderThroughSandboxExecCommandRouterParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDERR,
                task_id=self._task_id,
                object_id=process_id,
                command_router_client=self._command_router_client,
                deadline=exec_deadline,
            ),
            stream_type=stderr,
            text=text,
            by_line=by_line,
        )
        self._stdin = _StreamWriter(
            _StreamWriterThroughCommandRouterSandboxExecParams(
                task_id=self._task_id,
                object_id=process_id,
                command_router_client=self._command_router_client,
            )
        )
        self._returncode = None

    def __repr__(self) -> str:
        return f"ContainerProcess(process_id={self._process_id!r})"

    @property
    def stdout(self) -> _StreamReader[T]:
        """StreamReader for the container process's stdout stream."""
        return self._stdout

    @property
    def stderr(self) -> _StreamReader[T]:
        """StreamReader for the container process's stderr stream."""
        return self._stderr

    @property
    def stdin(self) -> _StreamWriter:
        """StreamWriter for the container process's stdin stream."""
        return self._stdin

    @property
    def returncode(self) -> int:
        if self._returncode is None:
            raise InvalidError(
                "You must call wait() before accessing the returncode. "
                "To poll for the status of a running process, use poll() instead."
            )
        return self._returncode

    async def poll(self) -> int | None:
        """Check if the container process has finished running.

        Returns `None` if the process is still running, else returns the exit code.
        """
        if self._returncode is not None:
            return self._returncode
        try:
            resp = await self._command_router_client.exec_poll(self._task_id, self._process_id, self._exec_deadline)
            which = resp.WhichOneof("exit_status")
            if which is None:
                return None

            if which == "code":
                self._returncode = int(resp.code)
                return self._returncode
            elif which == "signal":
                self._returncode = 128 + int(resp.signal)
                return self._returncode
            else:
                logger.debug(f"ContainerProcess {self._process_id} exited with unexpected status: {which}")
                raise InvalidError("Unexpected exit status")
        except ExecTimeoutError:
            logger.debug(f"ContainerProcess poll for {self._process_id} did not complete within deadline")
            # TODO(saltzm): This is a weird API, but customers currently may rely on it. This
            # should probably raise an ExecTimeoutError instead.
            self._returncode = -1
            return self._returncode
        except Exception as e:
            # Re-raise non-transient errors or errors resulting from exceeding retries on transient errors.
            logger.warning(f"ContainerProcess poll for {self._process_id} failed: {e}")
            raise

    async def wait(self) -> int:
        """Wait for the container process to finish running. Returns the exit code."""
        if self._returncode is not None:
            return self._returncode

        try:
            resp = await self._command_router_client.exec_wait(self._task_id, self._process_id, self._exec_deadline)
            which = resp.WhichOneof("exit_status")
            if which == "code":
                self._returncode = int(resp.code)
            elif which == "signal":
                self._returncode = 128 + int(resp.signal)
            else:
                logger.debug(f"ContainerProcess {self._process_id} exited with unexpected status: {which}")
                self._returncode = -1
                raise InvalidError("Unexpected exit status")
        except ExecTimeoutError:
            logger.debug(f"ContainerProcess {self._process_id} did not complete within deadline")
            # TODO(saltzm): This is a weird API, but customers currently may rely on it. This
            # should be a ExecTimeoutError.
            self._returncode = -1

        return self._returncode

    async def attach(self):
        """mdmd:hidden"""
        if platform.system() == "Windows":
            print("interactive exec is not currently supported on Windows.")  # noqa: T201
            return

        from .output import OutputManager

        output = OutputManager.get()
        connecting_status = output.status("Connecting...")
        connecting_status.start()
        on_connect = asyncio.Event()

        async def _write_to_fd_loop(stream: _StreamReader[T]):
            async for chunk in _iter_stream_as_bytes(stream):
                if chunk is None:
                    break

                if not on_connect.is_set():
                    connecting_status.stop()
                    on_connect.set()

                await write_to_fd(stream.file_descriptor, chunk)

        async def _handle_input(data: bytes, message_index: int):
            self.stdin.write(data)
            await self.stdin.drain()

        async with TaskContext() as tc:
            stdout_task = tc.create_task(_write_to_fd_loop(self.stdout))
            stderr_task = tc.create_task(_write_to_fd_loop(self.stderr))

            try:
                # Time out if we can't connect fast enough.
                await asyncio.wait_for(on_connect.wait(), timeout=60)

                async with stream_from_stdin(_handle_input, use_raw_terminal=True):
                    await stdout_task
                    await stderr_task

            except (asyncio.TimeoutError, TimeoutError):
                connecting_status.stop()
                stdout_task.cancel()
                stderr_task.cancel()
                raise InteractiveTimeoutError("Failed to establish connection to container. Please try again.")


ContainerProcess = synchronize_api(_ContainerProcess)
