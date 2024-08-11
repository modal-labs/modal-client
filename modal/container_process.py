# Copyright Modal Labs 2024
import asyncio
import platform
from typing import Optional

from rich.console import Console

from modal_proto import api_pb2

from ._utils.async_utils import TaskContext, synchronize_api
from ._utils.shell_utils import stream_from_stdin, write_to_fd
from .client import _Client
from .exception import InteractiveTimeoutError
from .io_streams import _StreamReader, _StreamWriter


class _ContainerProcess:
    _process_id: Optional[str] = None
    _stdout: _StreamReader
    _stderr: _StreamReader
    _stdin: _StreamWriter

    def __init__(self, process_id: str, client: _Client) -> None:
        self._process_id = process_id
        self._client = client
        self._stdout = _StreamReader(api_pb2.FILE_DESCRIPTOR_STDOUT, process_id, "container_process", self._client)
        self._stderr = _StreamReader(api_pb2.FILE_DESCRIPTOR_STDERR, process_id, "container_process", self._client)
        self._stdin = _StreamWriter(process_id, "container_process", self._client)

    @property
    def stdout(self) -> _StreamReader:
        """`StreamReader` for the container process's stdout stream."""

        return self._stdout

    @property
    def stderr(self) -> _StreamReader:
        """`StreamReader` for the container process's stderr stream."""

        return self._stderr

    @property
    def stdin(self) -> _StreamWriter:
        """`StreamWriter` for the container process's stdin stream."""

        return self._stdin

    async def attach(self, *, pty: bool):
        if platform.system() == "Windows":
            print("interactive exec is not currently supported on Windows.")
            return

        console = Console()

        connecting_status = console.status("Connecting...")
        connecting_status.start()
        on_connect = asyncio.Event()

        async def _write_to_fd_loop(stream: _StreamReader):
            async for line in stream:
                if not on_connect.is_set():
                    connecting_status.stop()
                    on_connect.set()

                await write_to_fd(stream.file_descriptor, line.encode("utf-8"))

        async def _handle_input(data: bytes, message_index: int):
            self.stdin.write(data)
            await self.stdin.drain()

        async with TaskContext() as tc:
            stdout_task = tc.create_task(_write_to_fd_loop(self.stdout))
            stderr_task = tc.create_task(_write_to_fd_loop(self.stderr))

            try:
                # time out if we can't connect to the server fast enough
                await asyncio.wait_for(on_connect.wait(), timeout=15)

                async with stream_from_stdin(_handle_input, use_raw_terminal=pty):
                    await stdout_task
                    await stderr_task

                # TODO: this doesn't work right now.
                # if exit_status != 0:
                #     raise ExecutionError(f"Process exited with status code {exit_status}")

            except (asyncio.TimeoutError, TimeoutError):
                connecting_status.stop()
                stdout_task.cancel()
                stderr_task.cancel()
                raise InteractiveTimeoutError("Failed to establish connection to container. Please try again.")


ContainerProcess = synchronize_api(_ContainerProcess)
