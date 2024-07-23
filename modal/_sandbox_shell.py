# Copyright Modal Labs 2024
import asyncio

from ._utils.shell_utils import connect_to_terminal, write_to_fd
from .sandbox import _Sandbox


async def connect_to_sandbox(sandbox: _Sandbox):
    """
    Connects the current terminal to the Sandbox process.
    """

    async def _handle_input(data: bytes, _):
        sandbox.stdin.write(data)
        await sandbox.stdin.drain.aio()  # type: ignore

    async def _stream_to_stdout(on_connect: asyncio.Event) -> int:
        return await _stream_logs_to_stdout(sandbox, on_connect)

    await connect_to_terminal(_handle_input, _stream_to_stdout, pty=True)


async def _stream_logs_to_stdout(sandbox: _Sandbox, on_connect: asyncio.Event) -> int:
    """
    Streams sandbox output logs to the current terminal's stdout.

    The on_connect event will be set when the client connects to the running process,
    and the event loop will be released.
    """

    # we are connected if we received at least one message from the server
    # (the server will send an empty message when the process spawns)
    connected = False

    # Since the sandbox process will run in a PTY, stderr will go to the PTY
    # slave. The PTY shell will then relay data from PTY master to stdout.
    # Therefore, we only need to stream from/to stdout here.
    async for message in sandbox.stdout:
        await write_to_fd(1, message.encode("utf-8"))

        if not connected:
            connected = True
            on_connect.set()
            # give up the event loop
            await asyncio.sleep(0)

    # Right now we don't propagate the exit_status to the TaskLogs, so setting
    # exit status to 0.
    return 0
