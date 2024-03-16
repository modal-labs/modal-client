# Copyright Modal Labs 2024
import asyncio
from typing import Optional

from ._utils.shell_utils import connect_to_terminal, write_to_fd
from .sandbox import _Sandbox


async def connect_to_sandbox(sandbox: _Sandbox):
    async def _handle_input(data: bytes, _message_index: int):
        sandbox.stdin.write(data)
        await sandbox.stdin.drain.aio()

    async def _stream_to_stdout(on_connect: asyncio.Event) -> int:
        return await stream_sandbox_logs_to_stdout(sandbox, on_connect)

    await connect_to_terminal(_handle_input, _stream_to_stdout, pty=True)


async def stream_sandbox_logs_to_stdout(sandbox: _Sandbox, on_connect: Optional[asyncio.Event] = None) -> int:
    """
    Streams sandbox output to stdout.

    If given, on_connect will be set when the client connects to the running process,
    and the event loop will be released.
    """

    # we are connected if we received at least one message from the server
    # (the server will send an empty message when the process spawns)
    connected = False

    async for message in sandbox.stdout:
        assert message.file_descriptor in [1, 2]
        await write_to_fd(message.file_descriptor, str.encode(message.data))

        if not connected:
            connected = True
            if on_connect is not None:
                on_connect.set()
                # give up the event loop
                await asyncio.sleep(0)
    # Right now we don't propagate the exit_status to the TaskLogs, so setting
    # exit status to 0.
    return 0
