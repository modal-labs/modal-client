# Copyright Modal Labs 2024
import asyncio
from typing import Optional

from grpclib.exceptions import GRPCError, StreamTerminatedError

from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream
from ._utils.shell_utils import write_to_fd, connect_to_terminal
from .sandbox import _Sandbox

async def connect_to_sandbox(sandbox: _Sandbox):
    async def _handle_input(data: bytes):
        print("Writing!")
        sandbox.stdin.write(data)
        print(f"draining sandbox! Data is: {data}")
        await sandbox.stdin.drain.aio()
    async def _stream_to_stdout(on_connect: asyncio.Event):
        print("streaming OUT")
        await stream_sandbox_logs_to_stdout(sandbox, on_connect)
    print("connecting to sandbox")
    await connect_to_terminal(_handle_input, _stream_to_stdout)

async def stream_sandbox_logs_to_stdout(sandbox: _Sandbox, on_connect: Optional[asyncio.Event] = None) -> int:
    """
    Streams sandbox output to stdout.

    If given, on_connect will be set when the client connects to the running process,
    and the event loop will be released.

    Returns the status code of the process.
    """

    last_batch_index = 0
    exit_status = None

    # we are connected if we received at least one message from the server
    # (the server will send an empty message when the process spawns)
    connected = False

    async def _get_output():
        nonlocal last_batch_index, exit_status, connected
        
        async for batch in sandbox.stdout.read_stream.aio():
            for message in batch.items:
                print(f"Kobe got batch!: {message}")
                assert message.file_descriptor in [1, 2]
                await write_to_fd(message.file_descriptor, str.encode(message.data))
        
            if not connected:
                connected = True
                if on_connect is not None:
                    on_connect.set()
                    # give up the event loop
                    await asyncio.sleep(0)

            if batch.eof:
                # exit_status = batch.exit_code
                exit_status = 0
                break
            # last_batch_index = batch.batch_index

    while exit_status is None:
        try:
            await _get_output()
        except (GRPCError, StreamTerminatedError) as exc:
            if isinstance(exc, GRPCError):
                if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                    continue
            elif isinstance(exc, StreamTerminatedError):
                continue
            raise

    return exit_status

