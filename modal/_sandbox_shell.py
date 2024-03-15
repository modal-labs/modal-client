# Copyright Modal Labs 2024
import asyncio
import contextlib
import errno
import os
import platform
import select
import sys
from typing import List, Optional

import rich
import rich.status
from grpclib import Status
from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.console import Console

from modal_proto import api_pb2


from ._pty import get_pty_info, raw_terminal, set_nonblocking
from ._utils.async_utils import TaskContext, asyncify
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream
from .client import _Client
from .config import config
from .exception import ExecutionError, InteractiveTimeoutError, NotFoundError
from typing import Callable
from ._utils.shell_utils import _write_to_fd
from sandbox import _Sandbox

async def connect_to_sandbox(sandbox: _Sandbox):
    client = await _Client.from_env()

    def stop_connecting_status():
        if connecting_status:
            connecting_status.stop()

    on_connect = asyncio.Event()
    async with TaskContext() as tc:
        exec_output_task = tc.create_task(handle_exec_output(client, exec_id, sandbox, on_connect=on_connect))
        try:
            # time out if we can't connect to the server fast enough
            await asyncio.wait_for(on_connect.wait(), timeout=15)
            stop_connecting_status()

            async def handle_input(data: bytes, message_index: int):
                await retry_transient_errors(
                    client.stub.ContainerExecPutInput,
                    api_pb2.ContainerExecPutInputRequest(
                        exec_id=exec_id, input=api_pb2.RuntimeInputMessage(message=data, message_index=message_index)
                    ),
                    total_timeout=10,
                )

            async with _stream_stdin(handle_input, use_raw_terminal=pty):
                exit_status = await exec_output_task

            if exit_status != 0:
                raise ExecutionError(f"Process exited with status code {exit_status}")

        except (asyncio.TimeoutError, TimeoutError):
            stop_connecting_status()
            exec_output_task.cancel()
            raise InteractiveTimeoutError("Failed to establish connection to container.")


async def handle_exec_output(client: _Client, exec_id: str, write: Callable[[int], int], sandbox: any, on_connect: Optional[asyncio.Event] = None) -> int:
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
                # print(f"Kobe got batch!: {message}")
                assert message.file_descriptor in [1, 2]
                await _write_to_fd(message.file_descriptor, str.encode(message.data))
        
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

