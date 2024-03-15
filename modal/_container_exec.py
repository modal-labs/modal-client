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
from ._utils.shell_utils import _write_to_fd, connect_to_terminal 

from modal_proto import api_pb2


from ._pty import get_pty_info, raw_terminal, set_nonblocking
from ._utils.async_utils import TaskContext, asyncify
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream
from .client import _Client
from .config import config
from .exception import ExecutionError, InteractiveTimeoutError, NotFoundError
from typing import Callable


async def container_exec(
    task_id: str, command: List[str], *, pty: bool, client: _Client, sandbox: any, terminate_container_on_exit: bool = False
):
    """Execute a command inside an active container"""
    if platform.system() == "Windows":
        print("container exec is not currently supported on Windows.")
        return

    client = await _Client.from_env()

    console = Console()
    connecting_status = console.status("Connecting...")
    connecting_status.start()

    try:
        res: api_pb2.ContainerExecResponse = await client.stub.ContainerExec(
            api_pb2.ContainerExecRequest(
                task_id=task_id,
                command=command,
                pty_info=get_pty_info(shell=True) if pty else None,
                terminate_container_on_exit=terminate_container_on_exit,
                runtime_debug=config.get("function_runtime_debug"),
            )
        )
    except GRPCError as err:
        connecting_status.stop()
        if err.status == Status.NOT_FOUND:
            raise NotFoundError(f"Container ID {task_id} not found")
        raise

    await connect_to_exec(res.exec_id, sandbox, pty, connecting_status)

async def connect_to_exec(exec_id: str, sandbox: any, write: Callable[[int], int], pty: bool = False, connecting_status: Optional[rich.status.Status] = None):
    """
    Connects the current terminal to the given exec id.

    If connecting_status is given, this function will stop the status spinner upon connection or error.
    """

    client = await _Client.from_env()

    async def handle_output(on_connect: asyncio.Event):
        await handle_exec_output(client, exec_id, on_connect)

    async def handle_input(data: bytes, message_index: int):
        await retry_transient_errors(
            client.stub.ContainerExecPutInput,
            api_pb2.ContainerExecPutInputRequest(
                exec_id=exec_id, input=api_pb2.RuntimeInputMessage(message=data, message_index=message_index)
            ),
            total_timeout=10,
        )
    await connect_to_terminal(handle_input, handle_output)
    
async def handle_exec_output(client: _Client, exec_id: str, on_connect: Optional[asyncio.Event] = None) -> int:
    """
    Streams exec output to stdout.

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
        req = api_pb2.ContainerExecGetOutputRequest(
            exec_id=exec_id,
            timeout=55,
            last_batch_index=last_batch_index,
        )
        async for batch in unary_stream(client.stub.ContainerExecGetOutput, req):
            for message in batch.items:
                # print(f"Kobe got message!: {message}")
                assert message.file_descriptor in [1, 2]

                await _write_to_fd(message.file_descriptor, str.encode(message.message))

            if not connected:
                connected = True
                if on_connect is not None:
                    on_connect.set()
                    # give up the event loop
                    await asyncio.sleep(0)

            if batch.HasField("exit_code"):
                exit_status = batch.exit_code
                break       

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
