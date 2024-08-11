# Copyright Modal Labs 2024
import asyncio
import platform
from typing import List, Optional

from grpclib import Status
from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.console import Console

from modal_proto import api_pb2

from ._pty import get_pty_info
from ._utils.async_utils import TaskContext
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream
from ._utils.shell_utils import stream_from_stdin, write_to_fd
from .client import _Client
from .config import config
from .exception import ExecutionError, InteractiveTimeoutError, NotFoundError


async def container_exec(
    task_id: str, command: List[str], *, pty: bool, client: Optional[_Client] = None, console: Optional[Console] = None
):
    """Execute a command inside an active container"""
    if platform.system() == "Windows":
        print("container exec is not currently supported on Windows.")
        return

    if client is None:
        client = await _Client.from_env()

    if console is None:
        console = Console()

    connecting_status = console.status("Connecting...")
    connecting_status.start()

    try:
        res: api_pb2.ContainerExecResponse = await client.stub.ContainerExec(
            api_pb2.ContainerExecRequest(
                task_id=task_id,
                command=command,
                pty_info=get_pty_info(shell=True) if pty else None,
                runtime_debug=config.get("function_runtime_debug"),
            )
        )
    except GRPCError as err:
        connecting_status.stop()
        if err.status == Status.NOT_FOUND:
            raise NotFoundError(err.message)
        raise

    async def _handle_input(data: bytes, message_index: int):
        await retry_transient_errors(
            client.stub.ContainerExecPutInput,
            api_pb2.ContainerExecPutInputRequest(
                exec_id=res.exec_id, input=api_pb2.RuntimeInputMessage(message=data, message_index=message_index)
            ),
            total_timeout=10,
        )

    on_connect = asyncio.Event()
    async with TaskContext() as tc:
        exec_output_task = tc.create_task(_handle_exec_output(client, res.exec_id, on_connect))
        try:
            # time out if we can't connect to the server fast enough
            await asyncio.wait_for(on_connect.wait(), timeout=15)

            if connecting_status:
                connecting_status.stop()

            async with stream_from_stdin(_handle_input, use_raw_terminal=pty):
                exit_status = await exec_output_task

            if exit_status != 0:
                raise ExecutionError(f"Process exited with status code {exit_status}")

        except (asyncio.TimeoutError, TimeoutError):
            if connecting_status:
                connecting_status.stop()

            exec_output_task.cancel()
            raise InteractiveTimeoutError("Failed to establish connection to container. Please try again.")


async def _handle_exec_output(client: _Client, exec_id: str, on_connect: Optional[asyncio.Event] = None) -> int:
    """
    Streams exec output to current terminal's stdout.

    The on_connect event will be set when the client connects to the running process,
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
                assert message.file_descriptor in [1, 2]

                await write_to_fd(message.file_descriptor, str.encode(message.message))

            if not connected:
                connected = True
                on_connect.set()
                # give up the event loop
                await asyncio.sleep(0)

            if batch.HasField("exit_code"):
                exit_status = batch.exit_code
                break
            last_batch_index = batch.batch_index

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
