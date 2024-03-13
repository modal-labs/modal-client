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
from .exception import ExecutionError, InteractiveTimeoutError, NotFoundError


async def container_exec(
    task_id: str, command: List[str], *, pty: bool, client: _Client, terminate_container_on_exit: bool = False
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
            )
        )
    except GRPCError as err:
        connecting_status.stop()
        if err.status == Status.NOT_FOUND:
            raise NotFoundError(f"Container ID {task_id} not found")
        raise

    await connect_to_exec(res.exec_id, pty, connecting_status)


async def connect_to_exec(exec_id: str, pty: bool = False, connecting_status: Optional[rich.status.Status] = None):
    """
    Connects the current terminal to the given exec id.

    If connecting_status is given, this function will stop the status spinner upon connection or error.
    """

    client = await _Client.from_env()

    def stop_connecting_status():
        if connecting_status:
            connecting_status.stop()

    on_connect = asyncio.Event()
    async with TaskContext() as tc:
        exec_output_task = tc.create_task(handle_exec_output(client, exec_id, on_connect=on_connect))
        try:
            # time out if we can't connect to the server fast enough
            await asyncio.wait_for(on_connect.wait(), timeout=15)
            stop_connecting_status()

            async with handle_exec_input(client, exec_id, use_raw_terminal=pty):
                exit_status = await exec_output_task

            if exit_status != 0:
                raise ExecutionError(f"Process exited with status code {exit_status}")

        except (asyncio.TimeoutError, TimeoutError):
            stop_connecting_status()
            exec_output_task.cancel()
            raise InteractiveTimeoutError("Failed to establish connection to container.")


# note: this is very similar to code in _pty.py.
@contextlib.asynccontextmanager
async def handle_exec_input(client: _Client, exec_id: str, use_raw_terminal=False):
    quit_pipe_read, quit_pipe_write = os.pipe()

    set_nonblocking(sys.stdin.fileno())

    @asyncify
    def _read_stdin() -> Optional[bytes]:
        nonlocal quit_pipe_read
        # TODO: Windows support.
        (readable, _, _) = select.select([sys.stdin.buffer, quit_pipe_read], [], [], 5)
        if quit_pipe_read in readable:
            return None
        if sys.stdin.buffer in readable:
            return sys.stdin.buffer.read()
        # we had 5 seconds of no input. send an empty string as a "heartbeat" to the server.
        return b""

    async def _write():
        message_index = 1
        while True:
            data = await _read_stdin()
            if data is None:
                return

            await retry_transient_errors(
                client.stub.ContainerExecPutInput,
                api_pb2.ContainerExecPutInputRequest(
                    exec_id=exec_id, input=api_pb2.RuntimeInputMessage(message=data, message_index=message_index)
                ),
                total_timeout=10,
            )
            message_index += 1

    write_task = asyncio.create_task(_write())

    if use_raw_terminal:
        with raw_terminal():
            yield
    else:
        yield

    os.write(quit_pipe_write, b"\n")
    write_task.cancel()


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


def _write_to_fd(fd: int, data: bytes):
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    def try_write():
        try:
            nbytes = os.write(fd, data)
            loop.remove_writer(fd)
            future.set_result(nbytes)
        except OSError as e:
            if e.errno != errno.EAGAIN:
                future.set_exception(e)
                raise

    loop.add_writer(fd, try_write)
    return future
