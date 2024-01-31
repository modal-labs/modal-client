# Copyright Modal Labs 2022

import asyncio
import contextlib
import errno
import os
import platform
import select
import sys
from typing import Callable, List, Optional, Union

import rich
import typer
from grpclib import Status
from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.console import Console
from rich.text import Text

from modal._pty import get_pty_info, raw_terminal, set_nonblocking
from modal.cli.utils import display_table, timestamp_to_local
from modal.client import _Client
from modal_proto import api_pb2
from modal_utils.async_utils import asyncify, synchronizer
from modal_utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream

container_cli = typer.Typer(name="container", help="Manage running containers.", no_args_is_help=True)


@container_cli.command("list")
@synchronizer.create_blocking
async def list():
    """List all containers that are currently running"""
    client = await _Client.from_env()
    res: api_pb2.TaskListResponse = await client.stub.TaskList(api_pb2.TaskListRequest())

    column_names = ["Container ID", "App ID", "App Name", "Start time"]
    rows: List[List[Union[Text, str]]] = []
    for task_stats in res.tasks:
        rows.append(
            [
                task_stats.task_id,
                task_stats.app_id,
                task_stats.app_description,
                timestamp_to_local(task_stats.started_at) if task_stats.started_at else "Pending",
            ]
        )

    display_table(column_names, rows, json=False, title="Active Containers")


@container_cli.command("exec")
@synchronizer.create_blocking
async def exec(
    container_id: str = typer.Argument(
        help="The ID of the container to run the command in",
    ),
    command: List[str] = typer.Argument(help="The command to run"),
    tty: bool = typer.Option(is_flag=True, default=True, help="Run the command inside a TTY"),
):
    """Execute a command inside an active container"""
    task_id = container_id
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
                task_id=task_id, command=command, pty_info=get_pty_info(shell=True) if tty else None
            )
        )
    except GRPCError as err:
        connecting_status.stop()
        if err.status == Status.NOT_FOUND:
            rich.print(f"Container ID {task_id} not found", file=sys.stderr)
            raise typer.Exit(code=1)
        raise

    exec_failed = False
    async with handle_exec_input(client, res.exec_id, use_raw_terminal=tty):
        try:
            exit_status = await handle_exec_output(client, res.exec_id, on_connect=connecting_status.stop)
            if exit_status != 0:
                rich.print(f"Process exited with status code {exit_status}", file=sys.stderr)
                exec_failed = True
        except TimeoutError:
            connecting_status.stop()
            rich.print("Failed to establish connection to process", file=sys.stderr)
            exec_failed = True

    if exec_failed:
        # we don't want to raise this inside the context manager
        # since otherwise the context manager cleanup doesn't get called
        raise typer.Exit(code=1)


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


async def handle_exec_output(client: _Client, exec_id: str, on_connect: Optional[Callable] = None) -> int:
    """
    Streams exec output to stdout.

    If given, on_connect will be called when the client connects to the running process.

    Returns the status code of the process.
    """
    # how long to wait for the first server response before we time out
    FIRST_OUTPUT_TIMEOUT = 15

    last_batch_index = 0
    exit_status = None

    # we are connected if we received at least one message from the server
    # (the server will send an empty message when the process spawns)
    connected = False

    def write_data(fd, data):
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

                await write_data(message.file_descriptor, str.encode(message.message))

            if not connected:
                connected = True
                on_connect()

            if batch.HasField("exit_code"):
                exit_status = batch.exit_code
                break

            last_batch_index = batch.batch_index

    while exit_status is None:
        try:
            if not connected:
                try:
                    output_task = asyncio.create_task(_get_output())
                    await asyncio.wait_for(asyncio.shield(output_task), timeout=FIRST_OUTPUT_TIMEOUT)
                except (asyncio.TimeoutError, TimeoutError):
                    if not connected:
                        output_task.cancel()
                        raise TimeoutError()
                    else:
                        await output_task
            else:
                await _get_output()
        except (GRPCError, StreamTerminatedError) as exc:
            if isinstance(exc, GRPCError):
                if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                    continue
            elif isinstance(exc, StreamTerminatedError):
                continue
            raise

    return exit_status
