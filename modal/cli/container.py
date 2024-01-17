# Copyright Modal Labs 2022

import asyncio
import contextlib
import errno
import os
import platform
import select
import sys
from typing import List, Optional, Union

import typer
from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.text import Text

from modal._pty import get_pty_info, set_nonblocking
from modal.cli.utils import display_table, timestamp_to_local
from modal.client import _Client
from modal_proto import api_pb2
from modal_utils.async_utils import asyncify, synchronizer
from modal_utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, unary_stream

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
async def exec(task_id: str, command: str):
    """Execute a command inside an active container"""
    if platform.system() == "Windows":
        print("container exec is not currently supported on Windows.")
        return

    client = await _Client.from_env()
    res: api_pb2.ContainerExecResponse = await client.stub.ContainerExec(
        api_pb2.ContainerExecRequest(task_id=task_id, command=command, pty_info=get_pty_info(shell=True))
    )
    if res.exec_id == "":
        print(f"Failed to execute command. Is the container ID ({task_id}) correct?")
        return

    async with handle_exec_input(client, res.exec_id):
        await handle_exec_output(client, res.exec_id)


# note: this is very similar to code in _pty.py.
@contextlib.asynccontextmanager
async def handle_exec_input(client: _Client, exec_id: str):
    quit_pipe_read, quit_pipe_write = os.pipe()

    set_nonblocking(sys.stdin.fileno())

    @asyncify
    def _read_stdin() -> Optional[bytes]:
        nonlocal quit_pipe_read
        # TODO: Windows support.
        (readable, _, _) = select.select([sys.stdin.buffer, quit_pipe_read], [], [])
        if quit_pipe_read in readable:
            return None
        return sys.stdin.buffer.read()

    async def _write():
        while True:
            data = await _read_stdin()
            if data is None:
                return

            await client.stub.ContainerExecPutInput(
                api_pb2.ContainerExecPutInputRequest(exec_id=exec_id, input=api_pb2.RuntimeInputMessage(message=data))
            )

    write_task = asyncio.create_task(_write())

    yield

    os.write(quit_pipe_write, b"\n")
    write_task.cancel()


async def handle_exec_output(client: _Client, exec_id: str):
    """Streams exec output to stdout."""
    last_entry_id = ""
    completed = False

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
        nonlocal last_entry_id, completed

        req = api_pb2.ContainerExecGetOutputRequest(
            exec_id=exec_id,
            timeout=55,
            last_entry_id=last_entry_id,
        )
        async for batch in unary_stream(client.stub.ContainerExecGetOutput, req):
            for message in batch.items:
                assert message.file_descriptor in [1, 2]

                await write_data(message.file_descriptor, str.encode(message.message))

            if batch.eof:
                completed = True
                break

            if batch.entry_id:
                last_entry_id = batch.entry_id

    while not completed:
        try:
            await _get_output()
        except (GRPCError, StreamTerminatedError) as exc:
            if isinstance(exc, GRPCError):
                if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                    continue
            elif isinstance(exc, StreamTerminatedError):
                continue
            raise
