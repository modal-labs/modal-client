# Copyright Modal Labs 2022

import asyncio
import contextlib
import os
import select
import sys
from typing import List, Optional, Union

import typer
from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.text import Text

from modal._pty import get_pty_info, raw_terminal, set_nonblocking
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
    client = await _Client.from_env()
    res: api_pb2.ContainerExecResponse = await client.stub.ContainerExec(
        api_pb2.ContainerExecRequest(task_id=task_id, command=command, pty_info=get_pty_info(shell=True))
    )
    if res.exec_id is None:
        # todo(nathan): proper error message?
        print("failed to execute command, unclear why")
        return

    async with handle_exec_input(client, res.exec_id):
        await handle_exec_output(client, res.exec_id)


# todo(nathan): duplicated code in _pty.py
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
    with raw_terminal():
        yield
    os.write(quit_pipe_write, b"\n")
    write_task.cancel()


async def handle_exec_output(client: _Client, exec_id: str):
    """Streams exec output to stdout."""
    last_entry_id = ""
    completed = False

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

                # todo(nathan): deal with resource temporarily unavailable error when there's too much output
                os.write(message.file_descriptor, str.encode(message.message))

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


@container_cli.command("connect")
@synchronizer.create_blocking
async def connect(task_id: str):
    """Connects to a container and spawns /bin/bash."""
    # todo(nathan): copy code from above
