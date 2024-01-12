# Copyright Modal Labs 2022

import asyncio
import os
import select
import sys
from typing import List, Union

import typer
from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.text import Text

from modal._pty import raw_terminal, set_nonblocking
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
        api_pb2.ContainerExecRequest(task_id=task_id, command=command)
    )
    if res.exec_id is None:
        # todo(nathan): proper error message?
        print("failed to execute command, unclear why")
        return

    # todo(nathan): feels like a context manager might be more appropriate?
    stream_stdin_task = asyncio.create_task(handle_exec_input(client, res.exec_id))
    try:
        await handle_exec_output(client, res.exec_id)
    except Exception as e:
        stream_stdin_task.cancel()
        raise e


async def handle_exec_input(client: _Client, exec_id: str):
    """Streams CLI stdin to exec stdin."""
    try:
        set_nonblocking(sys.stdin.fileno())

        @asyncify
        def _read_stdin() -> bytes:
            (readable, _, _) = select.select([sys.stdin.buffer], [], [])
            return sys.stdin.buffer.read()

        with raw_terminal():
            while True:
                data = await _read_stdin()

                # ctrl-c. todo(nathan): better way to detect this?
                if 0x03 in data:
                    break

                await client.stub.ContainerExecPutInput(
                    api_pb2.ContainerExecPutInputRequest(
                        exec_id=exec_id, input=api_pb2.RuntimeInputMessage(message=data)
                    )
                )
    except Exception as e:
        # todo(nathan): figure out a better way to deal with errors here
        print("Error: failed to read input", e)
        raise e


async def handle_exec_output(client: _Client, exec_id: str):
    """Streams exec output to stdout."""
    last_entry_id = ""
    completed = False

    async def _get_output():
        nonlocal last_entry_id, completed

        req = api_pb2.ContainerExecGetOutputRequest(
            exec_id=exec_id,
            timeout=0.5,
            last_entry_id=last_entry_id,
        )
        async for message in unary_stream(client.stub.ContainerExecGetOutput, req):
            if message.eof:
                completed = True
                break

            assert message.file_descriptor in [1, 2]

            os.write(message.file_descriptor, str.encode(message.message))

            if message.entry_id:
                last_entry_id = message.entry_id

    while not completed:
        try:
            await _get_output()
        except (GRPCError, StreamTerminatedError) as exc:
            # todo(nathan): consider debounce?
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
