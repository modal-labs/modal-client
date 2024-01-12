# Copyright Modal Labs 2022

from typing import List, Union

import typer
from grpclib.exceptions import GRPCError, StreamTerminatedError
from rich.text import Text

from modal.cli.utils import display_table, timestamp_to_local
from modal.client import _Client
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer
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

    last_entry_id = ""

    async def _get_output():
        nonlocal last_entry_id

        req = api_pb2.ContainerExecGetOutputRequest(
            exec_id=res.exec_id,
            timeout=55,
            last_entry_id=last_entry_id,
        )
        async for message in unary_stream(client.stub.ContainerExecGetOutput, req):
            print(f"[{message.file_descriptor}] {message.message}")
            # todo(nathan): ugh need to actually implement entry id's as well...
            # if log_batch.entry_id:
            #     # log_batch entry_id is empty for fd="server" messages from AppGetLogs
            #     last_entry_id = log_batch.entry_id

            # if log_batch.eof:
            #     completed = True
            #     break

    while True:
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
    # todo(nathan): get rid of repeated code from above
    client = await _Client.from_env()
    res: api_pb2.ContainerExecResponse = await client.stub.ContainerExec(
        api_pb2.ContainerExecRequest(task_id=task_id, command="/bin/bash")
    )
    if res.exec_id is None:
        # todo(nathan): proper error message?
        print("failed to execute command, unclear why")
        return

    last_entry_id = ""

    async def _get_output():
        nonlocal last_entry_id

        req = api_pb2.ContainerExecGetOutputRequest(
            exec_id=res.exec_id,
            timeout=55,
            last_entry_id=last_entry_id,
        )
        async for message in unary_stream(client.stub.ContainerExecGetOutput, req):
            print(f"[{message.file_descriptor}] {message.message}")
            # todo(nathan): ugh need to actually implement entry id's as well...
            # if log_batch.entry_id:
            #     # log_batch entry_id is empty for fd="server" messages from AppGetLogs
            #     last_entry_id = log_batch.entry_id

            # if log_batch.eof:
            #     completed = True
            #     break

    while True:
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
