# Copyright Modal Labs 2022

from typing import List, Union

import typer
from rich.text import Text

from modal._container_exec import container_exec
from modal._utils.async_utils import synchronizer
from modal.cli.utils import display_table, timestamp_to_local
from modal.client import _Client
from modal_proto import api_pb2

container_cli = typer.Typer(name="container", help="Manage running containers.", no_args_is_help=True)


@container_cli.command("list")
@synchronizer.create_blocking
async def list():
    """List all containers that are currently running."""
    client = await _Client.from_env()
    res: api_pb2.TaskListResponse = await client.stub.TaskList(api_pb2.TaskListRequest())

    column_names = ["Container ID", "App ID", "App Name", "Start Time"]
    rows: List[List[Union[Text, str]]] = []
    res.tasks.sort(key=lambda task: task.started_at, reverse=True)
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
    container_id: str = typer.Argument(help="Container ID."),
    command: List[str] = typer.Argument(help="A command to run inside the container."),
    pty: bool = typer.Option(is_flag=True, default=True, help="Run the command using a PTY."),
):
    """Execute a command in a container."""
    client = await _Client.from_env()
    await container_exec(container_id, command, pty=pty, client=client)
