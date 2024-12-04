# Copyright Modal Labs 2022

from typing import Optional, Union

import typer
from rich.text import Text

from modal._pty import get_pty_info
from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal.cli.utils import ENV_OPTION, display_table, stream_app_logs, timestamp_to_local
from modal.client import _Client
from modal.config import config
from modal.container_process import _ContainerProcess
from modal.environments import ensure_env
from modal.object import _get_environment_name
from modal_proto import api_pb2

container_cli = typer.Typer(name="container", help="Manage and connect to running containers.", no_args_is_help=True)


@container_cli.command("list")
@synchronizer.create_blocking
async def list_(env: Optional[str] = ENV_OPTION, json: bool = False):
    """List all containers that are currently running."""
    env = ensure_env(env)
    client = await _Client.from_env()
    environment_name = _get_environment_name(env)
    res: api_pb2.TaskListResponse = await client.stub.TaskList(
        api_pb2.TaskListRequest(environment_name=environment_name)
    )

    column_names = ["Container ID", "App ID", "App Name", "Start Time"]
    rows: list[list[Union[Text, str]]] = []
    res.tasks.sort(key=lambda task: task.started_at, reverse=True)
    for task_stats in res.tasks:
        rows.append(
            [
                task_stats.task_id,
                task_stats.app_id,
                task_stats.app_description,
                timestamp_to_local(task_stats.started_at, json) if task_stats.started_at else "Pending",
            ]
        )

    display_table(column_names, rows, json=json, title=f"Active Containers in environment: {environment_name}")


@container_cli.command("logs")
def logs(container_id: str = typer.Argument(help="Container ID")):
    """Show logs for a specific container, streaming while active."""
    stream_app_logs(task_id=container_id)


@container_cli.command("exec")
@synchronizer.create_blocking
async def exec(
    container_id: str = typer.Argument(help="Container ID"),
    command: list[str] = typer.Argument(help="A command to run inside the container."),
    pty: bool = typer.Option(default=True, help="Run the command using a PTY."),
):
    """Execute a command in a container."""

    client = await _Client.from_env()

    req = api_pb2.ContainerExecRequest(
        task_id=container_id,
        command=command,
        pty_info=get_pty_info(shell=True) if pty else None,
        runtime_debug=config.get("function_runtime_debug"),
    )
    res: api_pb2.ContainerExecResponse = await client.stub.ContainerExec(req)

    await _ContainerProcess(res.exec_id, client).attach(pty=pty)


@container_cli.command("stop")
@synchronizer.create_blocking
async def stop(container_id: str = typer.Argument(help="Container ID")):
    """Stop a currently-running container and reassign its in-progress inputs.

    This will send the container a SIGINT signal that Modal will handle.
    """
    client = await _Client.from_env()
    request = api_pb2.ContainerStopRequest(task_id=container_id)
    await retry_transient_errors(client.stub.ContainerStop, request)
