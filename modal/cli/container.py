# Copyright Modal Labs 2022

from typing import List, Union

import typer
from rich.text import Text

from modal.cli.utils import display_table
from modal.utils import container_exec, list_containers, stop_container, stream_container_logs

container_cli = typer.Typer(name="container", help="Manage and connect to running containers.", no_args_is_help=True)


@container_cli.command("list")
def list(json: bool = False):
    """List all containers that are currently running."""
    containers = list_containers()
    column_names = ["Container ID", "App ID", "App Name", "Start Time"]
    rows: List[List[Union[Text, str]]] = []

    for container_stats in containers:
        rows.append(
            [
                container_stats.task_id,
                container_stats.app_id,
                container_stats.app_description,
                str(container_stats.started_at) if str(container_stats.started_at) else "Pending",
            ]
        )

    display_table(column_names, rows, json=json, title="Active Containers")


@container_cli.command("logs")
def logs(container_id: str = typer.Argument(help="Container ID")):
    """Show logs for a specific container, streaming while active."""
    stream_container_logs(task_id=container_id)


@container_cli.command("exec")
def exec(
    container_id: str = typer.Argument(help="Container ID"),
    command: List[str] = typer.Argument(help="A command to run inside the container."),
    pty: bool = typer.Option(is_flag=True, default=True, help="Run the command using a PTY."),
):
    """Execute a command in a container."""
    container_exec(container_id, command, pty)


@container_cli.command("stop")
def stop(container_id: str = typer.Argument(help="Container ID")):
    """Stop a currently-running container and reassign its in-progress inputs.

    This will send the container a SIGINT signal that Modal will handle.
    """
    stop_container(container_id)
