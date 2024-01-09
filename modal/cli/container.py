# Copyright Modal Labs 2022

from typing import List, Union

import typer
from rich.text import Text

from modal.cli.utils import display_table, timestamp_to_local
from modal.client import _Client
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer

container_cli = typer.Typer(name="container", help="Manage running containers.", no_args_is_help=True)


@container_cli.command("list")
@synchronizer.create_blocking
async def list():
    """List all containers that are currently running"""
    client = await _Client.from_env()
    res: api_pb2.ContainerListResponse = await client.stub.ContainerList(api_pb2.ContainerListRequest())

    column_names = ["Container ID", "App ID", "App Name", "Start time"]
    rows: List[List[Union[Text, str]]] = []
    for container_stats in res.containers:
        rows.append(
            [
                container_stats.container_id,
                container_stats.app_id,
                container_stats.app_description,
                timestamp_to_local(container_stats.started_at) if container_stats.started_at else "Pending",
            ]
        )

    display_table(column_names, rows, json=False, title="Active Containers")
