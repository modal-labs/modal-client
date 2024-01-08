# Copyright Modal Labs 2022

import typer

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

    print("Hello World!")
    print(res)
