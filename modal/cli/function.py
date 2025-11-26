from typing import Optional

import typer

from modal._object import _get_environment_name
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal.environments import ensure_env
from modal_proto import api_pb2

from .utils import ENV_OPTION

function_cli = typer.Typer(name="function", help="Manage functions.", no_args_is_help=True)


@function_cli.command("drain-old", help="Drain containers that are not from the latest deployment.")
@synchronizer.create_blocking
async def drain_old(app_name: str, name: str, env: Optional[str] = ENV_OPTION):
    env = ensure_env(env)
    client = await _Client.from_env()

    resp: api_pb2.FunctionDrainOldContainersResponse = await client.stub.FunctionDrainOldContainers(
        api_pb2.FunctionDrainOldContainersRequest(
            app_name=app_name, name=name, environment_name=_get_environment_name(env)
        )
    )
    print(resp.drained_containers)
