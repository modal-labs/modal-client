# Copyright Modal Labs 2023
import json
from typing import Optional

from typing_extensions import Annotated

import typer

from modal.client import _Client
from modal.config import config
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer
from google.protobuf.empty_pb2 import Empty

ENVIRONMENT_HELP_TEXT = """Create and interact with Environments

Environments are sub-divisons of workspaces, allowing you to deploy the same app
in different namespaces. Each environment has their own set of Secrets and any
lookups performed from an app in an environment will by default look for entities
in the same environment.

Typical use cases for environments include having one for development and one for
production, to prevent overwriting production apps when developing new features
while still being able to deploy changes to a live environment.
"""

ENV_OPTION_HELP = """Environment to interact with

If none is specified, Modal will use the default environment of your current profile (can also be specified via the environment variable MODAL_ENVIRONMENT).
If neither is set, Modal will assume there is only one environment in the active workspace and use that one, or raise an error if there are multiple environments.
"""


environment_cli = typer.Typer(name="environment", help=ENVIRONMENT_HELP_TEXT, no_args_is_help=True)


def ensure_env(environment_name: Optional[str] = None) -> str:
    """Override config environment with environment from environment_name

    This is necessary since a cli command that runs Modal code, e.g. `modal.lookup()` without
    explicit environment specification wouldn't pick up the environment specified in a command
    line flag otherwise, e.g. when doing `modal run --env=foo`
    """
    if environment_name is not None:
        config.override_locally("environment", environment_name)

    return config.get("environment")


def display_results(items, fields):
    for item in items:
        typer.echo(json.dumps({field_name: getattr(item, field_name) for field_name in fields}))


@environment_cli.command(name="list", help="List all environments in the current workspace")
@synchronizer.create_blocking
async def list():
    client = await _Client.from_env()
    stub = client.stub
    resp = await stub.EnvironmentList(Empty())
    display_results(resp.items, ["name"])


ENVIRONMENT_CREATE_HELP = """Create a new environment in the current workspace"""


@environment_cli.command(name="create", help=ENVIRONMENT_CREATE_HELP)
@synchronizer.create_blocking
async def create(
    name: Annotated[str, typer.Argument(help="Name of the new environment. Must be unique. Case sensitive")]
):
    client = await _Client.from_env()
    stub = client.stub
    await stub.EnvironmentCreate(api_pb2.EnvironmentCreateRequest(name=name))
    typer.echo(f"Environment created: {name}")


ENVIRONMENT_DELETE_HELP = """Delete an environment in the current workspace

Deletes all apps in the selected environment and deletes the environment irrevocably.
"""


@environment_cli.command(name="delete", help=ENVIRONMENT_DELETE_HELP)
@synchronizer.create_blocking
async def delete(
    name: str = typer.Argument(help="Name of the environment to be deleted. Case sensitive"),
    confirm: bool = typer.Option(default=False, help="Set this flag to delete without prompting for confirmation"),
):
    client = await _Client.from_env()
    stub = client.stub
    if not confirm:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the environment '{name}' and all its associated apps and secrets?",
            default=False,
            abort=True,
        )

    await stub.EnvironmentDelete(api_pb2.EnvironmentDeleteRequest(name=name))
    typer.echo(f"Environment deleted: {name}")
