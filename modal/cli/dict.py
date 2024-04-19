# Copyright Modal Labs 2024
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.json import JSON
from typer import Argument, Typer

from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal.cli.utils import ENV_OPTION, display_table
from modal.client import _Client
from modal.dict import _Dict
from modal.environments import ensure_env
from modal_proto import api_pb2

dict_cli = Typer(
    name="dict",
    no_args_is_help=True,
    help="Inspect the contents of modal.Dict objects",
)


@dict_cli.command(name="list")
@synchronizer.create_blocking
async def list(*, json: bool = False, env: Optional[str] = ENV_OPTION):
    """List all named Dict objects."""
    env = ensure_env(env)
    client = await _Client.from_env()
    request = api_pb2.DictListRequest(environment_name=env)
    response = await retry_transient_errors(client.stub.DictList, request)

    def format_timestamp(t: float) -> str:
        return datetime.strftime(datetime.fromtimestamp(t), "%Y-%m-%d %H:%M")

    rows = [(d.name, format_timestamp(d.created_at)) for d in response.dicts]
    display_table(["Name", "Created at"], rows, json)


@dict_cli.command(name="show")
@synchronizer.create_blocking
async def show(
    name: str,
    n: int = Argument(default=None, help="Retrieve and show no more than this many entries"),
    *,
    json: bool = False,
    env: Optional[str] = ENV_OPTION,
):
    """Print the contents of a Dict."""
    # TODO alternate names: inspect, dump, items
    # TODO add an n: int option? or alternately a `modal dict peek` command or similar
    d = await _Dict.lookup(name, environment_name=env)
    i, items = 0, []
    async for item in d.items():
        i += 1
        items.append(item)
        if n is not None and i >= n:
            break
    if json:
        # Note, we don't use the json= option of display_table becuase we want to display
        # the dict itself as a JSON, rather than have a JSON representation of the table.
        console = Console()
        console.print(JSON.from_data(dict(items)))
    else:
        display_table(["Key", "Value"], [[str(k), str(v)] for k, v in items])


@dict_cli.command(name="get")
@synchronizer.create_blocking
async def get(name: str, key: str, *, env: Optional[str] = ENV_OPTION):
    """Print the value for a specific key.

    Note: When using the CLI, keys are always interpreted as having a string type.
    """
    # TODO would it be nice to be able to get multiple values? Should we do that here?
    d = await _Dict.lookup(name, environment_name=env)
    console = Console()
    val = await d.get(key)
    # TODO val will be `None` when key is not found
    console.print(val)


@dict_cli.command("clear")
@synchronizer.create_blocking
async def clear(name: str, *, env: Optional[str] = ENV_OPTION):
    """Clear the contents Dict by deleting all of its data."""
    d = await _Dict.lookup(name, environment_name=env)
    await d.clear()


@dict_cli.command(name="delete")
@synchronizer.create_blocking
async def delete(name: str, *, env: Optional[str] = ENV_OPTION):
    """Delete a named Dict object and all of its data."""
    d = await _Dict.lookup(name, environment_name=env)
    client = await _Client.from_env()
    req = api_pb2.AppStopRequest(app_id=d.object_id, source=api_pb2.APP_STOP_SOURCE_CLI)
    await client.stub.AppStop(req)
