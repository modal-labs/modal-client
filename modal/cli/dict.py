# Copyright Modal Labs 2024
import builtins
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON
from typer import Argument, Option, Typer

from modal._resolver import Resolver
from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal.cli.utils import ENV_OPTION, YES_OPTION, display_table
from modal.client import _Client
from modal.dict import _Dict
from modal.environments import ensure_env
from modal_proto import api_pb2

dict_cli = Typer(
    name="dict",
    no_args_is_help=True,
    help="Manage `modal.Dict` objects and inspect their contents.",
)


@dict_cli.command(name="create")
@synchronizer.create_blocking
async def create(name: str, *, env: Optional[str] = ENV_OPTION):
    """Create a named Dict object.

    Note: This is a no-op when the Dict already exists.
    """
    d = _Dict.from_name(name, environment_name=env, create_if_missing=True)
    client = await _Client.from_env()
    resolver = Resolver(client=client)
    await resolver.load(d)


@dict_cli.command(name="list")
@synchronizer.create_blocking
async def list(*, json: bool = False, env: Optional[str] = ENV_OPTION):
    """List all named Dict objects."""
    env = ensure_env(env)
    client = await _Client.from_env()
    request = api_pb2.DictListRequest(environment_name=env)
    response = await retry_transient_errors(client.stub.DictList, request)

    def format_timestamp(t: float) -> str:
        return datetime.strftime(datetime.fromtimestamp(t), "%Y-%m-%d %H:%M") + " UTC"

    rows = [(d.name, format_timestamp(d.created_at)) for d in response.dicts]
    display_table(["Name", "Created at"], rows, json)


@dict_cli.command("clear")
@synchronizer.create_blocking
async def clear(name: str, *, env: Optional[str] = ENV_OPTION):
    """Clear the contents of a named Dict by deleting all of its data."""
    d = await _Dict.lookup(name, environment_name=env)
    await d.clear()


@dict_cli.command(name="delete")
@synchronizer.create_blocking
async def delete(name: str, *, yes: bool = YES_OPTION, env: Optional[str] = ENV_OPTION):
    """Delete a named Dict object and all of its data."""
    if not yes:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the modal.Dict '{name}'?",
            default=False,
            abort=True,
        )
    await _Dict.delete(name, environment_name=env)


@dict_cli.command(name="get")
@synchronizer.create_blocking
async def get(name: str, key: str, *, env: Optional[str] = ENV_OPTION):
    """Print the value for a specific key.

    Note: When using the CLI, keys are always interpreted as having a string type.
    """
    d = await _Dict.lookup(name, environment_name=env)
    console = Console()
    val = await d.get(key)
    console.print(val)


@dict_cli.command(name="show")
@synchronizer.create_blocking
async def show(
    name: str,
    n: int = Argument(default=20, help="Limit the number of entries shown"),
    *,
    all: bool = Option(default=False, help="Ignore N and print all entries in the Dict (may be slow)"),
    repr: bool = Option(default=False, help="Display items using repr() to see more details"),
    json: bool = False,
    env: Optional[str] = ENV_OPTION,
):
    """Print the contents of a Dict.

    Note: By default, this command truncates the contents. Use the `N` argument to control the
    amount of data shown or the `--all` option to retrieve the entire Dict, which may be slow.
    """
    d = await _Dict.lookup(name, environment_name=env)

    i, items = 0, []
    async for key, val in d.items():
        i += 1
        if not all and i > n:
            items.append(("...", "..."))
            break
        else:
            display_item = (builtins.repr(key), builtins.repr(val)) if repr else (str(key), str(val))
            items.append(display_item)

    if json:
        # Note, we don't use the json= option of display_table because we want to display
        # the dict itself as a JSON, rather than have a JSON representation of the table.
        console = Console()
        console.print(JSON.from_data(dict(items)))
    else:
        display_table(["Key", "Value"], [[str(k), str(v)] for k, v in items])
