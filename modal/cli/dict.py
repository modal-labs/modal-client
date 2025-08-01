# Copyright Modal Labs 2024
from typing import Optional

import typer
from typer import Argument, Option, Typer

from modal._output import make_console
from modal._resolver import Resolver
from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal._utils.time_utils import timestamp_to_localized_str
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


@dict_cli.command(name="create", rich_help_panel="Management")
@synchronizer.create_blocking
async def create(name: str, *, env: Optional[str] = ENV_OPTION):
    """Create a named Dict object.

    Note: This is a no-op when the Dict already exists.
    """
    d = _Dict.from_name(name, environment_name=env, create_if_missing=True)
    client = await _Client.from_env()
    resolver = Resolver(client=client)
    await resolver.load(d)


@dict_cli.command(name="list", rich_help_panel="Management")
@synchronizer.create_blocking
async def list_(*, json: bool = False, env: Optional[str] = ENV_OPTION):
    """List all named Dicts."""
    env = ensure_env(env)
    client = await _Client.from_env()
    request = api_pb2.DictListRequest(environment_name=env)
    response = await retry_transient_errors(client.stub.DictList, request)

    rows = [(d.name, timestamp_to_localized_str(d.created_at, json)) for d in response.dicts]
    display_table(["Name", "Created at"], rows, json)


@dict_cli.command("clear", rich_help_panel="Management")
@synchronizer.create_blocking
async def clear(name: str, *, yes: bool = YES_OPTION, env: Optional[str] = ENV_OPTION):
    """Clear the contents of a named Dict by deleting all of its data."""
    d = _Dict.from_name(name, environment_name=env)
    if not yes:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the contents of modal.Dict '{name}'?",
            default=False,
            abort=True,
        )
    await d.clear()


@dict_cli.command(name="delete", rich_help_panel="Management")
@synchronizer.create_blocking
async def delete(name: str, *, yes: bool = YES_OPTION, env: Optional[str] = ENV_OPTION):
    """Delete a named Dict and all of its data."""
    # Lookup first to validate the name, even though delete is a staticmethod
    await _Dict.from_name(name, environment_name=env).hydrate()
    if not yes:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the modal.Dict '{name}'?",
            default=False,
            abort=True,
        )
    await _Dict.delete(name, environment_name=env)


@dict_cli.command(name="get", rich_help_panel="Inspection")
@synchronizer.create_blocking
async def get(name: str, key: str, *, env: Optional[str] = ENV_OPTION):
    """Print the value for a specific key.

    Note: When using the CLI, keys are always interpreted as having a string type.
    """
    d = _Dict.from_name(name, environment_name=env)
    console = make_console()
    val = await d.get(key)
    console.print(val)


def _display(input: str, use_repr: bool) -> str:
    val = repr(input) if use_repr else str(input)
    return val[:80] + "..." if len(val) > 80 else val


@dict_cli.command(name="items", rich_help_panel="Inspection")
@synchronizer.create_blocking
async def items(
    name: str,
    n: int = Argument(default=20, help="Limit the number of entries shown"),
    *,
    all: bool = Option(False, "-a", "--all", help="Ignore N and print all entries in the Dict (may be slow)"),
    use_repr: bool = Option(False, "-r", "--repr", help="Display items using `repr()` to see more details"),
    json: bool = False,
    env: Optional[str] = ENV_OPTION,
):
    """Print the contents of a Dict.

    Note: By default, this command truncates the contents. Use the `N` argument to control the
    amount of data shown or the `--all` option to retrieve the entire Dict, which may be slow.
    """
    d = _Dict.from_name(name, environment_name=env)

    i, items = 0, []
    async for key, val in d.items():
        i += 1
        if not json and not all and i > n:
            items.append(("...", "..."))
            break
        else:
            if json:
                display_item = key, val
            else:
                display_item = _display(key, use_repr), _display(val, use_repr)  # type: ignore  # mypy/issue/12056
            items.append(display_item)

    display_table(["Key", "Value"], items, json)
