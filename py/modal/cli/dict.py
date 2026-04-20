# Copyright Modal Labs 2024

from typing import Optional

import click

from modal._load_context import LoadContext
from modal._resolver import Resolver
from modal._utils.async_utils import TaskContext, synchronizer
from modal._utils.time_utils import timestamp_to_localized_str
from modal.cli.utils import display_table, env_option, yes_option
from modal.client import _Client
from modal.dict import _Dict
from modal.environments import ensure_env
from modal.output import OutputManager

from ._help import ModalGroup

dict_cli = ModalGroup(
    name="dict",
    help="Manage `modal.Dict` objects and inspect their contents.",
)


@dict_cli.command("create", panel="Management")
@click.argument("name")
@env_option
@synchronizer.create_blocking
async def create(name: str, *, env: Optional[str] = None):
    """Create a named Dict object.

    Note: This is a no-op when the Dict already exists.
    """
    d = _Dict.from_name(name, environment_name=env, create_if_missing=True)
    client = await _Client.from_env()
    resolver = Resolver()

    async with TaskContext() as tc:
        load_context = LoadContext(client=client, environment_name=env, task_context=tc)
        await resolver.load(d, load_context)


@dict_cli.command("list", panel="Management")
@click.option("--json", is_flag=True, default=False)
@env_option
@synchronizer.create_blocking
async def list_(*, json: bool = False, env: Optional[str] = None):
    """List all named Dicts."""
    env = ensure_env(env)
    dicts = await _Dict.objects.list(environment_name=env)
    rows = []
    for obj in dicts:
        info = await obj.info()
        rows.append((info.name, timestamp_to_localized_str(info.created_at.timestamp(), json), info.created_by))

    display_table(["Name", "Created at", "Created by"], rows, json)


@dict_cli.command("clear", panel="Management")
@click.argument("name")
@yes_option
@env_option
@synchronizer.create_blocking
async def clear(name: str, *, yes: bool = False, env: Optional[str] = None):
    """Clear the contents of a named Dict by deleting all of its data."""
    d = _Dict.from_name(name, environment_name=env)
    if not yes:
        click.confirm(
            f"Are you sure you want to irrevocably delete the contents of modal.Dict '{name}'?",
            default=False,
            abort=True,
        )
    await d.clear()


@dict_cli.command("delete", panel="Management")
@click.argument("name")
@click.option("--allow-missing", is_flag=True, default=False, help="Don't error if the Dict doesn't exist.")
@yes_option
@env_option
@synchronizer.create_blocking
async def delete(
    name: str,
    *,
    allow_missing: bool = False,
    yes: bool = False,
    env: Optional[str] = None,
):
    """Delete a named Dict and all of its data."""
    if not yes:
        click.confirm(
            f"Are you sure you want to irrevocably delete the modal.Dict '{name}'?",
            default=False,
            abort=True,
        )
    await _Dict.objects.delete(name, environment_name=env, allow_missing=allow_missing)


@dict_cli.command("get", panel="Inspection")
@click.argument("name")
@click.argument("key")
@env_option
@synchronizer.create_blocking
async def get(name: str, key: str, *, env: Optional[str] = None):
    """Print the value for a specific key.

    Note: When using the CLI, keys are always interpreted as having a string type.
    """
    d = _Dict.from_name(name, environment_name=env)
    val = await d.get(key)
    OutputManager.get().print(val)


def _display(input: str, use_repr: bool) -> str:
    val = repr(input) if use_repr else str(input)
    return val[:80] + "..." if len(val) > 80 else val


@dict_cli.command("items", panel="Inspection")
@click.argument("name")
@click.argument("n", default=20, type=int)
@click.option(
    "-a",
    "--all",
    "all",
    is_flag=True,
    default=False,
    help="Ignore N and print all entries in the Dict (may be slow)",
)
@click.option(
    "-r",
    "--repr",
    "use_repr",
    is_flag=True,
    default=False,
    help="Display items using `repr()` to see more details",
)
@click.option("--json", is_flag=True, default=False)
@env_option
@synchronizer.create_blocking
async def items(
    name: str,
    n: int = 20,
    *,
    all: bool = False,
    use_repr: bool = False,
    json: bool = False,
    env: Optional[str] = None,
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
