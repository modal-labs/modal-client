# Copyright Modal Labs 2024
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.json import JSON
from typer import Argument, Typer

from modal._resolver import Resolver
from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal.cli.utils import ENV_OPTION, display_table
from modal.client import _Client
from modal.dict import _Dict
from modal.environments import ensure_env
from modal.exception import ExecutionError
from modal_proto import api_pb2

dict_cli = Typer(
    name="dict",
    no_args_is_help=True,
    help="Manage `modal.Dict` objects and inspect their contents.",
)


@dict_cli.command(name="create")
@synchronizer.create_blocking
async def create(name: str, *, env: Optional[str] = ENV_OPTION):
    """Create a named Dict object."""
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
    """Clear the contents Dict by deleting all of its data."""
    d = await _Dict.lookup(name, environment_name=env)
    await d.clear()


@dict_cli.command(name="delete")
@synchronizer.create_blocking
async def delete(name: str, *, env: Optional[str] = ENV_OPTION):
    """Delete a named Dict object and all of its data."""
    client = await _Client.from_env()
    lookup_request = api_pb2.AppGetByDeploymentNameRequest(
        name=name,
        environment_name=ensure_env(env),
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    )
    resp = await client.stub.AppGetByDeploymentName(lookup_request)
    stop_req = api_pb2.AppStopRequest(app_id=resp.app_id, source=api_pb2.APP_STOP_SOURCE_CLI)
    await client.stub.AppStop(stop_req)


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
    n: int = Argument(default=None, help="Retrieve and show no more than this many entries"),
    *,
    json: bool = False,
    env: Optional[str] = ENV_OPTION,
):
    """Print the contents of a Dict.

    Note: By default, this command will retrieve the complete Dict contents.
    Using the N argument is recommended for quickly inspecting a small number of items.
    """
    d = await _Dict.lookup(name, environment_name=env)

    try:
        i, items = 0, []
        async for item in d.items():
            i += 1
            items.append(item)
            if n is not None and i >= n:
                break
    except ModuleNotFoundError as exc:
        # I think that on 3.10+ we could rewrite this to use anext and attribute errors to specific
        # items (perhaps represent them in the output as "<library>_object" or something.)
        msg = f"Dict contains objects from the `{exc.name}` library and cannot be deserialized locally."
        raise ExecutionError(msg)

    if json:
        # Note, we don't use the json= option of display_table because we want to display
        # the dict itself as a JSON, rather than have a JSON representation of the table.
        console = Console()
        console.print(JSON.from_data(dict(items)))
    else:
        display_table(["Key", "Value"], [[str(k), str(v)] for k, v in items])
