# Copyright Modal Labs 2024
from typing import Optional

import typer
from rich.console import Console
from typer import Argument, Option, Typer

from modal._resolver import Resolver
from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal._utils.time_utils import timestamp_to_local
from modal.cli.utils import ENV_OPTION, YES_OPTION, display_table
from modal.client import _Client
from modal.environments import ensure_env
from modal.queue import _Queue
from modal_proto import api_pb2

queue_cli = Typer(
    name="queue",
    no_args_is_help=True,
    help="Manage `modal.Queue` objects and inspect their contents.",
)

PARTITION_OPTION = Option(
    None,
    "-p",
    "--partition",
    help="Name of the partition to use, otherwise use the default (anonymous) partition.",
)


@queue_cli.command(name="create", rich_help_panel="Management")
@synchronizer.create_blocking
async def create(name: str, *, env: Optional[str] = ENV_OPTION):
    """Create a named Queue.

    Note: This is a no-op when the Queue already exists.
    """
    q = _Queue.from_name(name, environment_name=env, create_if_missing=True)
    client = await _Client.from_env()
    resolver = Resolver(client=client)
    await resolver.load(q)


@queue_cli.command(name="delete", rich_help_panel="Management")
@synchronizer.create_blocking
async def delete(name: str, *, yes: bool = YES_OPTION, env: Optional[str] = ENV_OPTION):
    """Delete a named Queue and all of its data."""
    # Lookup first to validate the name, even though delete is a staticmethod
    await _Queue.from_name(name, environment_name=env).hydrate()
    if not yes:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the modal.Queue '{name}'?",
            default=False,
            abort=True,
        )
    await _Queue.delete(name, environment_name=env)


@queue_cli.command(name="list", rich_help_panel="Management")
@synchronizer.create_blocking
async def list_(*, json: bool = False, env: Optional[str] = ENV_OPTION):
    """List all named Queues."""
    env = ensure_env(env)

    max_total_size = 100_000
    client = await _Client.from_env()
    request = api_pb2.QueueListRequest(environment_name=env, total_size_limit=max_total_size + 1)
    response = await retry_transient_errors(client.stub.QueueList, request)

    rows = [
        (
            q.name,
            timestamp_to_local(q.created_at, json),
            str(q.num_partitions),
            str(q.total_size) if q.total_size <= max_total_size else f">{max_total_size}",
        )
        for q in response.queues
    ]
    display_table(["Name", "Created at", "Partitions", "Total size"], rows, json)


@queue_cli.command(name="clear", rich_help_panel="Management")
@synchronizer.create_blocking
async def clear(
    name: str,
    partition: Optional[str] = PARTITION_OPTION,
    all: bool = Option(False, "-a", "--all", help="Clear the contents of all partitions."),
    yes: bool = YES_OPTION,
    *,
    env: Optional[str] = ENV_OPTION,
):
    """Clear the contents of a queue by removing all of its data."""
    q = _Queue.from_name(name, environment_name=env)
    if not yes:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the contents of modal.Queue '{name}'?",
            default=False,
            abort=True,
        )
    await q.clear(partition=partition, all=all)


@queue_cli.command(name="peek", rich_help_panel="Inspection")
@synchronizer.create_blocking
async def peek(
    name: str, n: int = Argument(1), partition: Optional[str] = PARTITION_OPTION, *, env: Optional[str] = ENV_OPTION
):
    """Print the next N items in the queue or queue partition (without removal)."""
    q = _Queue.from_name(name, environment_name=env)
    console = Console()
    i = 0
    async for item in q.iterate(partition=partition):
        console.print(item)
        i += 1
        if i >= n:
            break


@queue_cli.command(name="len", rich_help_panel="Inspection")
@synchronizer.create_blocking
async def len(
    name: str,
    partition: Optional[str] = PARTITION_OPTION,
    total: bool = Option(False, "-t", "--total", help="Compute the sum of the queue lengths across all partitions"),
    *,
    env: Optional[str] = ENV_OPTION,
):
    """Print the length of a queue partition or the total length of all partitions."""
    q = _Queue.from_name(name, environment_name=env)
    console = Console()
    console.print(await q.len(partition=partition, total=total))
