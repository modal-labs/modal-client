# Copyright Modal Labs 2022
import sys
from datetime import datetime
from typing import Optional

from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.table import Table
from typer import Argument, Typer

from modal.cli.utils import ENV_OPTION, display_table
from modal.client import _Client
from modal.environments import ensure_env
from modal.volume import _Volume, _VolumeHandle
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer
from modal_utils.grpc_utils import retry_transient_errors

FileType = api_pb2.VolumeListFilesEntry.FileType

vol_cli = Typer(name="vol", help="[Preview] Read and edit modal.Volume volumes.", no_args_is_help=True)


@vol_cli.command(name="list", help="List the names of all modal.Volume volumes.", hidden=True)
@synchronizer.create_blocking
async def list(env: Optional[str] = ENV_OPTION, json: Optional[bool] = False):
    env = ensure_env(env)
    client = await _Client.from_env()
    response = await retry_transient_errors(client.stub.VolumeList, api_pb2.VolumeListRequest(environment_name=env))
    env_part = f" in environment '{env}'" if env else ""
    column_names = ["Name", "Created at"]
    rows = []
    locale_tz = datetime.now().astimezone().tzinfo
    for item in response.items:
        rows.append(
            [
                item.label,
                str(datetime.fromtimestamp(item.created_at, tz=locale_tz)),
            ]
        )
    display_table(column_names, rows, json, title=f"Volumes{env_part}")


@vol_cli.command(name="ls", help="List files and directories in a modal.Volume volume.")
@synchronizer.create_blocking
async def ls(
    volume_name: str,
    path: str = Argument(default="/"),
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    vol: _VolumeHandle = await _Volume.lookup(volume_name, environment_name=env)
    if not isinstance(vol, _VolumeHandle):
        raise UsageError("The specified app entity is not a modal.Volume")

    try:
        entries = await vol.listdir(path)
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message)
        raise

    if sys.stdout.isatty():
        console = Console()
        console.print(f"Directory listing of '{path}' in '{volume_name}'")
        table = Table()
        for name in ["filename", "type", "created/modified"]:
            table.add_column(name)

        locale_tz = datetime.now().astimezone().tzinfo
        for entry in entries:
            filetype = "dir" if entry.type == FileType.DIRECTORY else "file"
            table.add_row(
                entry.path,
                filetype,
                str(datetime.fromtimestamp(entry.mtime, tz=locale_tz)),
            )
        console.print(table)
    else:
        for entry in entries:
            print(entry.path)
