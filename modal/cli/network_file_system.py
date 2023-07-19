# Copyright Modal Labs 2022
import asyncio
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.live import Live
from rich.syntax import Syntax
from rich.table import Table
from typer import Typer

import modal
from modal._location import display_location
from modal._output import step_completed, step_progress
from modal.cli.utils import ENV_OPTION, display_table
from modal.client import _Client
from modal.environments import ensure_env
from modal.network_file_system import _NetworkFileSystem, _NetworkFileSystemHandle
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer
from modal_utils.grpc_utils import retry_transient_errors

FileType = api_pb2.SharedVolumeListFilesEntry.FileType


nfs_cli = Typer(name="nfs", help="Read and edit modal.NetworkFileSystem volumes.", no_args_is_help=True)


@nfs_cli.command(name="list", help="List the names of all network file systems.")
@synchronizer.create_blocking
async def list(env: Optional[str] = ENV_OPTION, json: Optional[bool] = False):
    env = ensure_env(env)

    client = await _Client.from_env()
    response = await retry_transient_errors(
        client.stub.SharedVolumeList, api_pb2.SharedVolumeListRequest(environment_name=env)
    )
    env_part = f" in environment '{env}'" if env else ""
    column_names = ["Name", "Location", "Created at"]
    rows = []
    locale_tz = datetime.now().astimezone().tzinfo
    for item in response.items:
        rows.append(
            [
                item.label,
                display_location(item.cloud_provider),
                str(datetime.fromtimestamp(item.created_at, tz=locale_tz)),
            ]
        )
    display_table(column_names, rows, json, title=f"Shared Volumes{env_part}")


def gen_usage_code(label):
    return f"""
@stub.function(network_file_systems={{"/my_vol": modal.NetworkFileSystem.from_name("{label}")}})
def some_func():
    os.listdir("/my_vol")
"""


@nfs_cli.command(name="create", help="Create a named network file system.")
def create(
    name: str,
    cloud: str = typer.Option("aws", help="Cloud provider to create the file system in. One of aws|gcp."),
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    volume = modal.NetworkFileSystem.new(cloud=cloud)
    volume._deploy(name, environment_name=env)
    console = Console()
    console.print(f"Created volume '{name}' in {cloud.upper()}. \n\nCode example:\n")
    usage = Syntax(gen_usage_code(name), "python")
    console.print(usage)


async def _volume_from_name(deployment_name: str) -> _NetworkFileSystemHandle:
    network_file_system = await _NetworkFileSystem.lookup(
        deployment_name, environment_name=None
    )  # environment None will take value from config
    if not isinstance(network_file_system, _NetworkFileSystemHandle):
        raise Exception("The specified app entity is not a network file system")
    return network_file_system


@nfs_cli.command(name="ls", help="List files and directories in a network file system.")
@synchronizer.create_blocking
async def ls(
    volume_name: str,
    path: str = typer.Argument(default="/"),
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    volume = await _volume_from_name(volume_name)
    try:
        entries = await volume.listdir(path)
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message)
        raise

    if sys.stdout.isatty():
        console = Console()
        console.print(f"Directory listing of '{path}' in '{volume_name}'")
        table = Table()

        table.add_column("filename")
        table.add_column("type")

        for entry in entries:
            filetype = "dir" if entry.type == FileType.DIRECTORY else "file"
            table.add_row(entry.path, filetype)
        console.print(table)
    else:
        for entry in entries:
            print(entry.path)


PIPE_PATH = Path("-")


@nfs_cli.command(
    name="put",
    help="""Upload a file or directory to a network file system.

Remote parent directories will be created as needed.

Ending the REMOTE_PATH with a forward slash (/), it's assumed to be a directory and the file will be uploaded with its current name under that directory.
""",
)
@synchronizer.create_blocking
async def put(
    volume_name: str,
    local_path: str,
    remote_path: str = typer.Argument(default="/"),
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    volume = await _volume_from_name(volume_name)
    if remote_path.endswith("/"):
        remote_path = remote_path + os.path.basename(local_path)
    console = Console()

    if Path(local_path).is_dir():
        spinner = step_progress(f"Uploading directory '{local_path}' to '{remote_path}'...")
        with Live(spinner, console=console):
            await volume.add_local_dir(local_path, remote_path)
        console.print(step_completed(f"Uploaded directory '{local_path}' to '{remote_path}'"))

    elif "*" in local_path:
        raise UsageError("Glob uploads are currently not supported")
    else:
        spinner = step_progress(f"Uploading file '{local_path}' to '{remote_path}'...")
        with Live(spinner, console=console):
            written_bytes = await volume.add_local_file(local_path, remote_path)
        console.print(
            step_completed(f"Uploaded file '{local_path}' to '{remote_path}' ({written_bytes} bytes written)")
        )


class CliError(Exception):
    def __init__(self, message):
        self.message = message


async def _glob_download(
    volume: _NetworkFileSystemHandle, remote_glob_path: str, local_destination: Path, overwrite: bool
):
    q: asyncio.Queue[Tuple[Optional[Path], Optional[api_pb2.SharedVolumeListFilesEntry]]] = asyncio.Queue()

    async def producer():
        async for entry in volume.iterdir(remote_glob_path):
            output_path = local_destination / entry.path
            if output_path.exists():
                if overwrite:
                    if entry.type == api_pb2.SharedVolumeListFilesEntry.FILE:
                        os.remove(output_path)
                    else:
                        shutil.rmtree(output_path)
                else:
                    raise CliError(
                        f"Output path '{output_path}' already exists. Use --force to overwrite the output directory"
                    )
            await q.put((output_path, entry))
        for _ in range(10):
            await q.put((None, None))

    async def consumer():
        while 1:
            output_path, entry = await q.get()
            if output_path is None:
                return
            try:
                if entry.type == api_pb2.SharedVolumeListFilesEntry.FILE:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_path.open("wb") as fp:
                        b = 0
                        async for chunk in volume.read_file(entry.path):
                            b += fp.write(chunk)

                    print(f"Wrote {b} bytes to {output_path}", file=sys.stderr)
            finally:
                q.task_done()

    tasks = []
    tasks.append(asyncio.create_task(producer()))
    for _ in range(10):
        tasks.append(asyncio.create_task(consumer()))

    await asyncio.gather(*tasks)


@nfs_cli.command(name="get")
@synchronizer.create_blocking
async def get(
    volume_name: str,
    remote_path: str,
    local_destination: str = typer.Argument("."),
    force: bool = False,
    env: Optional[str] = ENV_OPTION,
):
    """Download a file from a shared volume.

    Specifying a glob pattern (using any `*` or `**` patterns) as the `remote_path` will download all matching *files*, preserving
    the source directory structure for the matched files.

    For example, to download an entire shared volume into `dump_volume`:

    ```bash
    modal volume get <volume-name> "**" dump_volume
    ```

    Use "-" (a hyphen) as LOCAL_DESTINATION to write contents of file to stdout (only for non-glob paths).
    """
    ensure_env(env)
    destination = Path(local_destination)
    volume = await _volume_from_name(volume_name)

    if "*" in remote_path:
        try:
            await _glob_download(volume, remote_path, destination, force)
        except CliError as exc:
            print(exc.message)
        return

    if destination != PIPE_PATH:
        if destination.is_dir():
            destination = destination / remote_path.rsplit("/")[-1]

        if destination.exists() and not force:
            raise UsageError(f"'{destination}' already exists")

        if not destination.parent.exists():
            raise UsageError(f"Local directory '{destination.parent}' does not exist")

    @contextmanager
    def _destination_stream():
        if destination == PIPE_PATH:
            yield sys.stdout.buffer
        else:
            with NamedTemporaryFile(delete=False) as fp:
                yield fp
            shutil.move(fp.name, destination)

    b = 0
    try:
        with _destination_stream() as fp:
            async for chunk in volume.read_file(remote_path):
                fp.write(chunk)
                b += len(chunk)
    except GRPCError as exc:
        if exc.status in (Status.NOT_FOUND, Status.INVALID_ARGUMENT):
            raise UsageError(exc.message)

    if destination != PIPE_PATH:
        print(f"Wrote {b} bytes to '{destination}'", file=sys.stderr)


@nfs_cli.command(name="rm", help="Delete a file or directory from a network file system.")
@synchronizer.create_blocking
async def rm(
    volume_name: str,
    remote_path: str,
    recursive: bool = typer.Option(False, "-r", "--recursive", help="Delete directory recursively"),
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    volume = await _volume_from_name(volume_name)
    try:
        await volume.remove_file(remote_path, recursive=recursive)
    except GRPCError as exc:
        if exc.status in (Status.NOT_FOUND, Status.INVALID_ARGUMENT):
            raise UsageError(exc.message)
        raise
