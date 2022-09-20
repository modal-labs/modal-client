import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import typer
from google.protobuf import empty_pb2
from grpclib import GRPCError, Status
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from typer import Typer

import modal
from modal.aio import aio_lookup
from modal.client import AioClient
from modal.shared_volume import AioSharedVolumeHandle, _SharedVolumeHandle
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer

FileType = api_pb2.SharedVolumeListFilesEntry.FileType

volume_cli = Typer(name="volume", help="Read and edit shared volumes.", no_args_is_help=True)


@volume_cli.command(name="list", help="List the names of all shared volumes.")
@synchronizer
async def list():
    client = await AioClient.from_env()
    response = await client.stub.SharedVolumeList(empty_pb2.Empty())
    if sys.stdout.isatty():
        table = Table(title="Shared Volumes")
        table.add_column("Name")
        table.add_column("Created at", justify="right")
        locale_tz = datetime.now().astimezone().tzinfo
        for item in response.items:
            table.add_row(item.label, str(datetime.fromtimestamp(item.created_at, tz=locale_tz)))
        console = Console()
        console.print(table)
    else:
        for item in response.items:
            print(item.label)


def gen_usage_code(label):
    return f"""
@stub.function(shared_volumes={{"/my_vol": modal.ref("{label}")}})
def some_func():
    os.listdir("/my_vol")
"""


@volume_cli.command(name="create", help="Create a named shared volume.")
def create(name: str):
    stub = modal.Stub()
    stub.entity = modal.SharedVolume()
    stub.deploy(name=name, show_progress=False)
    console = Console()
    console.print(f"Created volume '{name}'\n\nUsage:\n")
    usage = Syntax(gen_usage_code(name), "python")
    console.print(usage)


async def volume_from_name(deployment_name) -> _SharedVolumeHandle:
    shared_volume = await aio_lookup(deployment_name)
    if not isinstance(shared_volume, AioSharedVolumeHandle):
        raise Exception("The specified app entity is not a shared volume")
    return shared_volume


@volume_cli.command(name="ls", help="List files and directories in a shared volume.")
@synchronizer
async def ls(volume_name: str, path: str = typer.Argument(default="/")):
    volume = await volume_from_name(volume_name)
    try:
        entries = await volume.listdir(path)
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            print(exc.message)
            return
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


def copy(src, dst, overwrite):
    shutil.copytree(src, dst, dirs_exist_ok=overwrite)


PIPE_PATH = Path("-")


@volume_cli.command(
    name="put",
    short_help="Upload a file to a shared volume.",
    help="""Upload a file to a shared volume.
Remote parent directories will be created as needed.

Ending the REMOTE_PATH with a forward slash (/), it's assumed to be a directory and the file will be uploaded with its current name under that directory.
""",
)
@synchronizer
async def put(
    volume_name: str,
    local_path: str,
    remote_path: str = typer.Argument(default="/"),
):
    volume = await volume_from_name(volume_name)
    if remote_path.endswith("/"):
        remote_path = remote_path + os.path.basename(local_path)

    if Path(local_path).is_dir():
        print("Directory uploads are currently not supported")
        exit(1)
    else:
        with Path(local_path).open("rb") as fd:
            written_bytes = await volume.write_file(remote_path, fd)
        print(f"Wrote {written_bytes} bytes to remote file {remote_path}")


@volume_cli.command(
    name="get",
    short_help="Download a file from a shared volume",
    help="""Download a file from a shared volume

Use - as LOCAL_DIR_OR_FILE to write contents of file to stdout
""",
)
@synchronizer
async def get(volume_name: str, remote_path: str, local_dir_or_file: str, force: bool = False):
    destination = Path(local_dir_or_file)
    if not destination == PIPE_PATH:
        if destination.is_dir():
            destination = destination / remote_path.rsplit("/")[-1]

        if destination.exists() and not force:
            print(f"'{destination}' already exists")
            exit(1)

        if not destination.parent.exists():
            print(f"Local directory '{destination.parent}' does not exist")
            exit(1)

    volume = await volume_from_name(volume_name)

    @contextmanager
    def _destination_stream():
        if destination == PIPE_PATH:
            yield sys.stdout.buffer
        else:
            with NamedTemporaryFile(delete=False) as fp:
                yield fp
            Path(fp.name).rename(destination)

    b = 0
    try:
        with _destination_stream() as fp:
            async for chunk in volume.read_file(remote_path):
                fp.write(chunk)
                b += len(chunk)
    except GRPCError as exc:
        if exc.status in (Status.NOT_FOUND, Status.INVALID_ARGUMENT):
            print(exc.message)
            exit(1)

    if destination != PIPE_PATH:
        print(f"Wrote {b} bytes to '{destination}'", file=sys.stderr)


@volume_cli.command(name="rm", help="Delete a file or directory from a shared volume")
@synchronizer
async def rm(
    volume_name: str,
    remote_path: str,
    recursive: bool = typer.Option(False, "-r", "--recursive", help="Delete directory recursively"),
):
    volume = await volume_from_name(volume_name)
    try:
        await volume.remove_file(remote_path, recursive=recursive)
    except GRPCError as exc:
        if exc.status in (Status.NOT_FOUND, Status.INVALID_ARGUMENT):
            print(exc.message)
            exit(1)
        raise
