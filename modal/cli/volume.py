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
            print(exc.message, file=sys.stderr)
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

    if path == "/" and not entries:
        # TODO(erikbern): consider this a big fat TODO for
        # rethinking how we create and work with shared volumes
        # across cloud vendors
        print(
            "Note: this command only lists data in AWS."
            " If you created data on an A100 running in GCP,"
            " it will not be listed here."
        )


PIPE_PATH = Path("-")


@volume_cli.command(
    name="put",
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
        print("Directory uploads are currently not supported", file=sys.stderr)
        exit(1)
    elif "*" in local_path:
        print("Glob uploads are currently not supported", file=sys.stderr)
        exit(1)
    else:
        with Path(local_path).open("rb") as fd:
            written_bytes = await volume.write_file(remote_path, fd)
        print(f"Wrote {written_bytes} bytes to remote file {remote_path}", file=sys.stderr)


class CliError(Exception):
    def __init__(self, message):
        self.message = message


async def _glob_download(
    volume: AioSharedVolumeHandle, remote_glob_path: str, local_destination: Path, overwrite: bool
):
    q: asyncio.Queue[Tuple[Optional[Path], Optional[api_pb2.SharedVolumeListFilesEntry]]] = asyncio.Queue()

    for entry in await volume.listdir(remote_glob_path):
        output_path = local_destination / entry.path
        if output_path.exists():
            if overwrite:
                shutil.rmtree(output_path)
            else:
                raise CliError(
                    f"Output path '{output_path}' already exists. Use --force to overwrite the output directory"
                )
        await q.put((output_path, entry))

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
    for _ in range(10):
        tasks.append(asyncio.create_task(consumer()))
        await q.put((None, None))

    await asyncio.gather(*tasks)


@volume_cli.command(name="get")
@synchronizer
async def get(volume_name: str, remote_path: str, local_destination: str = typer.Argument("."), force: bool = False):
    """Download a file from a shared volume.\n
    Specifying a glob pattern (using any * or ** patterns) as the remote_path will download all matching *files*, preserving
    the source directory structure for the matched files.\n
    \n
    For example, to download an entire shared volume into `dump_volume`:\n

    ```bash\n
    modal volume get <volume-name> "**" dump_volume\n
    ```\n
    Use "-" (a hyphen) as LOCAL_DESTINATION to write contents of file to stdout (only for non-glob paths).
    """
    destination = Path(local_destination)
    volume = await volume_from_name(volume_name)

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
            print(f"'{destination}' already exists", file=sys.stderr)
            exit(1)

        if not destination.parent.exists():
            print(f"Local directory '{destination.parent}' does not exist", file=sys.stderr)
            exit(1)

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
            print(exc.message, file=sys.stderr)
            exit(1)

    if destination != PIPE_PATH:
        print(f"Wrote {b} bytes to '{destination}'", file=sys.stderr)


@volume_cli.command(name="rm", help="Delete a file or directory from a shared volume.")
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
            print(exc.message, file=sys.stderr)
            exit(1)
        raise
