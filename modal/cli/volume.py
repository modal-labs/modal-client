# Copyright Modal Labs 2022
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.live import Live
from rich.syntax import Syntax
from rich.table import Table
from typer import Argument, Option, Typer

import modal
from modal._output import step_completed, step_progress
from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal.cli._download import _glob_download
from modal.cli.utils import ENV_OPTION, display_table
from modal.client import _Client
from modal.environments import ensure_env
from modal.volume import _Volume, _VolumeUploadContextManager
from modal_proto import api_pb2

FileType = api_pb2.VolumeListFilesEntry.FileType
PIPE_PATH = Path("-")

volume_cli = Typer(
    name="volume",
    no_args_is_help=True,
    help="""
    Read and edit `modal.Volume` volumes.

    Note: users of `modal.NetworkFileSystem` should use the `modal nfs` command instead.
    """,
)


def humanize_filesize(value: int) -> str:
    if value < 0:
        raise ValueError("value should be >= 0")
    suffix = (" KiB", " MiB", " GiB", " TiB", " PiB", " EiB", " ZiB")
    format = "%.1f"
    base = 1024
    bytes_ = float(value)
    if bytes_ < base:
        return f"{bytes_:0.0f} B"
    for i, s in enumerate(suffix):
        unit = base ** (i + 2)
        if bytes_ < unit:
            break
    return format % (base * bytes_ / unit) + s


@volume_cli.command(name="create", help="Create a named, persistent modal.Volume.")
def create(
    name: str,
    env: Optional[str] = ENV_OPTION,
):
    env_name = ensure_env(env)
    modal.Volume.create_deployed(name, environment_name=env)
    usage_code = f"""
@stub.function(volumes={{"/my_vol": modal.Volume.from_name("{name}")}})
def some_func():
    os.listdir("/my_vol")
"""

    console = Console()
    console.print(f"Created volume '{name}' in environment '{env_name}'. \n\nCode example:\n")
    usage = Syntax(usage_code, "python")
    console.print(usage)


@volume_cli.command(name="get")
@synchronizer.create_blocking
async def get(
    volume_name: str,
    remote_path: str,
    local_destination: str = Argument("."),
    force: bool = False,
    env: Optional[str] = ENV_OPTION,
):
    """Download files from a modal.Volume.

    Specifying a glob pattern (using any `*` or `**` patterns) as the `remote_path` will download all matching *files*, preserving
    the source directory structure for the matched files.

    **Example**

    ```bash
    modal volume get <volume-name> logs/april-12-1.txt .
    modal volume get <volume-name> "**" dump_volume
    ```

    Use "-" (a hyphen) as LOCAL_DESTINATION to write contents of file to stdout (only for non-glob paths).
    """
    ensure_env(env)
    destination = Path(local_destination)
    volume = await _Volume.lookup(volume_name, environment_name=env)

    def is_file_fn(entry):
        return entry.type == FileType.FILE

    if "*" in remote_path:
        await _glob_download(volume, is_file_fn, remote_path, destination, force)
        return

    if destination != PIPE_PATH:
        if destination.is_dir():
            destination = destination / remote_path.rsplit("/")[-1]

        if destination.exists() and not force:
            raise UsageError(f"'{destination}' already exists")
        elif not destination.parent.exists():
            raise UsageError(f"Local directory '{destination.parent}' does not exist")

    @contextmanager
    def _destination_stream():
        if destination == PIPE_PATH:
            yield sys.stdout.buffer
        else:
            with NamedTemporaryFile(delete=False) as fp:
                yield fp
            shutil.move(fp.name, destination)

    try:
        with _destination_stream() as fp:
            await volume.read_file_into_fileobj(remote_path.lstrip("/"), fileobj=fp, progress=destination != PIPE_PATH)
    except FileNotFoundError as exc:
        raise UsageError(str(exc))
    except GRPCError as exc:
        raise UsageError(exc.message) if exc.status == Status.INVALID_ARGUMENT else exc


@volume_cli.command(name="list", help="List the details of all modal.Volume volumes in an environment.")
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


@volume_cli.command(name="ls", help="List files and directories in a modal.Volume volume.")
@synchronizer.create_blocking
async def ls(
    volume_name: str,
    path: str = Argument(default="/"),
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    vol = await _Volume.lookup(volume_name, environment_name=env)
    if not isinstance(vol, _Volume):
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
        for name in ["filename", "type", "created/modified", "size"]:
            table.add_column(name)

        locale_tz = datetime.now().astimezone().tzinfo
        for entry in entries:
            if entry.type == FileType.DIRECTORY:
                filetype = "dir"
            elif entry.type == FileType.SYMLINK:
                filetype = "link"
            else:
                filetype = "file"
            table.add_row(
                entry.path,
                filetype,
                str(datetime.fromtimestamp(entry.mtime, tz=locale_tz)),
                humanize_filesize(entry.size),
            )
        console.print(table)
    else:
        for entry in entries:
            print(entry.path)


@volume_cli.command(
    name="put",
    help="""Upload a file or directory to a volume.

Remote parent directories will be created as needed.

Ending the REMOTE_PATH with a forward slash (/), it's assumed to be a directory and the file will be uploaded with its current name under that directory.
""",
)
@synchronizer.create_blocking
async def put(
    volume_name: str,
    local_path: str = Argument(),
    remote_path: str = Argument(default="/"),
    force: bool = Option(False, "-f", "--force", help="Overwrite existing files."),
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    vol = await _Volume.lookup(volume_name, environment_name=env)
    if not isinstance(vol, _Volume):
        raise UsageError("The specified app entity is not a modal.Volume")

    if remote_path.endswith("/"):
        remote_path = remote_path + os.path.basename(local_path)
    console = Console()

    if Path(local_path).is_dir():
        spinner = step_progress(f"Uploading directory '{local_path}' to '{remote_path}'...")
        with Live(spinner, console=console):
            try:
                async with _VolumeUploadContextManager(vol.object_id, vol._client, force=force) as batch:
                    batch.put_directory(local_path, remote_path)
            except FileExistsError as exc:
                raise UsageError(str(exc))
        console.print(step_completed(f"Uploaded directory '{local_path}' to '{remote_path}'"))
    elif "*" in local_path:
        raise UsageError("Glob uploads are currently not supported")
    else:
        spinner = step_progress(f"Uploading file '{local_path}' to '{remote_path}'...")
        with Live(spinner, console=console):
            try:
                async with _VolumeUploadContextManager(vol.object_id, vol._client, force=force) as batch:
                    batch.put_file(local_path, remote_path)
            except FileExistsError as exc:
                raise UsageError(str(exc))
        console.print(step_completed(f"Uploaded file '{local_path}' to '{remote_path}'"))


@volume_cli.command(name="rm", help="Delete a file or directory from a volume.")
@synchronizer.create_blocking
async def rm(
    volume_name: str,
    remote_path: str,
    recursive: bool = Option(False, "-r", "--recursive", help="Delete directory recursively"),
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    volume = await _Volume.lookup(volume_name, environment_name=env)
    if not isinstance(volume, _Volume):
        raise UsageError("The specified app entity is not a modal.Volume")
    try:
        await volume.remove_file(remote_path, recursive=recursive)
    except GRPCError as exc:
        if exc.status in (Status.NOT_FOUND, Status.INVALID_ARGUMENT):
            raise UsageError(exc.message)
        raise


@volume_cli.command(
    name="cp", help="Copy source file to destination file or multiple source files to destination directory."
)
@synchronizer.create_blocking
async def cp(
    volume_name: str,
    paths: List[str],  # accepts multiple paths, last path is treated as destination path
    env: Optional[str] = ENV_OPTION,
):
    ensure_env(env)
    volume = await _Volume.lookup(volume_name, environment_name=env)
    if not isinstance(volume, _Volume):
        raise UsageError("The specified app entity is not a modal.Volume")
    *src_paths, dst_path = paths
    await volume.copy_files(src_paths, dst_path)


@volume_cli.command(name="delete", help="Delete a named, persistent modal.Volume.")
@synchronizer.create_blocking
async def delete(
    name: str = Argument(help="Name of the modal.Volume to be deleted. Case sensitive"),
    confirm: bool = Option(default=False, help="Set this flag to delete without prompting for confirmation"),
    env: Optional[str] = ENV_OPTION,
):
    if not confirm:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the modal.Volume '{name}'?",
            default=False,
            abort=True,
        )
    env = ensure_env(env)
    client = await _Client.from_env()
    response = await retry_transient_errors(client.stub.VolumeList, api_pb2.VolumeListRequest(environment_name=env))
    for item in response.items:
        if item.label == name:
            await retry_transient_errors(
                client.stub.VolumeDelete, api_pb2.VolumeDeleteRequest(volume_id=item.volume_id, environment_name=env)
            )
            return
    else:
        raise UsageError(f"No modal.Volume found with name '{name}' in modal.Environment '{env}'")
