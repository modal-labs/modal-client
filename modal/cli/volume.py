# Copyright Modal Labs 2022
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.syntax import Syntax
from typer import Argument, Option, Typer

import modal
from modal._output import OutputManager, ProgressHandler
from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal.cli._download import _volume_download
from modal.cli.utils import ENV_OPTION, YES_OPTION, display_table, timestamp_to_local
from modal.client import _Client
from modal.environments import ensure_env
from modal.volume import _Volume, _VolumeUploadContextManager
from modal_proto import api_pb2

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


@volume_cli.command(name="create", help="Create a named, persistent modal.Volume.", rich_help_panel="Management")
def create(
    name: str,
    env: Optional[str] = ENV_OPTION,
    version: Optional[int] = Option(default=None, help="VolumeFS version. (Experimental)"),
):
    env_name = ensure_env(env)
    modal.Volume.create_deployed(name, environment_name=env, version=version)
    usage_code = f"""
@app.function(volumes={{"/my_vol": modal.Volume.from_name("{name}")}})
def some_func():
    os.listdir("/my_vol")
"""

    console = Console()
    console.print(f"Created Volume '{name}' in environment '{env_name}'. \n\nCode example:\n")
    usage = Syntax(usage_code, "python")
    console.print(usage)


@volume_cli.command(name="get", rich_help_panel="File operations")
@synchronizer.create_blocking
async def get(
    volume_name: str,
    remote_path: str,
    local_destination: str = Argument("."),
    force: bool = False,
    env: Optional[str] = ENV_OPTION,
):
    """Download files from a modal.Volume object.

    If a folder is passed for REMOTE_PATH, the contents of the folder will be downloaded
    recursively, including all subdirectories.

    **Example**

    ```
    modal volume get <volume_name> logs/april-12-1.txt
    modal volume get <volume_name> / volume_data_dump
    ```

    Use "-" as LOCAL_DESTINATION to write file contents to standard output.
    """
    ensure_env(env)
    destination = Path(local_destination)
    volume = await _Volume.lookup(volume_name, environment_name=env)
    console = Console()
    progress_handler = ProgressHandler(type="download", console=console)
    with progress_handler.live:
        await _volume_download(volume, remote_path, destination, force, progress_cb=progress_handler.progress)
    console.print(OutputManager.step_completed("Finished downloading files to local!"))


@volume_cli.command(
    name="list",
    help="List the details of all modal.Volume volumes in an Environment.",
    rich_help_panel="Management",
)
@synchronizer.create_blocking
async def list(env: Optional[str] = ENV_OPTION, json: Optional[bool] = False):
    env = ensure_env(env)
    client = await _Client.from_env()
    response = await retry_transient_errors(client.stub.VolumeList, api_pb2.VolumeListRequest(environment_name=env))
    env_part = f" in environment '{env}'" if env else ""
    column_names = ["Name", "Created at"]
    rows = []
    for item in response.items:
        rows.append([item.label, timestamp_to_local(item.created_at, json)])
    display_table(column_names, rows, json, title=f"Volumes{env_part}")


@volume_cli.command(
    name="ls",
    help="List files and directories in a modal.Volume volume.",
    rich_help_panel="File operations",
)
@synchronizer.create_blocking
async def ls(
    volume_name: str,
    path: str = Argument(default="/"),
    json: bool = False,
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

    if not json and not sys.stdout.isatty():
        # Legacy behavior -- I am not sure why exactly we did this originally but I don't want to break it
        for entry in entries:
            print(entry.path)
    else:
        rows = []
        for entry in entries:
            if entry.type == api_pb2.FileEntry.FileType.DIRECTORY:
                filetype = "dir"
            elif entry.type == api_pb2.FileEntry.FileType.SYMLINK:
                filetype = "link"
            else:
                filetype = "file"
            rows.append(
                (
                    entry.path.encode("unicode_escape").decode("utf-8"),
                    filetype,
                    timestamp_to_local(entry.mtime, False),
                    humanize_filesize(entry.size),
                )
            )
        columns = ["Filename", "Type", "Created/Modified", "Size"]
        title = f"Directory listing of '{path}' in '{volume_name}'"
        display_table(columns, rows, json, title)


@volume_cli.command(
    name="put",
    help="""Upload a file or directory to a modal.Volume.

Remote parent directories will be created as needed.

Ending the REMOTE_PATH with a forward slash (/), it's assumed to be a directory
and the file will be uploaded with its current name under that directory.
""",
    rich_help_panel="File operations",
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
    progress_handler = ProgressHandler(type="upload", console=console)

    if Path(local_path).is_dir():
        with progress_handler.live:
            try:
                async with _VolumeUploadContextManager(
                    vol.object_id, vol._client, progress_cb=progress_handler.progress, force=force
                ) as batch:
                    batch.put_directory(local_path, remote_path)
            except FileExistsError as exc:
                raise UsageError(str(exc))
        console.print(OutputManager.step_completed(f"Uploaded directory '{local_path}' to '{remote_path}'"))
    elif "*" in local_path:
        raise UsageError("Glob uploads are currently not supported")
    else:
        with progress_handler.live:
            try:
                async with _VolumeUploadContextManager(
                    vol.object_id, vol._client, progress_cb=progress_handler.progress, force=force
                ) as batch:
                    batch.put_file(local_path, remote_path)

            except FileExistsError as exc:
                raise UsageError(str(exc))
        console.print(OutputManager.step_completed(f"Uploaded file '{local_path}' to '{remote_path}'"))


@volume_cli.command(
    name="rm", help="Delete a file or directory from a modal.Volume.", rich_help_panel="File operations"
)
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
    name="cp",
    help=(
        "Copy within a modal.Volume. "
        "Copy source file to destination file or multiple source files to destination directory."
    ),
    rich_help_panel="File operations",
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


@volume_cli.command(
    name="delete",
    help="Delete a named, persistent modal.Volume.",
    rich_help_panel="Management",
)
@synchronizer.create_blocking
async def delete(
    volume_name: str = Argument(help="Name of the modal.Volume to be deleted. Case sensitive"),
    yes: bool = YES_OPTION,
    env: Optional[str] = ENV_OPTION,
):
    if not yes:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the modal.Volume '{volume_name}'?",
            default=False,
            abort=True,
        )

    await _Volume.delete(label=volume_name, environment_name=env)
