import contextlib
import os
import shutil
import subprocess
import sys
from datetime import datetime

import typer
from google.protobuf import empty_pb2
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from typer import Typer

import modal
from modal.client import AioClient
from modal_utils.async_utils import synchronizer

volume_cli = Typer(no_args_is_help=True, help="Manage read/write shared volumes")


@volume_cli.command(name="list")
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


@volume_cli.command(name="create", help="Create a named volume")
def create(volume_name: str):
    stub = modal.Stub()
    stub.entity = modal.SharedVolume()
    stub.deploy(name=volume_name, show_progress=False)
    console = Console()
    console.print(f"Created volume '{volume_name}'\n\nUsage:\n")
    usage = Syntax(gen_usage_code(volume_name), "python")
    console.print(usage)


def shell(cmd):
    subprocess.call(cmd)


@contextlib.contextmanager
def shared_volume_stub(volume_name):
    stub = modal.Stub()
    # TODO: would be nice if functions could be accessed on the app directly, e.g. `app.listdir(...)` like other objects
    functions = {"shell": stub.function(shared_volumes={"/vol": modal.ref(volume_name)})(shell)}
    with stub.run(show_progress=False):
        yield functions


@volume_cli.command(name="ls", help="List files and directories in a named volume")
def ls(volume_name: str, path: str = typer.Argument(default="/")):
    console = Console()
    console.print(f"Directory listing of '{path}' in '{volume_name}':")
    remote_path = os.path.join("/vol", path.lstrip("/"))
    with shared_volume_stub(volume_name) as functions:
        functions["shell"](["ls", "-l", remote_path])


def copy(src, dst, overwrite):
    shutil.copytree(src, dst, dirs_exist_ok=overwrite)


@volume_cli.command(name="put", help="Upload a file or directory to a named volume")
def put(volume_name: str, local_dir: str, remote_path: str = typer.Argument(default="/"), overwrite: bool = False):
    stub = modal.Stub()
    local_basename = os.path.basename(local_dir)
    remote_mount_dir = os.path.join("/source", local_basename)
    remote_shared_volume_rel_path = os.path.join(remote_path, local_basename)

    modal_copy = stub.function(
        mounts=[modal.Mount(local_dir=local_dir, remote_dir=remote_mount_dir)],
        shared_volumes={"/destination": modal.ref(volume_name)},
    )(copy)

    with stub.run(show_progress=False):
        destination = os.path.join("/destination", remote_shared_volume_rel_path.lstrip("/"))
        modal_copy(remote_mount_dir, destination, overwrite)


@volume_cli.command(name="cat", help="Print a file from a named volume to stdout")
def cat(volume_name: str, remote_path: str):
    full_remote_path = os.path.join("/vol", remote_path.lstrip("/"))
    with shared_volume_stub(volume_name) as functions:
        functions["shell"](["cat", full_remote_path])
