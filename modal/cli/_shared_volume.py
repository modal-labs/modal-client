# Copyright Modal Labs 2022
from typing import Optional

import typer
from typer import Typer

from modal.cli.utils import ENV_OPTION
from modal.exception import DeprecationError
from modal_proto import api_pb2

FileType = api_pb2.SharedVolumeListFilesEntry.FileType


def depr_cmd(cmd):
    return f"Use `{cmd}` instead!"


vol_cli = Typer(name="volume", help=depr_cmd("modal nfs"), no_args_is_help=True, hidden=True)


@vol_cli.command(name="list", help=depr_cmd("modal nfs list"), deprecated=True)
def list(env: Optional[str] = ENV_OPTION, json: Optional[bool] = False):
    raise DeprecationError(depr_cmd("modal nfs list"))


@vol_cli.command(name="create", help=depr_cmd("modal nfs create"), deprecated=True)
def create(name: str, cloud: str = typer.Option("aws"), env: Optional[str] = ENV_OPTION):
    raise DeprecationError(depr_cmd("modal nfs create"))


@vol_cli.command(name="ls", help=depr_cmd("modal nfs ls"), deprecated=True)
def ls(
    volume_name: str,
    path: str = typer.Argument(default="/"),
    env: Optional[str] = ENV_OPTION,
):
    raise DeprecationError(depr_cmd("modal nfs ls"))


@vol_cli.command(name="get", help=depr_cmd("modal nfs get"), deprecated=True)
async def get(
    volume_name: str,
    remote_path: str,
    local_destination: str = typer.Argument("."),
    force: bool = False,
    env: Optional[str] = ENV_OPTION,
):
    raise DeprecationError(depr_cmd("modal nfs get"))


@vol_cli.command(name="rm", help=depr_cmd("modal nfs rm"), deprecated=True)
async def rm(
    volume_name: str,
    remote_path: str,
    recursive: bool = typer.Option(False, "-r", "--recursive"),
    env: Optional[str] = ENV_OPTION,
):
    raise DeprecationError(depr_cmd("modal nfs rm"))
