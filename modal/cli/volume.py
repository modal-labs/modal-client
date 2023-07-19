# Copyright Modal Labs 2022
from datetime import datetime
from typing import Optional

from typer import Typer

from modal.cli.utils import ENV_OPTION, display_table
from modal.client import _Client
from modal.environments import ensure_env
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer
from modal_utils.grpc_utils import retry_transient_errors

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
