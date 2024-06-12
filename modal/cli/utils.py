# Copyright Modal Labs 2022
import asyncio
from datetime import datetime
from typing import Optional, Sequence, Union

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.json import JSON
from rich.table import Column, Table
from rich.text import Text

from modal_proto import api_pb2

from .._output import OutputManager, get_app_logs_loop
from .._utils.async_utils import synchronizer
from ..client import _Client
from ..environments import ensure_env
from ..exception import NotFoundError


@synchronizer.create_blocking
async def stream_app_logs(app_id: Optional[str] = None, task_id: Optional[str] = None):
    client = await _Client.from_env()
    output_mgr = OutputManager(None, None, f"Tailing logs for {app_id}")
    try:
        with output_mgr.show_status_spinner():
            await get_app_logs_loop(client, output_mgr, app_id, task_id)
    except asyncio.CancelledError:
        pass
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message)
        else:
            raise
    except KeyboardInterrupt:
        pass


@synchronizer.create_blocking
async def get_app_id_from_name(name: str, env: Optional[str], client: Optional[_Client] = None) -> str:
    if client is None:
        client = await _Client.from_env()
    env_name = ensure_env(env)
    request = api_pb2.AppGetByDeploymentNameRequest(
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, name=name, environment_name=env_name
    )
    try:
        resp = await client.stub.AppGetByDeploymentName(request)
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message or "")
        raise
    if not resp.app_id:
        env_comment = f" in the '{env_name}' environment" if env_name else ""
        raise NotFoundError(f"Could not find a deployed app named '{name}'{env_comment}.")
    return resp.app_id


def timestamp_to_local(ts: float, isotz: bool = True) -> str:
    if ts > 0:
        locale_tz = datetime.now().astimezone().tzinfo
        dt = datetime.fromtimestamp(ts, tz=locale_tz)
        if isotz:
            return dt.isoformat(sep=" ", timespec="seconds")
        else:
            return f"{datetime.strftime(dt, '%Y-%m-%d %H:%M')} {locale_tz.tzname(dt)}"
    else:
        return None


def _plain(text: Union[Text, str]) -> str:
    return text.plain if isinstance(text, Text) else text


def display_table(
    columns: Sequence[Union[Column, str]],
    rows: Sequence[Sequence[Union[Text, str]]],
    json: bool = False,
    title: str = "",
):
    def col_to_str(col: Union[Column, str]) -> str:
        return str(col.header) if isinstance(col, Column) else col

    console = Console()
    if json:
        json_data = [{col_to_str(col): _plain(row[i]) for i, col in enumerate(columns)} for row in rows]
        console.print(JSON.from_data(json_data))
    else:
        table = Table(*columns, title=title)
        for row in rows:
            table.add_row(*row)
        console.print(table)


ENV_OPTION_HELP = """Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
"""
ENV_OPTION = typer.Option(default=None, help=ENV_OPTION_HELP)

YES_OPTION = typer.Option(False, "-y", "--yes", help="Run without pausing for confirmation.")
