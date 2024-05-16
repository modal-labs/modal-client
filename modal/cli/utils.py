# Copyright Modal Labs 2022
import asyncio
from datetime import datetime
from typing import Sequence, Union

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.json import JSON
from rich.table import Column, Table
from rich.text import Text

from .._output import OutputManager, get_app_logs_loop
from .._utils.async_utils import synchronizer
from ..client import _Client


@synchronizer.create_blocking
async def stream_app_logs(app_id: str):
    client = await _Client.from_env()
    output_mgr = OutputManager(None, None, f"Tailing logs for {app_id}")
    try:
        with output_mgr.show_status_spinner():
            await get_app_logs_loop(app_id, client, output_mgr)
    except asyncio.CancelledError:
        pass
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message)
        else:
            raise
    except KeyboardInterrupt:
        pass


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
    title: str = None,
):
    console = Console()
    if json:
        json_data = [{col: _plain(row[i]) for i, col in enumerate(columns)} for row in rows]
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
