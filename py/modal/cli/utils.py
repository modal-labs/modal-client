# Copyright Modal Labs 2022
import asyncio
import io
from collections.abc import Sequence
from csv import writer as csv_writer
from json import dumps
from typing import Optional, Union

import typer
from rich.table import Column, Table
from rich.text import Text

from modal_proto import api_pb2

from .._output.pty import get_app_logs_loop
from .._utils.async_utils import synchronizer
from ..client import _Client
from ..environments import ensure_env
from ..exception import InvalidError, NotFoundError
from ..output import OutputManager


@synchronizer.create_blocking
async def stream_app_logs(
    app_id: Optional[str] = None,
    task_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    app_logs_url: Optional[str] = None,
    show_timestamps: bool = False,
):
    client = await _Client.from_env()
    output_mgr = OutputManager.get()
    output_mgr.set_timestamps(show_timestamps)

    # Determine the display ID for the status message
    display_id = app_id or sandbox_id or task_id
    status_text = f"Tailing logs for {display_id}..." if display_id else "Tailing logs..."

    try:
        with output_mgr.show_status_spinner(status_text):
            await get_app_logs_loop(client, output_mgr, app_id=app_id, task_id=task_id, sandbox_id=sandbox_id)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass


@synchronizer.create_blocking
async def get_app_id_from_name(name: str, env: Optional[str], client: Optional[_Client] = None) -> str:
    if client is None:
        client = await _Client.from_env()
    env_name = ensure_env(env)
    request = api_pb2.AppGetByDeploymentNameRequest(name=name, environment_name=env_name)
    resp = await client.stub.AppGetByDeploymentName(request)
    if not resp.app_id:
        env_comment = f" in the '{env_name}' environment" if env_name else ""
        raise NotFoundError(f"Could not find a deployed app named '{name}'{env_comment}.")
    return resp.app_id


def _plain(text: Union[Text, str]) -> str:
    return text.plain if isinstance(text, Text) else text


def is_tty() -> bool:
    return OutputManager.get().is_terminal


def display_table(
    columns: Sequence[Union[Column, str]],
    rows: Sequence[Sequence[Union[Text, str]]],
    json: bool = False,
    csv: bool = False,
    title: str = "",
):
    def col_to_str(col: Union[Column, str]) -> str:
        return str(col.header) if isinstance(col, Column) else col

    if csv and json:
        raise InvalidError("Cannot output both JSON and CSV at the same time.")

    output = OutputManager.get()
    if json:
        json_data = [{col_to_str(col): _plain(row[i]) for i, col in enumerate(columns)} for row in rows]
        output.print_json(dumps(json_data))
    elif csv:
        csv_buffer = io.StringIO()
        writer = csv_writer(csv_buffer)
        writer.writerow([col_to_str(col) for col in columns])
        for row in rows:
            writer.writerow([_plain(cell) for cell in row])
        output.print(csv_buffer.getvalue(), end="")
    else:
        table = Table(*columns, title=title)
        for row in rows:
            table.add_row(*row)
        output.print(table)


ENV_OPTION_HELP = """Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
"""
ENV_OPTION = typer.Option(None, "-e", "--env", help=ENV_OPTION_HELP)

YES_OPTION = typer.Option(False, "-y", "--yes", help="Run without pausing for confirmation.")
