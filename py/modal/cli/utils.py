# Copyright Modal Labs 2022
import asyncio
import io
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from csv import writer as csv_writer
from datetime import datetime
from json import dumps
from typing import Optional, Union

import typer
from rich.table import Column, Table
from rich.text import Text

from .._logs import LogsFilters, fetch_logs, tail_logs
from .._output.pty import _build_log_prefix, get_app_logs_loop
from .._utils.async_utils import synchronizer
from ..client import _Client
from ..exception import InvalidError
from ..output import OutputManager


@synchronizer.create_blocking
async def stream_app_logs(
    app_id: Optional[str] = None,
    task_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    app_logs_url: Optional[str] = None,
    show_timestamps: bool = False,
    follow: bool = False,
    prefix_fields: Optional[list[str]] = None,
    filters: Optional[LogsFilters] = None,
):
    if filters is None:
        filters = LogsFilters()
    client = await _Client.from_env()
    output_mgr = OutputManager.get()
    output_mgr.set_timestamps(show_timestamps)

    # Determine the display ID for the status message
    display_id = app_id or sandbox_id or task_id

    if follow:
        status_text = f"Following logs for {display_id}..." if display_id else "Following logs..."
        log_context = output_mgr.show_status_spinner(status_text)
    else:
        log_context = nullcontext()

    try:
        with log_context:
            await get_app_logs_loop(
                client,
                output_mgr,
                app_id=app_id,
                task_id=task_id,
                sandbox_id=sandbox_id,
                follow=follow,
                prefix_fields=prefix_fields or [],
                file_descriptor=filters.source,
                function_id=filters.function_id,
                function_call_id=filters.function_call_id,
                search_text=filters.search_text,
            )
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass


async def _drain_batches(output_mgr, batches, prefixes, search_text=""):
    """Iterate over log batches and display them via the output manager."""
    last_data = ""
    async for batch in batches:
        for log in batch.items:
            if log.data:
                # Search results may lack trailing newlines
                if search_text and not log.data.endswith("\n"):
                    log.data += "\n"
                last_data = log.data
                log_prefix = _build_log_prefix(batch, log, prefixes)
                await output_mgr.put_fetched_log(log, prefix=log_prefix)
    output_mgr.flush_lines()
    # Ensure the terminal prompt starts on a new line
    if last_data and not last_data.endswith("\n"):
        output_mgr.print("")


@synchronizer.create_blocking
async def tail_app_logs(
    app_id: str,
    n: int,
    show_timestamps: bool = False,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    prefix_fields: Optional[list[str]] = None,
    filters: Optional[LogsFilters] = None,
):
    """Fetch up to the last n log entries for an app."""
    if filters is None:
        filters = LogsFilters()
    client = await _Client.from_env()
    output_mgr = OutputManager.get()
    output_mgr.set_timestamps(show_timestamps)
    batches = tail_logs(client, app_id, n, since=since, until=until, filters=filters)
    await _drain_batches(output_mgr, batches, prefix_fields or [], filters.search_text)


@synchronizer.create_blocking
async def fetch_app_logs(
    app_id: str,
    since: datetime,
    until: datetime,
    show_timestamps: bool = False,
    prefix_fields: Optional[list[str]] = None,
    filters: Optional[LogsFilters] = None,
):
    """Fetch historical logs for an app over a time range."""
    if filters is None:
        filters = LogsFilters()
    client = await _Client.from_env()
    output_mgr = OutputManager.get()
    output_mgr.set_timestamps(show_timestamps)
    batches = fetch_logs(client, app_id, since, until, filters=filters)
    await _drain_batches(output_mgr, batches, prefix_fields or [], filters.search_text)


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


def confirm_or_suggest_yes(msg: str) -> None:
    """Prompt for confirmation, or abort with a hint to use --yes if stdin is not a TTY."""
    if not sys.stdin.isatty():
        typer.echo(f"{msg} [y/N]: ")
        raise SystemExit("Aborted: no interactive terminal detected. Rerun with --yes (-y) to skip confirmation.")
    typer.confirm(msg, default=False, abort=True)
