# Copyright Modal Labs 2022
import asyncio
import io
import re
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from csv import writer as csv_writer
from datetime import datetime
from json import dumps

import click
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
    app_id: str | None = None,
    task_id: str | None = None,
    sandbox_id: str | None = None,
    app_logs_url: str | None = None,
    show_timestamps: bool = False,
    follow: bool = False,
    prefix_fields: list[str] | None = None,
    filters: LogsFilters | None = None,
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
    since: datetime | None = None,
    until: datetime | None = None,
    prefix_fields: list[str] | None = None,
    filters: LogsFilters | None = None,
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
    prefix_fields: list[str] | None = None,
    filters: LogsFilters | None = None,
):
    """Fetch historical logs for an app over a time range."""
    if filters is None:
        filters = LogsFilters()
    client = await _Client.from_env()
    output_mgr = OutputManager.get()
    output_mgr.set_timestamps(show_timestamps)
    batches = fetch_logs(client, app_id, since, until, filters=filters)
    await _drain_batches(output_mgr, batches, prefix_fields or [], filters.search_text)


def _plain(text: "Text | str | bool | None") -> "str | bool | None":
    return text.plain if isinstance(text, Text) else text


def _col_name_to_json_key(name: str) -> str:
    """Convert a display column name like "App ID" to a snake_case JSON key like "app_id"."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).lower().strip("_")


def is_tty() -> bool:
    return OutputManager.get().is_terminal


def display_table(
    columns: Sequence[Column | str],
    rows: Sequence[Sequence["Text | str | bool | None"]],
    json: bool = False,
    csv: bool = False,
    title: str = "",
):
    def col_to_str(col: Column | str) -> str:
        return str(col.header) if isinstance(col, Column) else col

    if csv and json:
        raise InvalidError("Cannot output both JSON and CSV at the same time.")

    output = OutputManager.get()
    if json:
        json_keys = [_col_name_to_json_key(col_to_str(col)) for col in columns]
        json_data = [{json_keys[i]: _plain(row[i]) for i in range(len(columns))} for row in rows]
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
            # rich can't render bare scalars like bools; stringify anything that isn't already
            # a renderable (str/Text) or None (which rich treats as an empty cell).
            cells: list[Text | str | None] = [
                cell if cell is None or isinstance(cell, (str, Text)) else str(cell) for cell in row
            ]
            table.add_row(*cells)
        output.print(table)


ENV_OPTION_HELP = (
    "Environment to interact with. If unspecified, defers to `MODAL_ENVIRONMENT`, "
    "your active local profile, or your workspace default, in that order."
)


def env_option(func):
    """Reusable Click decorator for the --env / -e option."""
    return click.option("-e", "--env", default=None, help=ENV_OPTION_HELP)(func)


def yes_option(func):
    """Reusable Click decorator for the --yes / -y option."""
    return click.option("-y", "--yes", is_flag=True, default=False, help="Run without pausing for confirmation.")(func)


def confirm_or_suggest_yes(msg: str) -> None:
    """Prompt for confirmation, or abort with a hint to use --yes if stdin is not a TTY."""
    if not sys.stdin.isatty():
        click.echo(f"{msg} [y/N]: ")
        raise SystemExit("Aborted: no interactive terminal detected. Rerun with --yes (-y) to skip confirmation.")
    click.confirm(msg, default=False, abort=True)
