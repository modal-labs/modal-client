# Copyright Modal Labs 2022
import asyncio
import io
import re
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from csv import writer as csv_writer
from datetime import date, datetime, timezone
from json import dumps
from typing import Literal

import click
from click.exceptions import UsageError
from google.protobuf.timestamp_pb2 import Timestamp
from rich.box import SIMPLE_HEAD, Box
from rich.table import Column, Table
from rich.text import Text

from modal_proto import api_pb2

from .._logs import LogsFilters, fetch_logs, tail_logs
from .._output.pty import _build_log_prefix, get_app_logs_loop
from .._traceback import print_server_warnings
from .._utils.async_utils import synchronizer
from .._utils.grpc_utils import is_class_function_lookup_error
from ..client import _Client
from ..exception import InvalidError, NotFoundError
from ..output import OutputManager

HEADER_ONLY = SIMPLE_HEAD


def grouped_utc_timestamp(timestamp: Timestamp, previous_date: date | None) -> tuple[str, date]:
    timestamp_datetime = timestamp.ToDatetime(tzinfo=timezone.utc)
    current_date = timestamp_datetime.date()
    date_prefix = f"{current_date.isoformat()} " if current_date != previous_date else ""
    time_label = timestamp_datetime.strftime("%H:%M:%S")
    return f"{date_prefix}{time_label}", current_date


async def _stream_app_logs(
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
                parametrized_function_id=filters.parametrized_function_id,
                function_call_id=filters.function_call_id,
                search_text=filters.search_text,
            )
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass


# Blocking variant just used by the `modal deploy` handler,
# which must run on synchronizer thread.
stream_app_logs = synchronizer.create_blocking(_stream_app_logs)


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


async def _tail_app_logs(
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


async def _fetch_app_logs(
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


def humanize_filesize(value: int) -> str:
    if value < 0:
        raise ValueError("value should be >= 0")
    base = 1024
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if size < base:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= base
    return f"{size:.1f} ZiB"


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
    table_box: Box | None = None,
    header_style: str | None = None,
    border_style: str | None = None,
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
        table = (
            Table(*columns, title=title, header_style=header_style, border_style=border_style)
            if table_box is None
            else Table(
                *columns,
                title=title,
                box=table_box,
                header_style=header_style,
                border_style=border_style,
            )
        )
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


def _is_function_id(ref: str) -> bool:
    return bool(re.match(r"^fu-[0-9a-zA-Z]+$", ref))


async def _resolve_function_id(
    client: _Client,
    function_identifier: str,
    environment_name: str,
    *,
    object_type: Literal["Function", "Server"] = "Function",
    command: Literal["info", "calls", "logs", "requests", "stats", "variants"],
) -> tuple[str, api_pb2.FunctionHandleMetadata, api_pb2.FunctionData]:
    identifier_label = object_type.upper()
    usage = (
        f"{identifier_label} must be a Function ID (fu-…) "
        f"or a deployed {object_type} name (APP_NAME/{identifier_label}_NAME)."
    )
    if function_identifier == "":
        raise UsageError(usage)

    if "/" in function_identifier:
        app_name, function_name = function_identifier.split("/", 1)

        if not (app_name and function_name):
            raise UsageError(usage)

        try:
            response = await client._stub.FunctionGet(
                api_pb2.FunctionGetRequest(
                    app_name=app_name,
                    object_tag=function_name,
                    environment_name=environment_name,
                )
            )
        except NotFoundError as exc:
            if object_type == "Function" and is_class_function_lookup_error(exc):
                raise NotFoundError(
                    f"'{function_identifier}' is a modal.Cls. Use\n modal function {command} '{function_identifier}.*'"
                ) from None
            raise

        print_server_warnings(response.server_warnings)
        function_id = response.function_id
        function = response.function
        metadata = response.handle_metadata
    elif _is_function_id(function_identifier):
        get_by_id_response = await client._stub.FunctionGetById(
            api_pb2.FunctionGetByIdRequest(function_id=function_identifier)
        )
        function_id = function_identifier
        function = get_by_id_response.function
        metadata = get_by_id_response.handle_metadata
    else:
        raise UsageError(usage)

    if function.is_server != (object_type == "Server"):
        actual_type = "Server" if function.is_server else "Function"
        if function.is_server and command == "variants":
            raise UsageError(f"'{function_identifier}' is a Server.")
        raise UsageError(f"'{function_identifier}' is a {actual_type}. Use `modal {actual_type.lower()} {command}`.")

    return function_id, metadata, function
