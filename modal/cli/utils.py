# Copyright Modal Labs 2022
from datetime import datetime
from typing import List, Union

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table
from rich.text import Text


def timestamp_to_local(ts: float) -> str:
    if ts > 0:
        locale_tz = datetime.now().astimezone().tzinfo
        dt = datetime.fromtimestamp(ts, tz=locale_tz)
        return dt.isoformat(sep=" ", timespec="seconds")
    else:
        return None


def _plain(text: Union[Text, str]) -> str:
    return text.plain if isinstance(text, Text) else text


def display_table(column_names: List[str], rows: List[List[Union[Text, str]]], json: bool, title: str = None):
    console = Console()
    if json:
        json_data = [{col: _plain(row[i]) for i, col in enumerate(column_names)} for row in rows]
        console.print(JSON.from_data(json_data))
    else:
        table = Table(*column_names, title=title)
        for row in rows:
            table.add_row(*row)
        console.print(table)


ENV_OPTION_HELP = """Environment to interact with.

If not specified, Modal will use the default environment of your current profile, or the `MODAL_ENVIRONMENT` variable.
Otherwise, raises an error if the workspace has multiple environments.
"""
ENV_OPTION = typer.Option(default=None, help=ENV_OPTION_HELP)
