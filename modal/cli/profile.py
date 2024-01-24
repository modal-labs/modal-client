# Copyright Modal Labs 2022

from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from modal.config import Config, _lookup_workspace, _profile, config_profiles, config_set_active_profile
from modal.exception import AuthError
from modal_utils.async_utils import synchronizer

profile_cli = typer.Typer(name="profile", help="Switch between Modal profiles.", no_args_is_help=True)


@profile_cli.command(help="Change the active Modal profile.")
def activate(profile: str = typer.Argument(..., help="Modal profile to activate.")):
    config_set_active_profile(profile)


@profile_cli.command(help="Print the currently active Modal profile.")
def current():
    typer.echo(_profile)


@profile_cli.command(name="list", help="Show all Modal profiles and highlight the active one.")
@synchronizer.create_blocking
async def list(json: Optional[bool] = False):
    config = Config()
    column_names = [" ", "Profile", "Workspace"]
    rows = []
    for profile in config_profiles():
        try:
            resp = await _lookup_workspace(config, profile)
            workspace = resp.workspace_name
        except AuthError:
            workspace = "Unknown (authentication failed)"
        except Exception:
            workspace = "Unknown (profile misconfigured)"
        active = profile == _profile
        content = ["*" if active else "", profile, workspace]
        style = "bold green" if active else "dim"
        rows.append((content, style))

    console = Console()

    if json:
        json_data = []
        for content, _ in rows:
            json_data.append({"name": content[1], "active": content[0] == "*", "workspace": content[2]})
        console.print(JSON.from_data(json_data))
    else:
        table = Table(*column_names)
        for (content, style) in rows:
            table.add_row(*content, style=style)
        console.print(table)
