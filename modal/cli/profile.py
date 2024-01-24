# Copyright Modal Labs 2022

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from modal.config import Config, _lookup_workspace, _profile, config_profiles, config_set_active_profile
from modal.exception import AuthError
from modal_utils.async_utils import synchronizer

profile_cli = typer.Typer(name="profile", help="Switch between Modal profiles.", no_args_is_help=True)


@profile_cli.command(help="Change the active Modal profile.")
def activate(profile: str = typer.Argument(..., help="Modal profile to activate.")):
    config_set_active_profile(profile)


@profile_cli.command(help="Print the active Modal profile.")
def current():
    typer.echo(_profile)


@profile_cli.command(name="list", help="Show all Modal profiles that are defined.")
@synchronizer.create_blocking
async def list(json: Optional[bool] = False):
    config = Config()
    column_names = ["", "Profile", "Workspace"]
    rows = []
    for profile in config_profiles():
        try:
            resp = await _lookup_workspace(config, profile)
            workspace = resp.workspace_name
        except AuthError:
            workspace = "Unknown (authentication failed)"
        except Exception:
            workspace = "Unknown (profile misconfigured)"
        rows.append(["*" if profile == _profile else "", profile, workspace])

    console = Console()
    table = Table(*column_names)
    for row in rows:
        table.add_row(*row, style="green" if row[0] == "*" else "dim")
    console.print(table)
