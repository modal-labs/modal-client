# Copyright Modal Labs 2022

import asyncio
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
    profiles = config_profiles()
    responses = await asyncio.gather(
        *(_lookup_workspace(config, profile) for profile in profiles), return_exceptions=True
    )

    rows = []
    for profile, resp in zip(profiles, responses):
        active = profile == _profile
        if isinstance(resp, AuthError):
            workspace = "Unknown (authentication failure)"
        elif isinstance(resp, Exception):
            # Catch-all for other exceptions, like incorrect server url
            workspace = "Unknown (profile misconfigured)"
        else:
            workspace = resp.workspace_name
        content = ["•" if active else "", profile, workspace]
        rows.append((active, content))

    console = Console()
    if json:
        json_data = []
        for active, content in rows:
            json_data.append({"name": content[1], "workspace": content[2], "active": active})
        console.print(JSON.from_data(json_data))
    else:
        table = Table(" ", "Profile", "Workspace")
        for active, content in rows:
            table.add_row(*content, style="bold green" if active else "dim")
        console.print(table)
