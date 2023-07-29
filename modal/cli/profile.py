# Copyright Modal Labs 2022

from typing import Optional

import typer

from modal.cli.utils import display_selection
from modal.config import _profile, config_profiles, config_set_active_profile

profile_cli = typer.Typer(name="profile", help="Set the active Modal profile.", no_args_is_help=True)


@profile_cli.command(help="Change the active Modal profile.")
def activate(profile: str = typer.Argument(..., help="Modal profile to activate.")):
    config_set_active_profile(profile)


@profile_cli.command(help="Print the active Modal profile.")
def current():
    typer.echo(_profile)


@profile_cli.command(help="List all Modal profiles that are defined.")
def list(json: Optional[bool] = False):
    display_selection(config_profiles(), _profile, json)
