# Copyright Modal Labs 2022

import typer

from modal.config import _profile, config_profiles, config_set_active_profile, _store_user_config

profile_cli = typer.Typer(name="profile", help="Set the active Modal profile.", no_args_is_help=True)


@profile_cli.command(help="Change the active Modal profile.")
def activate(profile: str = typer.Argument(..., help="Modal profile to activate.")):
    config_set_active_profile(profile)


@profile_cli.command(help="Print the active Modal profile.")
def current():
    typer.echo(_profile)


@profile_cli.command(help="List all Modal profiles that are defined.")
def list():
    for env in config_profiles():
        typer.echo(f"{env} [active]" if _profile == env else env)


SET_DEFAULT_ENV_HELP = """Set the default Modal environment for profile

The default environment of a profile is used when no --env flag is passed to `modal run`, `modal deploy` etc.

If no default environment is set, and there exists multiple environments in a workspace, an error will be raised
when running a command that requires an environment.
"""


@profile_cli.command(help=SET_DEFAULT_ENV_HELP)
def set_default_environment(name: str):
    _store_user_config({"environment": name})
    typer.echo("Done")
