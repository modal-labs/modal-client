# Copyright Modal Labs 2022
import typer

from modal.config import _profile, config_profiles, config_set_active_profile

profile_cli = typer.Typer(name="profile", help="Set the current environment.", no_args_is_help=True)


@profile_cli.command(help="Change the active Modal environment.")
def activate(profile: str = typer.Argument(..., help="Modal environment to activate.")):
    config_set_active_profile(profile)


@profile_cli.command(help="Print the active Modal environment.")
def current():
    print(_profile)


@profile_cli.command(help="List all Modal environments that are defined.")
def list():
    for env in config_profiles():
        print(f"{env} [active]" if _profile == env else env)
