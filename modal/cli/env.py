# Copyright Modal Labs 2022
import datetime

import typer

from modal.cli import profile as profile_cli
from modal.exception import deprecation_warning


def warn_env_deprecated():
    pass


DEPRECATION_PREFIX = "[Deprecated, use `modal profile` instead] "


def print_env_deprecated_warning():
    deprecation_warning(
        datetime.date(2023, 5, 24), "`modal env` will soon be deprecated. Use `modal profile` instead", pending=True
    )


env_cli = typer.Typer(name="env", help=f"{DEPRECATION_PREFIX}Set the current environment.", no_args_is_help=True)


@env_cli.command(help=f"{DEPRECATION_PREFIX}Change the active Modal environment.")
def activate(env: str = typer.Argument(..., help="Modal environment to activate.")):
    print_env_deprecated_warning()
    profile_cli.activate(env)


@env_cli.command(help=f"{DEPRECATION_PREFIX}Print the active Modal environment.")
def current():
    print_env_deprecated_warning()
    profile_cli.current()


@env_cli.command(help=f"{DEPRECATION_PREFIX}List all Modal environments that are defined.")
def list():
    print_env_deprecated_warning()
    profile_cli.list()
