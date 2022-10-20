# Copyright Modal Labs 2022
import typer

from modal.config import config

config_cli = typer.Typer(
    name="config",
    help="""
    Manage client configuration.

    Refer to https://modal.com/docs/reference/modal.config for a full explanation
    of what these options mean, and how to set them.
    """,
    no_args_is_help=True,
)


@config_cli.command(help="Show current configuration values (debug command).")
def show():
    # This is just a test command
    print(config)
