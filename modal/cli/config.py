import typer

from modal.config import config

config_cli = typer.Typer(no_args_is_help=True)


@config_cli.command(help="[Debug command] show currently applied configuration values")
def show():
    # This is just a test command
    print(config)
