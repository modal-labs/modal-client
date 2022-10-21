# Copyright Modal Labs 2022
import typer

from .app import app_cli
from .config import config_cli
from .env import env_cli
from .secret import secret_cli
from .token import token_cli
from .volume import volume_cli


def version_callback(value: bool):
    if value:
        from modal_version import __version__

        typer.echo(f"modal client version: {__version__}")
        raise typer.Exit()


entrypoint_cli = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
    help="""
    Modal is the fastest way to run code in the cloud.

    See the website at https://modal.com/ for documentation and more information
    about running code on Modal.
    """,
)


@entrypoint_cli.callback()
def root(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", callback=version_callback),
):
    pass


entrypoint_cli.add_typer(app_cli)
entrypoint_cli.add_typer(config_cli)
entrypoint_cli.add_typer(env_cli)
entrypoint_cli.add_typer(secret_cli)
entrypoint_cli.add_typer(token_cli)
entrypoint_cli.add_typer(volume_cli)

if __name__ == "__main__":
    # this module is only called from tests, otherwise the parent package __init__.py is used as the entrypoint
    entrypoint_cli()
