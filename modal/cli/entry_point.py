# Copyright Modal Labs 2022
import typer

from modal.cli import run
from modal.cli.environment import environment_cli

from ._shared_volume import vol_cli as old_vol_cli
from .app import app_cli
from .config import config_cli
from .network_file_system import nfs_cli
from .profile import profile_cli
from .secret import secret_cli
from .token import token_cli
from .volume import vol_cli


def version_callback(value: bool):
    if value:
        from modal_version import __version__

        typer.echo(f"modal client version: {__version__}")
        raise typer.Exit()


entrypoint_cli_typer = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
    help="""
    Modal is the fastest way to run code in the cloud.

    See the website at https://modal.com/ for documentation and more information
    about running code on Modal.
    """,
)


@entrypoint_cli_typer.callback()
def modal(
    ctx: typer.Context,
    version: bool = typer.Option(None, "--version", callback=version_callback),
):
    pass


entrypoint_cli_typer.add_typer(app_cli)
entrypoint_cli_typer.add_typer(config_cli)
entrypoint_cli_typer.add_typer(environment_cli)
entrypoint_cli_typer.add_typer(old_vol_cli)
entrypoint_cli_typer.add_typer(nfs_cli)
entrypoint_cli_typer.add_typer(vol_cli)
entrypoint_cli_typer.add_typer(profile_cli)
entrypoint_cli_typer.add_typer(secret_cli)
entrypoint_cli_typer.add_typer(token_cli)

entrypoint_cli_typer.command("deploy", help="Deploy a Modal stub as an application.", no_args_is_help=True)(run.deploy)
entrypoint_cli_typer.command("serve", no_args_is_help=True)(run.serve)
entrypoint_cli_typer.command("shell", no_args_is_help=True)(run.shell)

entrypoint_cli = typer.main.get_command(entrypoint_cli_typer)
entrypoint_cli.add_command(run.run, name="run")  # type: ignore
entrypoint_cli.list_commands(None)  # type: ignore

if __name__ == "__main__":
    # this module is only called from tests, otherwise the parent package __init__.py is used as the entrypoint
    entrypoint_cli()
