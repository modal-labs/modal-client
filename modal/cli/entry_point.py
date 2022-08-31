import typer

from modal.cli.app import app_cli
from modal.cli.config import config_cli
from modal.cli.env import env_cli
from modal.cli.token import token_cli
from modal.cli.volume import volume_cli

from .secret import secret_cli

entrypoint_cli = typer.Typer(no_args_is_help=True)

entrypoint_cli.add_typer(token_cli, name="token", help="Manage tokens")

entrypoint_cli.add_typer(
    config_cli,
    name="config",
    help="""
    Manage client configuration

    Refer to https://modal.com/docs/reference/modal.config for a full explanation of what these
    options mean, and how to set them.
    """,
)

entrypoint_cli.add_typer(app_cli, name="app", help="Manage Modal applications")

entrypoint_cli.add_typer(env_cli, name="env", help="Manage currently activated Modal environment")

entrypoint_cli.add_typer(secret_cli, name="secret", help="Manage Modal Secrets")

entrypoint_cli.add_typer(volume_cli, name="volume")

if __name__ == "__main__":
    # this module is only called from tests, otherwise the parent package __init__.py is used as the entrypoint
    entrypoint_cli()
