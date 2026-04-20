# Copyright Modal Labs 2022
import subprocess
from typing import Optional

import click
from rich.rule import Rule

from modal._utils.async_utils import synchronizer
from modal.output import OutputManager

from . import run, shell as shell_module
from ._help import ModalCommand, ModalGroup
from .app import app_cli
from .billing import billing_cli
from .bootstrap import bootstrap
from .changelog import changelog
from .cluster import cluster_cli
from .config import config_cli
from .container import container_cli
from .dashboard import dashboard
from .dict import dict_cli
from .environment import environment_cli
from .launch import launch_cli
from .logo import print_logo
from .network_file_system import nfs_cli
from .profile import profile_cli
from .queues import queue_cli
from .secret import secret_cli
from .token import _new_token, token_cli
from .volume import volume_cli


@click.group(
    cls=ModalGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="""
    Modal is the fastest way to run code in the cloud.

    See the website at https://modal.com/ for documentation and more information
    about running code on Modal.
    """,
)
@click.option(
    "--version",
    is_flag=True,
    default=False,
    is_eager=True,
    expose_value=False,
    callback=lambda ctx, param, value: _version_callback(ctx, value),
)
def entrypoint_cli():
    pass


def _version_callback(ctx, value):
    if value:
        from modal_version import __version__

        click.echo(f"modal client version: {__version__}")
        ctx.exit(0)


def check_path():
    """Checks whether the `modal` executable is on the path and usable."""
    url = "https://modal.com/docs/guide/troubleshooting#command-not-found-errors"
    try:
        subprocess.run(["modal", "--help"], capture_output=True)
        # TODO(erikbern): check returncode?
        return
    except FileNotFoundError:
        text = (
            "[red]The `[white]modal[/white]` command was not found on your path!\n"
            "You may need to add it to your path or use `[white]python -m modal[/white]` as a workaround.[/red]\n"
        )
    except PermissionError:
        text = (
            "[red]The `[white]modal[/white]` command is not executable!\n"
            "You may need to give it permissions or use `[white]python -m modal[/white]` as a workaround.[/red]\n"
        )
    text += f"See more information here:\n\n[link={url}]{url}[/link]\n"
    output = OutputManager.get()
    output.print(text)
    output.print(Rule(style="white"))


@click.command("setup", cls=ModalCommand, help="Bootstrap Modal's configuration.")
@click.option("--profile", default=None)
@synchronizer.create_blocking
async def setup(profile: Optional[str] = None):
    check_path()
    print_logo()

    # Fetch a new token (same as `modal token new` but redirect to /home once finishes)
    await _new_token(profile=profile, next_url="/home")


entrypoint_cli.add_command(run.deploy, "deploy", panel="Commands")
entrypoint_cli.add_command(run.serve, "serve", panel="Commands")
entrypoint_cli.add_command(shell_module.shell, "shell", panel="Commands")
entrypoint_cli.add_command(run.run, "run", panel="Commands")
# launch is hidden as it's experimental and we're tracking towards removing it
entrypoint_cli.add_command(launch_cli, hidden=True)

entrypoint_cli.add_command(app_cli, panel="Deployments")
entrypoint_cli.add_command(container_cli, panel="Deployments")
# cluster is hidden while multi-node is in beta/experimental
entrypoint_cli.add_command(cluster_cli, panel="Deployments", hidden=True)

entrypoint_cli.add_command(dict_cli, panel="Storage")
entrypoint_cli.add_command(nfs_cli, panel="Storage")
entrypoint_cli.add_command(secret_cli, panel="Storage")
entrypoint_cli.add_command(queue_cli, panel="Storage")
entrypoint_cli.add_command(volume_cli, panel="Storage")

entrypoint_cli.add_command(setup, panel="Onboarding")
entrypoint_cli.add_command(bootstrap, panel="Onboarding")

entrypoint_cli.add_command(config_cli, panel="Configuration")
entrypoint_cli.add_command(environment_cli, panel="Configuration")
entrypoint_cli.add_command(profile_cli, panel="Configuration")
entrypoint_cli.add_command(token_cli, panel="Configuration")

entrypoint_cli.add_command(billing_cli, panel="Observability")
entrypoint_cli.add_command(changelog, panel="Observability")
entrypoint_cli.add_command(dashboard, panel="Observability")

if __name__ == "__main__":
    # this module is only called from tests, otherwise the parent package __main__.py is used as the entrypoint
    from modal.output import enable_output

    with enable_output():
        entrypoint_cli()
