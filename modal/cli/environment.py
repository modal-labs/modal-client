# Copyright Modal Labs 2023
from typing import Optional

import typer
from click import UsageError
from typing_extensions import Annotated

from modal import environments
from modal.cli.utils import display_table
from modal.config import config

ENVIRONMENT_HELP_TEXT = """Create and interact with Environments

Environments are sub-divisons of workspaces, allowing you to deploy the same app
in different namespaces. Each environment has their own set of Secrets and any
lookups performed from an app in an environment will by default look for entities
in the same environment.

Typical use cases for environments include having one for development and one for
production, to prevent overwriting production apps when developing new features
while still being able to deploy changes to a live environment.
"""

environment_cli = typer.Typer(name="environment", help=ENVIRONMENT_HELP_TEXT, no_args_is_help=True)


class RenderableBool:
    def __init__(self, value: bool):
        self.value = value

    def __rich__(self):
        return repr(self.value)


@environment_cli.command(name="list", help="List all environments in the current workspace")
def list(json: Optional[bool] = False):
    envs = environments.list_environments()
    table_data = []
    for item in envs:
        is_active = item.name == config.get("environment")
        row = [item.name, item.webhook_suffix, is_active if json else RenderableBool(is_active)]
        table_data.append(row)
    display_table(["name", "web suffix", "active"], table_data, json=json)


ENVIRONMENT_CREATE_HELP = """Create a new environment in the current workspace"""


@environment_cli.command(name="create", help=ENVIRONMENT_CREATE_HELP)
def create(name: Annotated[str, typer.Argument(help="Name of the new environment. Must be unique. Case sensitive")]):
    environments.create_environment(name)
    typer.echo(f"Environment created: {name}")


ENVIRONMENT_DELETE_HELP = """Delete an environment in the current workspace

Deletes all apps in the selected environment and deletes the environment irrevocably.
"""


@environment_cli.command(name="delete", help=ENVIRONMENT_DELETE_HELP)
def delete(
    name: str = typer.Argument(help="Name of the environment to be deleted. Case sensitive"),
    confirm: bool = typer.Option(default=False, help="Set this flag to delete without prompting for confirmation"),
):
    if not confirm:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the environment '{name}' and all its associated apps and secrets?",
            default=False,
            abort=True,
        )

    environments.delete_environment(name)
    typer.echo(f"Environment deleted: {name}")


ENVIRONMENT_UPDATE_HELP = """Update the name or web suffix of an environment"""


@environment_cli.command(name="update", help=ENVIRONMENT_UPDATE_HELP)
def update(
    current_name: str,
    set_name: Optional[str] = typer.Option(default=None, help="New name of the environment"),
    set_web_suffix: Optional[str] = typer.Option(
        default=None, help="New web suffix of environment (empty string is no suffix)"
    ),
):
    if set_name is None and set_web_suffix is None:
        raise UsageError("You need to at least one new property (using --set-name or --set-web-suffix")

    environments.update_environment(current_name, new_name=set_name, new_web_suffix=set_web_suffix)
    typer.echo("Environment updated")
