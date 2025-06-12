# Copyright Modal Labs 2022
import json
import os
import platform
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import click
import typer
from rich.console import Console
from rich.syntax import Syntax
from typer import Argument

from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal._utils.time_utils import timestamp_to_local
from modal.cli.utils import ENV_OPTION, YES_OPTION, display_table
from modal.client import _Client
from modal.environments import ensure_env
from modal.secret import _Secret
from modal_proto import api_pb2

secret_cli = typer.Typer(name="secret", help="Manage secrets.", no_args_is_help=True)


@secret_cli.command("list", help="List your published secrets.")
@synchronizer.create_blocking
async def list_(env: Optional[str] = ENV_OPTION, json: bool = False):
    env = ensure_env(env)
    client = await _Client.from_env()
    response = await retry_transient_errors(client.stub.SecretList, api_pb2.SecretListRequest(environment_name=env))
    column_names = ["Name", "Created at", "Last used at"]
    rows = []

    for item in response.items:
        rows.append(
            [
                item.label,
                timestamp_to_local(item.created_at, json),
                timestamp_to_local(item.last_used_at, json) if item.last_used_at else "-",
            ]
        )

    env_part = f" in environment '{env}'" if env else ""
    display_table(column_names, rows, json, title=f"Secrets{env_part}")


@secret_cli.command("create", help="Create a new secret.")
@synchronizer.create_blocking
async def create(
    secret_name: str,
    keyvalues: Optional[list[str]] = typer.Argument(default=None, help="Space-separated KEY=VALUE items."),
    env: Optional[str] = ENV_OPTION,
    from_dotenv: Optional[Path] = typer.Option(default=None, help="Path to a .env file to load secrets from."),
    from_json: Optional[Path] = typer.Option(default=None, help="Path to a JSON file to load secrets from."),
    force: bool = typer.Option(False, "--force", help="Overwrite the secret if it already exists."),
):
    env = ensure_env(env)
    env_dict = {}

    for arg in keyvalues or []:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if value == "-":
                value = get_text_from_editor(key)
            env_dict[key] = value
        else:
            raise click.UsageError(
                """Each item should be of the form <KEY>=VALUE. To enter secrets using your $EDITOR, use `<KEY>=-`. To
enter secrets from environment variables, use `<KEY>="$ENV_VAR"`.

E.g.

modal secret create my-credentials username=john password=-
modal secret create my-credentials username=john password="$PASSWORD"
"""
            )

    if from_dotenv:
        if not from_dotenv.is_file():
            raise click.UsageError(f"Could not read .env file at {from_dotenv}")

        try:
            from dotenv import dotenv_values
        except ImportError:
            raise ImportError(
                "Need the `python-dotenv` package installed. You can install it by running `pip install python-dotenv`."
            )

        try:
            env_dict.update(dotenv_values(from_dotenv))
        except Exception as e:
            raise click.UsageError(f"Could not parse .env file at {from_dotenv}: {e}")

    if from_json:
        if not from_json.is_file():
            raise click.UsageError(f"Could not read JSON file at {from_json}")

        try:
            with from_json.open("r") as f:
                env_dict.update(json.load(f))
        except Exception as e:
            raise click.UsageError(f"Could not parse JSON file at {from_json}: {e}")

    if not env_dict:
        raise click.UsageError("You need to specify at least one key for your secret")

    for k, v in env_dict.items():
        if not isinstance(k, str) or not k:
            raise click.UsageError(f"Invalid key: '{k}'")
        if not isinstance(v, str):
            raise click.UsageError(f"Non-string value for secret '{k}'")

    # Create secret
    await _Secret.create_deployed(secret_name, env_dict, overwrite=force)

    # Print code sample
    console = Console()
    env_var_code = "\n    ".join(f'os.getenv("{name}")' for name in env_dict.keys()) if env_dict else "..."
    example_code = f"""
@app.function(secrets=[modal.Secret.from_name("{secret_name}")])
def some_function():
    {env_var_code}
"""
    plural_s = "s" if len(env_dict) > 1 else ""
    console.print(
        f"""Created a new secret '{secret_name}' with the key{plural_s} {", ".join(repr(k) for k in env_dict.keys())}"""
    )
    console.print("\nUse it in your Modal app:\n")
    console.print(Syntax(example_code, "python"))


@secret_cli.command("delete", help="Delete a named secret.")
@synchronizer.create_blocking
async def delete(
    secret_name: str = Argument(help="Name of the modal.Secret to be deleted. Case sensitive"),
    yes: bool = YES_OPTION,
    env: Optional[str] = ENV_OPTION,
):
    """TODO"""
    env = ensure_env(env)
    secret = await _Secret.from_name(secret_name, environment_name=env).hydrate()
    if not yes:
        typer.confirm(
            f"Are you sure you want to irrevocably delete the modal.Secret '{secret_name}'?",
            default=False,
            abort=True,
        )
    client = await _Client.from_env()

    # TODO: replace with API on `modal.Secret` when we add it
    await client.stub.SecretDelete(api_pb2.SecretDeleteRequest(secret_id=secret.object_id))


def get_text_from_editor(key) -> str:
    with NamedTemporaryFile("w+", prefix="secret_buffer", suffix=".txt") as bufferfile:
        if platform.system() != "Windows":
            editor = os.getenv("EDITOR", "vi")
            input(f"Pressing enter will open an external editor ({editor}) for editing '{key}'...")
            status_code = subprocess.call([editor, bufferfile.name])
        else:
            # not tested, but according to https://stackoverflow.com/questions/1442841/lauch-default-editor-like-webbrowser-module
            # this should open an editor on Windows...
            input("Pressing enter will open an external editor to allow you to edit the secret value...")
            status_code = os.system(bufferfile.name)

        if status_code != 0:
            raise ValueError(
                "Something went wrong with the external editor. "
                "Try again, or use '--' as the value to pass input through stdin instead"
            )

        bufferfile.seek(0)
        return bufferfile.read()
