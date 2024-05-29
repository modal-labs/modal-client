# Copyright Modal Labs 2022
import os
import platform
import subprocess
from tempfile import NamedTemporaryFile
from typing import List, Optional

import click
import typer
from rich.console import Console
from rich.syntax import Syntax

from modal._utils.async_utils import synchronizer
from modal._utils.grpc_utils import retry_transient_errors
from modal.cli.utils import ENV_OPTION, display_table, timestamp_to_local
from modal.client import _Client
from modal.environments import ensure_env
from modal.secret import _Secret
from modal_proto import api_pb2

secret_cli = typer.Typer(name="secret", help="Manage secrets.", no_args_is_help=True)


@secret_cli.command("list", help="List your published secrets.")
@synchronizer.create_blocking
async def list(env: Optional[str] = ENV_OPTION, json: Optional[bool] = False):
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


@secret_cli.command("create", help="Create a new secret. Use `--force` to overwrite an existing one.")
@synchronizer.create_blocking
async def create(
    secret_name,
    keyvalues: List[str] = typer.Argument(..., help="Space-separated KEY=VALUE items"),
    env: Optional[str] = ENV_OPTION,
    force: bool = typer.Option(False, "--force", help="Overwrite the secret if it already exists."),
):
    env = ensure_env(env)
    env_dict = {}
    for arg in keyvalues:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if value == "-":
                value = get_text_from_editor(key)
            env_dict[key] = value
        else:
            raise click.UsageError(
                """Each item should be of the form <KEY>=VALUE. To enter secrets using your $EDITOR, use `<KEY>=-`.

E.g.

modal secret create my-credentials username=john password=-
"""
            )

    if not env_dict:
        raise click.UsageError("You need to specify at least one key for your secret")

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
        f"""Created a new secret '{secret_name}' with the key{plural_s} {', '.join(repr(k) for k in env_dict.keys())}"""
    )
    console.print("\nUse it in to your Modal app using:\n")
    console.print(Syntax(example_code, "python"))


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
