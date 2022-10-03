import os
import platform
import subprocess
from tempfile import NamedTemporaryFile
from typing import List

import click
import typer
from google.protobuf import empty_pb2
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

import modal
from modal.cli.utils import timestamp_to_local
from modal.client import Client, _Client
from modal_utils.async_utils import synchronizer

secret_cli = typer.Typer(name="secret", help="Manage secrets.", no_args_is_help=True)


@secret_cli.command("list", help="List your published secrets.")
@synchronizer
async def list():
    client = await _Client.from_env()
    response = await client.stub.SecretList(empty_pb2.Empty())
    table = Table()
    table.add_column("Name")
    table.add_column("Created at", justify="right")
    table.add_column("Last used at", justify="right")

    for item in response.items:
        table.add_row(
            item.label,
            timestamp_to_local(item.created_at),
            timestamp_to_local(item.last_used_at) if item.last_used_at else "-",
        )

    console = Console()
    console.print(table)


@secret_cli.command("create", help="Create a new secret, or overwrite an existing one.")
def create(secret_name, keyvalues: List[str]):
    env_dict = {}

    for arg in keyvalues:
        if "=" in arg:
            key, value = arg.split("=")
            if value == "-":
                value = get_text_from_editor(key)
            env_dict[key] = value
        else:
            raise click.UsageError(
                """Each key should be of the form <KEY>=VALUE. To enter secrets using your $EDITOR, use `<KEY>=-`.

E.g.

modal secret create my-credentials username=john password=-
"""
            )

    if not env_dict:
        raise click.UsageError("You need to specify at least one key for your secret")

    stub = modal.Stub()
    stub["secret"] = modal.Secret(
        env_dict=env_dict,
    )
    stub.deploy(secret_name, client=Client.from_env(), show_progress=False)
    console = Console()

    env_var_code = "\n    ".join(f'os.getenv("{name}")' for name in env_dict.keys()) if env_dict else "..."

    example_code = f"""
@stub.function(secret=modal.ref("{secret_name}"))
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
                "Something went wrong with the external editor. Try again, or use '--' as the value to pass input through stdin instead"
            )

        bufferfile.seek(0)
        return bufferfile.read()
