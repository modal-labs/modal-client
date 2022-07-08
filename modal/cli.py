import asyncio
import getpass
import inspect
import sys
import traceback
from typing import Optional

import typer
from rich.console import Console

from modal_proto import api_pb2
from modal_utils.package_utils import import_stub_by_ref

from .client import Client
from .config import (
    _config_envs,
    _config_set_active_env,
    _env,
    _store_user_config,
    config,
    user_config_path,
)

app = typer.Typer()

token_app = typer.Typer()
app.add_typer(token_app, name="token", help="Manage tokens")
config_app = typer.Typer()
app.add_typer(
    config_app,
    name="config",
    help="""
    Manage client configuration

    Refer to https://modal.com/docs/reference/modal.config for a full explanation of what these
    options mean, and how to set them.
    """,
)
app_app = typer.Typer()
app.add_typer(app_app, name="app", help="Manage Modal applications")
env_app = typer.Typer()
app.add_typer(env_app, name="env", help="Manage currently activated Modal environment")


@token_app.command(
    help="Set account credentials for connecting to Modal. If not provided with the command, you will be prompted to enter your credentials."
)
def set(
    token_id: Optional[str] = typer.Option(None, help="Token ID"),
    token_secret: Optional[str] = typer.Option(None, help="Token secret"),
    env: Optional[str] = typer.Option(
        None,
        help="Modal environment to set credentials for. You can switch the currently active Modal environment with the `modal env` command. If unspecified, uses `default` environment.  ",
    ),
    no_verify: bool = False,
):
    if token_id is None:
        token_id = getpass.getpass("Token ID:")
    if token_secret is None:
        token_secret = getpass.getpass("Token secret:")

    if not no_verify:
        server_url = config.get("server_url", env=env)
        print(f"Verifying token against {server_url}...")
        client = Client(server_url, api_pb2.CLIENT_TYPE_CLIENT, (token_id, token_secret))
        client.verify()
        print("Token verified successfully")

    _store_user_config({"token_id": token_id, "token_secret": token_secret}, env=env)
    print(f"Token written to {user_config_path}")


@config_app.command(help="[Debug command] show currently applied configuration values")
def show():
    # This is just a test command
    print(config)


@config_app.command()
def main():
    app()


@env_app.command(help="Change the currently active Modal environment.")
def activate(env: str = typer.Argument(..., help="Modal environment to activate")):
    _config_set_active_env(env)


@env_app.command(help="Print the active Modal environments.")
def current():
    print(_env)


@env_app.command(help="List all Modal environments that are defined.")
def list():
    for env in _config_envs():
        print(f"{env} [active]" if _env == env else env)


@app_app.command()
def app_main():
    app_app()


@app_app.command("deploy", help="Deploy a Modal stub as an application.")
def deploy(
    stub_ref: str = typer.Argument(..., help="Path to a Modal stub."),
    name: str = typer.Option(None, help="Name of the deployment."),
):
    try:
        stub = import_stub_by_ref(stub_ref)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    if name is None:
        name = stub.name

    res = stub.deploy(name=name)
    if inspect.iscoroutine(res):
        app_id = asyncio.run(res)
    else:
        app_id = res

    console = Console()
    console.print(f"\nView Deployment: [magenta]https://modal.com/deployments/{app_id}[/magenta]")


if __name__ == "__main__":
    app()
