import asyncio
import getpass
import inspect
import os
import sys
import traceback

import typer

from modal_proto import api_pb2
from modal_utils.package_utils import import_app_by_ref

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
app.add_typer(token_app, name="token")
config_app = typer.Typer()
app.add_typer(config_app, name="config")
app_app = typer.Typer()
app.add_typer(app_app, name="app")
env_app = typer.Typer()
app.add_typer(env_app, name="env")


class Symbols:
    PARTY_POPPER = "\U0001F389"


@token_app.command()
def set(token_id: str = "-", token_secret: str = "-", env: str = None, no_verify: bool = False):
    if token_id == "-":
        token_id = getpass.getpass("Token ID:")
    if token_secret == "-":
        token_secret = getpass.getpass("Token secret:")

    if not no_verify:
        server_url = config.get("server_url", env=env)
        print(f"Verifying token against {server_url}...")
        client = Client(server_url, api_pb2.CLIENT_TYPE_CLIENT, (token_id, token_secret))
        client.verify()
        print("Token verified successfully")

    _store_user_config({"token_id": token_id, "token_secret": token_secret}, env=env)
    print(f"Token written to {user_config_path} {Symbols.PARTY_POPPER}")


@config_app.command()
def show():
    # This is just a test command
    print(config)


@config_app.command()
def main():
    app()


@env_app.command()
def activate(env: str):
    _config_set_active_env(env)


@env_app.command()
def current():
    print(_env)


@env_app.command()
def list():
    for env in _config_envs():
        print(f"{env} [active]" if _env == env else env)


@app_app.command()
def app_main():
    app_app()


@app_app.command("deploy")
def deploy(app_ref: str, name: str = None):
    try:
        app = import_app_by_ref(app_ref)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    if name is None and app.provided_name() is None:
        # replace os.sep for convenience
        name = app_ref.replace(os.sep, ".")

    res = app.deploy(name=name)
    if inspect.iscoroutine(res):
        app_id = asyncio.run(res)
    else:
        app_id = res

    print(f"App deployed! {Symbols.PARTY_POPPER}")
    print(f"App ID: {app_id}")  # TODO: print URL instead


if __name__ == "__main__":
    app()
