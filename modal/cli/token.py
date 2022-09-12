import getpass
from typing import Optional

import typer

from modal.client import Client
from modal.config import _store_user_config, config, user_config_path
from modal_proto import api_pb2

token_cli = typer.Typer(name="token", help="Manage tokens.", no_args_is_help=True)


@token_cli.command(
    help="Set account credentials for connecting to Modal. If not provided with the command, you will be prompted to enter your credentials."
)
def set(
    token_id: Optional[str] = typer.Option(None, help="Account token ID."),
    token_secret: Optional[str] = typer.Option(None, help="Account token secret."),
    env: Optional[str] = typer.Option(
        None,
        help="Modal environment to set credentials for. You can switch the currently active Modal environment with the `modal env` command. If unspecified, uses `default` environment.",
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
