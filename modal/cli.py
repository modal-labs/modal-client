import getpass

import typer

from ._client import Client
from .config import config, store_user_config, user_config_path
from .proto import api_pb2

app = typer.Typer()

token_app = typer.Typer()
app.add_typer(token_app, name="token")
config_app = typer.Typer()
app.add_typer(config_app, name="config")


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
        client = Client(server_url, api_pb2.ClientType.CLIENT, (token_id, token_secret))
        client.verify()
        print(f"Token verified successfully")

    store_user_config({"token_id": token_id, "token_secret": token_secret}, env=env)
    print(f"Token written to {user_config_path} {Symbols.PARTY_POPPER}")


@config_app.command()
def show():
    # This is just a test command
    print(config)


def main():
    app()


if __name__ == "__main__":
    app()
