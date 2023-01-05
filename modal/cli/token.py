# Copyright Modal Labs 2022
import getpass
from typing import Optional
import webbrowser

import rich
from rich.console import Console
import typer

from modal.client import Client
from modal.config import _store_user_config, config, user_config_path

token_cli = typer.Typer(name="token", help="Manage tokens.", no_args_is_help=True)

env_option = typer.Option(
    None,
    help="Modal environment to set credentials for. You can switch the currently active Modal environment with the `modal env` command. If unspecified, uses `default` environment.",
)


@token_cli.command(
    help="Set account credentials for connecting to Modal. If not provided with the command, you will be prompted to enter your credentials."
)
def set(
    token_id: Optional[str] = typer.Option(None, help="Account token ID."),
    token_secret: Optional[str] = typer.Option(None, help="Account token secret."),
    env: Optional[str] = env_option,
    no_verify: bool = False,
):
    if token_id is None:
        token_id = getpass.getpass("Token ID:")
    if token_secret is None:
        token_secret = getpass.getpass("Token secret:")

    if not no_verify:
        server_url = config.get("server_url", env=env)
        rich.print(f"Verifying token against [blue]{server_url}[/blue]")
        Client.verify(server_url, (token_id, token_secret))
        rich.print("[green]Token verified successfully[/green]")

    _store_user_config({"token_id": token_id, "token_secret": token_secret}, env=env)
    rich.print(f"Token written to {user_config_path}")


@token_cli.command(help="Creates a new token by using an authenticated web session.")
def new(env: Optional[str] = env_option, no_verify: bool = False):
    server_url = config.get("server_url", env=env)

    with Client.unauthenticated_client(env, server_url) as client:
        token_flow_id, web_url = client.start_token_flow()
        console = Console()
        with console.status("Waiting for authentication in the web browser...", spinner="dots"):
            # Open the web url in the browser
            link_text = f"[link={web_url}]{web_url}[/link]"
            console.print(f"Launching {link_text} in your browser window")
            if webbrowser.open_new_tab(web_url):
                console.print("If this is not showing up, please copy the URL into your web browser manually")
            else:
                console.print(
                    "[red]Was not able to launch web browser[/red]"
                    " - please go to the URL manually and complete the flow"
                )
        token_id, token_secret = client.finish_token_flow(token_flow_id)
        console.print("[green]Success![/green]")

    if not no_verify:
        rich.print(f"Verifying token against [blue]{server_url}[/blue]")
        Client.verify(server_url, (token_id, token_secret))
        rich.print("[green]Token verified successfully[/green]")

    _store_user_config({"token_id": token_id, "token_secret": token_secret}, env=env)
    rich.print(f"Token written to {user_config_path}")
