# Copyright Modal Labs 2022
import getpass
import webbrowser
from typing import Optional

import rich
import typer
from rich.console import Console

from modal.client import Client
from modal.config import _store_user_config, config, user_config_path

token_cli = typer.Typer(name="token", help="Manage tokens.", no_args_is_help=True)

profile_option = typer.Option(
    None,
    help="Modal profile to set credentials for. You can switch the currently active Modal profile with the `modal profile` command. If unspecified, uses `default` profile.",
)


@token_cli.command(
    help="Set account credentials for connecting to Modal. If not provided with the command, you will be prompted to enter your credentials."
)
def set(
    token_id: Optional[str] = typer.Option(None, help="Account token ID."),
    token_secret: Optional[str] = typer.Option(None, help="Account token secret."),
    profile: Optional[str] = profile_option,
    no_verify: bool = False,
):
    if token_id is None:
        token_id = getpass.getpass("Token ID:")
    if token_secret is None:
        token_secret = getpass.getpass("Token secret:")

    if not no_verify:
        server_url = config.get("server_url", profile=profile)
        rich.print(f"Verifying token against [blue]{server_url}[/blue]")
        Client.verify(server_url, (token_id, token_secret))
        rich.print("[green]Token verified successfully[/green]")

    _store_user_config({"token_id": token_id, "token_secret": token_secret}, profile=profile)
    rich.print(f"Token written to {user_config_path}")


@token_cli.command(help="Creates a new token by using an authenticated web session.")
def new(profile: Optional[str] = profile_option, no_verify: bool = False, source: Optional[str] = None):
    server_url = config.get("server_url", profile=profile)

    with Client.unauthenticated_client(server_url) as client:
        token_flow_id, web_url = client.start_token_flow(source)
        console = Console()
        with console.status("Waiting for authentication in the web browser...", spinner="dots"):
            # Open the web url in the browser
            console.print("Launching login page in your browser window...")
            if webbrowser.open_new_tab(web_url):
                console.print("If this is not showing up, please copy this URL into your web browser manually:")
            else:
                console.print(
                    "[red]Was not able to launch web browser[/red]"
                    " - please go to this URL manually and complete the flow:"
                )
            console.print(f"\n[link={web_url}]{web_url}[/link]\n")

        token_id, token_secret = client.finish_token_flow(token_flow_id)
        console.print("[green]Success![/green]")

    if not no_verify:
        rich.print(f"Verifying token against [blue]{server_url}[/blue]")
        Client.verify(server_url, (token_id, token_secret))
        rich.print("[green]Token verified successfully[/green]")

    _store_user_config({"token_id": token_id, "token_secret": token_secret}, profile=profile)
    rich.print(f"Token written to {user_config_path}")
