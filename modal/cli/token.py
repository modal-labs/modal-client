# Copyright Modal Labs 2022
import getpass
import itertools
import os
import webbrowser
from typing import Optional

import rich
import typer
from rich.console import Console

from modal.client import Client
from modal.config import _store_user_config, config, user_config_path
from modal.token_flow import TokenFlow
from modal_proto import api_pb2

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


def _open_url(url: str) -> bool:
    """Opens url in web browser, making sure we use a modern one (not Lynx etc)"""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    try:
        browser = webbrowser.get()
        # zpresto defines `BROWSER=open` by default on macOS, which causes `webbrowser` to return `GenericBrowser`.
        if isinstance(browser, webbrowser.GenericBrowser) and browser.name != "open":
            return False
        else:
            return browser.open_new_tab(url)
    except webbrowser.Error:
        return False


def _new_token(
    profile: Optional[str] = None, no_verify: bool = False, source: Optional[str] = None, next_url: Optional[str] = None
):
    server_url = config.get("server_url", profile=profile)

    console = Console()

    result: Optional[api_pb2.TokenFlowWaitResponse] = None
    with Client.unauthenticated_client(server_url) as client:
        token_flow = TokenFlow(client)

        with token_flow.start(source, next_url) as (token_flow_id, web_url, code):
            with console.status("Waiting for authentication in the web browser", spinner="dots"):
                # Open the web url in the browser
                if _open_url(web_url):
                    console.print(
                        "The web browser should have opened for you to authenticate and get an API token.\n"
                        "If it didn't, please copy this URL into your web browser manually:\n"
                    )
                else:
                    console.print(
                        "[red]Was not able to launch web browser[/red]\n"
                        "Please go to this URL manually and complete the flow:\n"
                    )
                console.print(f"[link={web_url}]{web_url}[/link]\n")
                if code:
                    console.print(f"Enter this code: [yellow]{code}[/yellow]\n")

            with console.status("Waiting for token flow to complete...", spinner="dots") as status:
                for attempt in itertools.count():
                    result = token_flow.finish()
                    if result is not None:
                        break
                    status.update(f"Waiting for token flow to complete... (attempt {attempt+2})")

        console.print("[green]Web authentication finished successfully![/green]")

    assert result is not None

    if result.workspace_username:
        console.print(f"[green]Token is connected to the [white]{result.workspace_username}[/white] workspace.[/green]")

    if not no_verify:
        with console.status(f"Verifying token against [blue]{server_url}[/blue]", spinner="dots"):
            Client.verify(server_url, (result.token_id, result.token_secret))
            console.print("[green]Token verified successfully![/green]")

    with console.status("Storing token", spinner="dots"):
        _store_user_config({"token_id": result.token_id, "token_secret": result.token_secret}, profile=profile)
        console.print(f"[green]Token written to [white]{user_config_path}[/white] successfully![/green]")


@token_cli.command(help="Creates a new token by using an authenticated web session.")
def new(profile: Optional[str] = profile_option, no_verify: bool = False, source: Optional[str] = None):
    _new_token(profile, no_verify, source)
