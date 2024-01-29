# Copyright Modal Labs 2022
import getpass
from typing import Optional

import rich
import typer

from modal.client import _Client
from modal.config import _lookup_workspace, _store_user_config, config, config_profiles, user_config_path
from modal.token_flow import _new_token
from modal_utils.async_utils import synchronizer

token_cli = typer.Typer(name="token", help="Manage tokens.", no_args_is_help=True)

profile_option = typer.Option(
    None,
    help="Modal profile to set credentials for. You can switch the currently active Modal profile with the `modal profile` command. If unspecified, uses `default` profile.",
)


@token_cli.command(
    name="set",
    help="Set account credentials for connecting to Modal. If not provided with the command, you will be prompted to enter your credentials.",
)
@synchronizer.create_blocking
async def set(
    token_id: Optional[str] = typer.Option(None, help="Account token ID."),
    token_secret: Optional[str] = typer.Option(None, help="Account token secret."),
    profile: Optional[str] = profile_option,
    no_verify: bool = False,
):
    if token_id is None:
        token_id = getpass.getpass("Token ID:")
    if token_secret is None:
        token_secret = getpass.getpass("Token secret:")

    # TODO add server_url as a parameter for verification?
    server_url = config.get("server_url", profile=profile)
    if not no_verify:
        rich.print(f"Verifying token against [blue]{server_url}[/blue]")
        await _Client.verify(server_url, (token_id, token_secret))
        rich.print("[green]Token verified successfully[/green]")

    if profile is None:
        # TODO what if this fails verification but no_verify was False?
        workspace = await _lookup_workspace(server_url, token_id, token_secret)
        profile = workspace.username

    # TODO add activate as a parameter?
    config_data = {"token_id": token_id, "token_secret": token_secret}
    if not config_profiles():  # TODO or use activate flag?
        config_data["active"] = True
    _store_user_config(config_data, profile=profile)
    # TODO unify formatting with new_token output
    rich.print(f"Token written to {user_config_path} in profile {profile}")


@token_cli.command(name="new", help="Creates a new token by using an authenticated web session.")
@synchronizer.create_blocking
async def new(profile: Optional[str] = profile_option, no_verify: bool = False, source: Optional[str] = None):
    await _new_token(profile=profile, no_verify=no_verify, source=source)
