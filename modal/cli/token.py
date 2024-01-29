# Copyright Modal Labs 2022
import getpass
from typing import Optional

import typer

from modal.token_flow import _new_token, _set_token
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
    await _set_token(token_id, token_secret, profile=profile, no_verify=no_verify)


@token_cli.command(name="new", help="Creates a new token by using an authenticated web session.")
@synchronizer.create_blocking
async def new(profile: Optional[str] = profile_option, no_verify: bool = False, source: Optional[str] = None):
    await _new_token(profile=profile, no_verify=no_verify, source=source)
