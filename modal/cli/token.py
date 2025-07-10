# Copyright Modal Labs 2022
import getpass
from typing import Optional

import typer

from modal._utils.async_utils import synchronizer
from modal.token_flow import _new_token, _set_token

token_cli = typer.Typer(name="token", help="Manage tokens.", no_args_is_help=True)

profile_option = typer.Option(
    None,
    help=(
        "Modal profile to set credentials for. If unspecified "
        "(and MODAL_PROFILE environment variable is not set), "
        "uses the workspace name associated with the credentials."
    ),
)
activate_option = typer.Option(
    True,
    help="Activate the profile containing this token after creation.",
)

verify_option = typer.Option(
    True,
    help="Make a test request to verify the new credentials.",
)


@token_cli.command(name="set")
@synchronizer.create_blocking
async def set(
    token_id: Optional[str] = typer.Option(None, help="Account token ID."),
    token_secret: Optional[str] = typer.Option(None, help="Account token secret."),
    profile: Optional[str] = profile_option,
    activate: bool = activate_option,
    verify: bool = verify_option,
):
    """Set account credentials for connecting to Modal.

    If the credentials are not provided on the command line, you will be prompted to enter them.
    """
    if token_id is None:
        token_id = getpass.getpass("Token ID:")
    if token_secret is None:
        token_secret = getpass.getpass("Token secret:")
    await _set_token(token_id, token_secret, profile=profile, activate=activate, verify=verify)


@token_cli.command(name="new")
@synchronizer.create_blocking
async def new(
    profile: Optional[str] = profile_option,
    activate: bool = activate_option,
    verify: bool = verify_option,
    source: Optional[str] = None,
):
    """Create a new token by using an authenticated web session."""
    await _new_token(profile=profile, activate=activate, verify=verify, source=source)
