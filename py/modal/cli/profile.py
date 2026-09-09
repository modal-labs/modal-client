# Copyright Modal Labs 2022

import asyncio
import os
from contextlib import suppress

import click
from rich.json import JSON
from rich.table import Table

from modal import config as config_module
from modal._utils.async_utils import synchronizer
from modal.config import (
    Config,
    _lookup_workspace,
    config_profiles,
    config_set_active_profile,
)
from modal.environments import Environment
from modal.exception import AuthError, InvalidError
from modal.output import OutputManager

from ._help import ModalGroup

profile_cli = ModalGroup(name="profile", help="Switch between Modal profiles.")


@profile_cli.command("activate", help="Change the active Modal profile.", no_args_is_help=True)
@click.argument("profile")
def activate(profile: str):
    config_set_active_profile(profile)
    config = Config()
    click.echo(f"Active profile: {profile}")
    env = config.get("environment", profile=profile)
    if not env:
        with suppress(Exception):
            env = Environment.from_context().hydrate().name
    if env:
        click.echo(f"Active environment: {env}")


@profile_cli.command("current", help="Print the currently active Modal profile.")
def current():
    click.echo(config_module._profile)


@profile_cli.command("list", help="Show all Modal profiles and highlight the active one.")
@click.option("--json", is_flag=True, default=False)
@synchronizer.create_blocking
async def list_(json: bool | None = False):
    config = Config()
    profiles = config_profiles()
    lookup_coros = [
        _lookup_workspace(
            config.get("server_url", profile=profile),
            token_id=config.get("token_id", profile=profile, use_env=False),
            token_secret=config.get("token_secret", profile=profile, use_env=False),
            oauth_refresh_token=config.get("oauth_refresh_token", profile=profile, use_env=False),
            oauth_client_id=config.get("oauth_client_id", profile=profile, use_env=False),
            oauth_client_secret=config.get("oauth_client_secret", profile=profile, use_env=False),
        )
        for profile in profiles
    ]
    responses = await asyncio.gather(*lookup_coros, return_exceptions=True)

    rows = []
    for profile, resp in zip(profiles, responses):
        active = profile == config_module._profile
        if isinstance(resp, AuthError):
            workspace = "Unknown (authentication failure)"
        elif isinstance(resp, TimeoutError):
            workspace = "Unknown (timed out)"
        elif isinstance(resp, Exception):
            # Catch-all for other exceptions, like incorrect server url
            workspace = "Unknown (profile misconfigured)"
        else:
            assert not isinstance(resp, BaseException)
            workspace = resp.username
        content = ["•" if active else "", profile, workspace]
        rows.append((active, content))

    env_based_workspace: str | None = None
    credential_env_vars = (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_OAUTH_REFRESH_TOKEN",
        "MODAL_OAUTH_CLIENT_ID",
        "MODAL_OAUTH_CLIENT_SECRET",
    )
    if any(env_var in os.environ for env_var in credential_env_vars):
        try:
            env_based_resp = await _lookup_workspace(
                config.get("server_url", profile=config_module._profile),
                token_id=config.get("token_id", profile=config_module._profile),
                token_secret=config.get("token_secret", profile=config_module._profile),
                oauth_refresh_token=config.get("oauth_refresh_token", profile=config_module._profile),
                oauth_client_id=config.get("oauth_client_id", profile=config_module._profile),
                oauth_client_secret=config.get("oauth_client_secret", profile=config_module._profile),
            )
            env_based_workspace = env_based_resp.username
        except AuthError:
            env_based_workspace = "Unknown (authentication failure)"
        except InvalidError:
            env_based_workspace = "Unknown (invalid credential configuration)"

    output = OutputManager.get()
    highlight = "bold green" if env_based_workspace is None else "yellow"
    if json:
        json_data = []
        for active, content in rows:
            json_data.append({"name": content[1], "workspace": content[2], "active": active})
        output.print(JSON.from_data(json_data))
    else:
        table = Table(" ", "Profile", "Workspace")
        for active, content in rows:
            table.add_row(*content, style=highlight if active else "dim")
        output.print(table)

    if env_based_workspace is not None:
        output.print(
            f"[yellow]Using [bold]{env_based_workspace}[/bold] workspace based on environment variables[/yellow]"
        )
