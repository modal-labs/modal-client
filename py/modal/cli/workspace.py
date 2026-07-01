# Copyright Modal Labs 2025
import json as json_mod

import click
import rich

from modal._utils.time_utils import timestamp_to_localized_str
from modal.cli.utils import confirm_or_suggest_yes, display_table, yes_option
from modal.output import OutputManager
from modal.workspace import Workspace

from ._help import ModalGroup

WORKSPACE_HELP_TEXT = """Interact with the current Modal Workspace.

A Workspace is the top-level account that owns your Modal resources. Use these commands
to manage workspace-level settings such as proxy tokens.
"""

workspace_cli = ModalGroup(name="workspace", help=WORKSPACE_HELP_TEXT)


MEMBERS_HELP_TEXT = """View the members of the current Workspace."""

members_cli = ModalGroup(name="members", help=MEMBERS_HELP_TEXT)
workspace_cli.add_command(members_cli)


@members_cli.command("list", help="List the members of the current Workspace.")
@click.option("--json", is_flag=True, default=False)
def members_list(json: bool = False):
    members = Workspace.from_context().members.list()
    rows = [
        [
            member.name,
            member.email,
            member.role,
            member.user_id,
            timestamp_to_localized_str(member.joined_at.timestamp(), json),
            timestamp_to_localized_str(member.last_active_at.timestamp(), json) if member.last_active_at else None,
        ]
        for member in members
    ]
    display_table(["Name", "Email", "Role", "User ID", "Joined", "Last active"], rows, json=json)


PROXY_TOKENS_HELP_TEXT = """Manage the proxy tokens of the current Workspace.

Proxy tokens provide authentication to HTTP interfaces on Modal Servers and Web Functions.
They are passed as request headers (`Modal-Key` and `Modal-Secret`). See
https://modal.com/docs/guide/webhook-proxy-auth for more information.

Proxy tokens and secrets have `wk-` and `ws-` prefixes, respectively. The cannot be
interchanged with API tokens (which use `ak-` and `as-` prefixes).

On workspaces with RBAC enabled, tokens are scoped to specific environments;
use the `allow` and `revoke` commands to manage environment associations.
"""

proxy_tokens_cli = ModalGroup(name="proxy-tokens", help=PROXY_TOKENS_HELP_TEXT)
workspace_cli.add_command(proxy_tokens_cli)


@proxy_tokens_cli.command("create", help="Create a proxy token in the current Workspace.")
@click.option("--json", is_flag=True, default=False)
def proxy_tokens_create(*, json: bool = False):
    """Create a proxy token in the current Workspace.

    The new token's ID and secret will be printed to stdout. The secret is only
    shown at creation time and cannot be retrieved later.
    """
    token = Workspace.from_context().proxy_tokens.create()
    output_manager = OutputManager.get()
    if json:
        output_manager.print_json(json_mod.dumps({"Modal-Key": token.token_id, "Modal-Secret": token.token_secret}))
    else:
        output_manager.print(f"Modal-Key: {token.token_id}\nModal-Secret: {token.token_secret}")


@proxy_tokens_cli.command("list", help="List the proxy tokens of the current Workspace.")
@click.option(
    "-e",
    "--environment",
    default=None,
    help="Only list tokens associated with this environment. Lists all tokens when omitted.",
)
@click.option("--json", is_flag=True, default=False)
def proxy_tokens_list(environment: str | None = None, json: bool = False):
    tokens = Workspace.from_context().proxy_tokens.list(environment_name=environment)
    # Emit a real boolean for JSON output, but a string for the rich table (which can't render a bare bool).
    rows = [
        [
            token.token_id,
            timestamp_to_localized_str(token.created_at.timestamp(), json),
            token.scoped if json else str(token.scoped),
        ]
        for token in tokens
    ]
    display_table(["Token ID", "Created at", "Scoped"], rows, json=json)


@proxy_tokens_cli.command("allow", help="Allow a proxy token to authenticate to an environment.", no_args_is_help=True)
@click.argument("token_id")
@click.argument("environment_name")
def proxy_tokens_allow(token_id: str, environment_name: str):
    Workspace.from_context().proxy_tokens.allow(token_id, environment_name)
    rich.print(f"[green]✓[/green] Allowed proxy token {token_id!r} to authenticate to environment {environment_name!r}")


@proxy_tokens_cli.command("revoke", help="Revoke a proxy token's access to an environment.", no_args_is_help=True)
@click.argument("token_id")
@click.argument("environment_name")
def proxy_tokens_revoke(token_id: str, environment_name: str):
    Workspace.from_context().proxy_tokens.revoke(token_id, environment_name)
    rich.print(f"[green]✓[/green] Revoked proxy token {token_id!r} access to environment {environment_name!r}")


@proxy_tokens_cli.command("delete", help="Delete a proxy token from the current Workspace.", no_args_is_help=True)
@click.argument("token_id")
@yes_option
def proxy_tokens_delete(token_id: str, yes: bool = False):
    if not yes:
        message = (
            f"Are you sure you want to delete proxy token {token_id!r}? "
            "Any web endpoints relying on it will no longer accept it."
        )
        confirm_or_suggest_yes(message)
    Workspace.from_context().proxy_tokens.delete(token_id)
    rich.print(f"[green]✓[/green] Deleted proxy token {token_id!r}")


settings_cli = ModalGroup(name="settings", help="Manage workspace settings. Must be workspace manager or owner.")
workspace_cli.add_command(settings_cli)


@settings_cli.command("list")
@click.option("--json", is_flag=True, default=False)
@click.pass_context
def settings_group(
    ctx: click.Context,
    json: bool,
):
    """View the current settings for the workspace."""
    if ctx.invoked_subcommand is not None:
        return
    s = Workspace.from_context().settings.list()
    rows = [
        ["default-environment", s.default_environment],
        ["image-builder-version", s.image_builder_version],
    ]
    display_table(["Setting", "Value"], rows, json=json)


@settings_cli.command("set", no_args_is_help=True)
@click.argument("setting")
@click.argument("value")
def settings_set(setting: str, value: str):
    """Update a workspace setting. Must be workspace manager or owner.

    The following settings can be updated:
    - `image-builder-version`: The image builder version determines the software included in our base images.
    - `default-environment`: The default environment to use when the environment is omitted from SDK or CLI methods.

    Usage:
    - `modal workspace settings set image-builder-version 2025.06`
    - `modal workspace settings set default-environment main`
    """
    ws = Workspace.from_context().hydrate()
    s = ws.settings.list()
    try:
        ws.settings.set(setting, value)
        rich.print(f"[green]✓[/green] {setting}: updated from {getattr(s, setting.replace('-', '_'))} to {value}")
    except Exception as e:
        rich.print(f"[red]✗[/red] {setting}: {e}")
        raise SystemExit(1)
