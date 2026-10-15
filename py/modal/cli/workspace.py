# Copyright Modal Labs 2025
import json as json_mod
import shutil
from datetime import timezone

import click
import rich
import rich.box
import rich.table
from rich.box import SIMPLE, SIMPLE_HEAD
from rich.syntax import Syntax
from rich.text import Text

from modal._utils.async_utils import synchronizer
from modal._utils.time_utils import parse_duration, timestamp_to_localized_str
from modal._workspace import _Workspace
from modal.cli.utils import confirm_or_suggest_yes, display_table, yes_option
from modal.output import OutputManager
from modal.workspace import Workspace
from modal_proto import api_pb2

from ._help import _MAX_HELP_WIDTH, ModalGroup

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

Proxy tokens provide authentication to Modal Endpoints, Servers, and Web Functions.

Proxy tokens and secrets have `wk-` and `ws-` prefixes, respectively. They cannot be
interchanged with API tokens (which use `ak-` and `as-` prefixes).

Proxy tokens are passed as request headers, either as a key / secret pair:

```
Modal-Key: wk-123
Modal-Secret: ws-456
```

Or as a single Bearer token:

```
Authorization: Bearer wk-123.ws-456
```

See https://modal.com/docs/guide/webhook-proxy-auth for more information.

On workspaces with RBAC enabled, tokens are scoped to specific environments;
use the `allow` and `revoke` commands to manage environment associations.
"""

proxy_tokens_cli = ModalGroup(name="proxy-tokens", help=PROXY_TOKENS_HELP_TEXT)
workspace_cli.add_command(proxy_tokens_cli)


@proxy_tokens_cli.command("list")
@click.option(
    "-e",
    "--environment",
    default=None,
    help="Only list tokens associated with this environment. Lists all tokens when omitted.",
)
@click.option("--json", is_flag=True, default=False)
def proxy_tokens_list(environment: str | None = None, json: bool = False):
    """List the proxy tokens of the current Workspace.

    Examples:

    ```
    modal workspace proxy-tokens list
    modal workspace proxy-tokens list --environment prod
    modal workspace proxy-tokens list --json
    ```
    """
    tokens = Workspace.from_context().proxy_tokens.list(environment_name=environment)
    # Emit a real boolean for JSON output, but a string for the rich table (which can't render a bare bool).
    rows = [
        [
            token.name,
            token.token_id,
            timestamp_to_localized_str(token.created_at.timestamp(), json),
            token.created_by,
            token.scoped if json else str(token.scoped),
        ]
        for token in tokens
    ]
    display_table(["Name", "Token ID", "Created at", "Created by", "Scoped"], rows, json=json)


@proxy_tokens_cli.command("create")
@click.option("--name", default="", help="Name to help identify the token.")
@click.option("--json", is_flag=True, default=False)
def proxy_tokens_create(*, name: str = "", json: bool = False):
    """Create a proxy token in the current Workspace.

    The new token's ID and secret will be printed to stdout. The secret is only
    shown at creation time and cannot be retrieved later.

    Examples:

    ```
    modal workspace proxy-tokens create --name production-webhooks
    modal workspace proxy-tokens create --json
    ```
    """
    token = Workspace.from_context().proxy_tokens.create(name=name)
    output_manager = OutputManager.get()
    bearer = f"Bearer {token.token_id}.{token.token_secret}"
    if json:
        output_manager.print_json(
            json_mod.dumps(
                {
                    "Modal-Key": token.token_id,
                    "Modal-Secret": token.token_secret,
                    "Authorization": bearer,
                }
            )
        )
    else:
        table = rich.table.Table(show_header=False, box=rich.box.HORIZONTALS, show_lines=True)
        table.add_row(f"Modal-Key: {token.token_id}\nModal-Secret: {token.token_secret}")
        table.add_row(f"Authorization: {bearer}")
        output_manager.print(table)


@proxy_tokens_cli.command("update", no_args_is_help=True)
@click.argument("token_id")
@click.argument("setting")
@click.argument("value")
def proxy_tokens_update(token_id: str, setting: str, value: str):
    """Update a proxy token in the current Workspace.

    The following settings can be updated:

    - `name`: A name to help identify the token. Pass an empty value to remove it.

    Example:

    ```
    modal workspace proxy-tokens update wk-123 name production-webhooks
    ```
    """
    if setting == "name":
        Workspace.from_context().proxy_tokens.update(token_id, name=value)
    else:
        raise click.UsageError(f"Unknown proxy token setting {setting!r}. Supported settings: name.")
    rich.print(f"[green]✓[/green] Updated proxy token {token_id!r}")


@proxy_tokens_cli.command("allow", no_args_is_help=True)
@click.argument("token_id")
@click.argument("environment_name")
def proxy_tokens_allow(token_id: str, environment_name: str):
    """Allow a proxy token to authenticate to an environment.

    Example:

    ```
    modal workspace proxy-tokens allow wk-123 prod
    ```
    """
    Workspace.from_context().proxy_tokens.allow(token_id, environment_name)
    rich.print(f"[green]✓[/green] Allowed proxy token {token_id!r} to authenticate to environment {environment_name!r}")


@proxy_tokens_cli.command("revoke", no_args_is_help=True)
@click.argument("token_id")
@click.argument("environment_name")
def proxy_tokens_revoke(token_id: str, environment_name: str):
    """Revoke a proxy token's access to an environment.

    Example:

    ```
    modal workspace proxy-tokens revoke wk-123 prod
    ```
    """
    Workspace.from_context().proxy_tokens.revoke(token_id, environment_name)
    rich.print(f"[green]✓[/green] Revoked proxy token {token_id!r} access to environment {environment_name!r}")


@proxy_tokens_cli.command("delete", no_args_is_help=True)
@click.argument("token_id")
@yes_option
def proxy_tokens_delete(token_id: str, yes: bool = False):
    """Delete a proxy token from the current Workspace.

    Example:

    ```
    modal workspace proxy-tokens delete wk-123
    ```
    """
    if not yes:
        message = (
            f"Are you sure you want to delete proxy token {token_id!r}? Requests using this token will be rejected."
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


SERVICE_USERS_HELP_TEXT = """Manage Service Users. Must be workspace manager or owner.

Service users are programmatic accounts that allow automated systems to interact with Modal. They're
ideal for CI/CD pipelines, automated deployments, and other workflows that need to authenticate.

Service User credentials are typically made available to a process as environment variables:

```
export MODAL_TOKEN_ID=your-token-id
export MODAL_TOKEN_SECRET=your-token-secret
```

A newly created Service User has no permissions by default. In order to interact with a specific
Environment, Service Users must be granted explicit permissions.

See https://modal.com/docs/guide/service-users for more information.
"""

service_users_cli = ModalGroup(name="service-users", help=SERVICE_USERS_HELP_TEXT)
workspace_cli.add_command(service_users_cli)


@service_users_cli.command("create", no_args_is_help=True)
@click.argument("name")
@click.option(
    "--expires-in",
    "expires_in_str",
    default=None,
    help="Expiration time. Accepts integers (interpreted as seconds), as well as values suffixed by "
    "`s`, `m`, `h`, or `d`.",
)
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@synchronizer.create_blocking
async def service_users_create(name: str, expires_in_str: str | None, json: bool):
    """Create a Service User in this workspace.

    In order to run this command, you must have at least Manager level permissions in the Workspace.

    By default, Service Users have no permissions. To explicitly grant the Service User permissions
    to interact with a given environment, use [`modal workspace service-users set-role`](https://modal.com/docs/cli/latest/workspace#modal-workspace-service-users-set-role)
    or [`modal environment roles update`](https://modal.com/docs/cli/latest/environment#modal-environment-roles-update).

    Optionally, you can also specify an expiration window by passing `--expires-in`. After the specified
    amount of time elapses, this Service User will be deleted, and any credentials associated with
    this Service User will stop working. The option expects a positive integer value, optionally
    suffixed by the units `s`, `m`, `h`, or `d`, meaning seconds, minutes, hours, and days respectively.
    If no suffix is provided, the value is interpreted as seconds. If `--expires-in` is not provided,
    the newly created Service User will never expire.

    Examples:

    ```bash
    # Create a Service User named `test-1`, which never expires:
    modal workspace service-users create test-1

    # Two ways of creating a Service User which is valid for an hour:
    modal workspace service-users create test-2 --expires-in 3600
    modal workspace service-users create test-3 --expires-in 1h

    # Create a Service User which is valid for 1 year, outputting JSON:
    modal workspace service-users create test-4 --expires-in 365d --json
    ```
    """

    expires_in: int | None
    if expires_in_str is None:
        expires_in = None
    else:
        try:
            expires_in = int(expires_in_str)
        except ValueError:
            try:
                expires_in = int(parse_duration(expires_in_str).total_seconds())
            except ValueError as e:
                raise click.UsageError(str(e)) from e

    if expires_in is not None and expires_in <= 0:
        raise click.UsageError("Value passed to `--expires-in` must be a positive integer.")

    if expires_in is None:
        req = api_pb2.ServiceUserCreateRequest(name=name)
    else:
        req = api_pb2.ServiceUserCreateRequest(name=name, expires_in_seconds=expires_in)

    ws = _Workspace.from_context()
    await ws.hydrate()

    resp = await ws.client._stub.ServiceUserCreate(req)

    output_manager = OutputManager.get()

    if json:
        output_manager.print_json(
            json_mod.dumps(
                {
                    "token_id": resp.token_id,
                    "token_secret": resp.token_secret,
                    "expires_at": timestamp_to_localized_str(resp.expires_at.ToDatetime(timezone.utc).timestamp())
                    if expires_in
                    else None,
                }
            )
        )
        return

    base = f"[green]✓[/green] Successfully created Service User [bold]{name}[/bold]"
    if expires_in:
        expiration_time = timestamp_to_localized_str(resp.expires_at.ToDatetime(timezone.utc).timestamp(), isotz=False)
        base = f"{base}, expiring at [bold]{expiration_time}[/bold]"

    output_manager.print(f"{base}.\n", highlight=False)

    output_manager.print("To use it, set the following environment variables in your program:\n")

    columns, _ = shutil.get_terminal_size()
    s = Syntax(
        f"\n export MODAL_TOKEN_ID={resp.token_id}\n export MODAL_TOKEN_SECRET={resp.token_secret}\n",
        "sh",
        code_width=min(_MAX_HELP_WIDTH, columns),
        word_wrap=True,
    )

    output_manager.print(s)
    output_manager.print("")

    output_manager.print("[yellow]:warning:[/yellow] This is the only time you will be able to see the Token Secret.")


@service_users_cli.command("list")
@click.option("--json", is_flag=True, default=False, help="Output as JSON.")
def service_users_list(json: bool):
    """List all Service Users in this workspace.

    In order to run this command, you must have at least Manager level permissions in the Workspace.

    Examples:

    ```bash
    # List all Service Users with a human-readable output format
    modal workspace service-users list

    # List all Service Users and return JSON output
    modal workspace service-users list --json
    ```
    """

    ws = Workspace.from_context()
    service_users = ws.service_users.list()

    output_manager = OutputManager.get()

    if json:
        output_manager.print_json(
            json_mod.dumps(
                [
                    {
                        "name": su.name,
                        "service_user_id": su.service_user_id,
                        "token_id": su.token_id,
                        "created_by": su.created_by,
                        "created_at": timestamp_to_localized_str(su.created_at.timestamp()),
                        "last_used_at": timestamp_to_localized_str(su.last_used_at.timestamp())
                        if su.last_used_at is not None
                        else None,
                        "expires_at": timestamp_to_localized_str(su.expires_at.timestamp())
                        if su.expires_at is not None
                        else None,
                        "environment_roles": su.environment_roles,
                    }
                    for su in service_users
                ]
            )
        )
        return

    table = rich.table.Table(box=SIMPLE_HEAD)
    table.add_column("Name")
    table.add_column("Token ID")
    table.add_column("Created By")
    table.add_column("Expires")

    for su in service_users:
        table.add_row(
            su.name,
            su.token_id,
            su.created_by,
            timestamp_to_localized_str(su.expires_at.timestamp(), isotz=False) if su.expires_at is not None else "-",
        )

    output_manager.print(table)


@service_users_cli.command("delete", no_args_is_help=True)
@click.argument("name")
@click.option("--allow-missing", is_flag=True, default=False, help="Don't error if the Volume doesn't exist.")
@yes_option
def service_users_delete(name: str, *, allow_missing: bool, yes: bool):
    """Delete a Service User in this Workspace.

    Note that this action is irreversible. Any existing processes / scripts that use the credentials
    associated with the target Service User will stop being able to authenticate with Modal.

    ```bash
    # Delete the Service User named `test-service-user`
    modal workspace service-users delete test-service-user
    ```
    """
    if not yes:
        confirm_or_suggest_yes(
            f"Are you sure you want to irrevocably delete the Service User '{name}' and its associated credentials?"
        )

    ws = Workspace.from_context()
    ws.service_users.delete(name, allow_missing=allow_missing)

    output_manager = OutputManager.get()
    output_manager.print(f"[green]✓[/green] Deleted Service User `{name}`.")


@service_users_cli.command("info", no_args_is_help=True)
@click.argument("name")
@click.option("--json", is_flag=True, default=False, help="Output as JSON.")
def service_users_info(name: str, json: bool):
    """Inspect a Service User in this workspace.

    Examples:

    ```bash
    # Inspect the Service User named `test-service-user`
    modal workspace service-users info test-service-user

    # Inspect the Service User named `test-service-user-2`, outputting JSON
    modal workspace service-users info test-service-user-2 --json
    ```
    """

    ws = Workspace.from_context()
    su = ws.service_users.info(name)

    output_manager = OutputManager.get()
    if json:
        output_manager.print_json(
            json_mod.dumps(
                {
                    "name": su.name,
                    "service_user_id": su.service_user_id,
                    "token_id": su.token_id,
                    "created_by": su.created_by,
                    "created_at": timestamp_to_localized_str(su.created_at.timestamp()),
                    "last_used_at": timestamp_to_localized_str(su.last_used_at.timestamp())
                    if su.last_used_at is not None
                    else None,
                    "expires_at": timestamp_to_localized_str(su.expires_at.timestamp())
                    if su.expires_at is not None
                    else None,
                    "environment_roles": su.environment_roles,
                }
            )
        )
        return

    rows: list[tuple[str, str]] = []

    rows.append(("Name:", su.name))
    rows.append(("Service User ID:", su.service_user_id))
    rows.append(("Token ID:", su.token_id))

    environment_roles: list[tuple[str, str]] = []
    for env_name, role in su.environment_roles.items():
        if role == "no-access":
            continue

        environment_roles.append((env_name, role))

    rows.append(("Created By:", su.created_by))
    rows.append(("Created At:", str(timestamp_to_localized_str(su.created_at.timestamp(), isotz=False))))
    rows.append(
        (
            "Last Used:",
            str(timestamp_to_localized_str(su.last_used_at.timestamp(), isotz=False))
            if su.last_used_at is not None
            else "-",
        )
    )
    rows.append(
        (
            "Expires:",
            str(timestamp_to_localized_str(su.expires_at.timestamp(), isotz=False))
            if su.expires_at is not None
            else "-",
        )
    )

    if len(environment_roles) > 0:
        rows.append(("", ""))
        rows.append(("Environment Access:", ""))
        for env_name, r in environment_roles:
            rows.append((f"  {env_name}", r.capitalize()))

    if len(environment_roles) == 0:
        output_manager.print("")
        output_manager.print("  [yellow]:warning:[/yellow] This Service User does not have access to any Environments.")

    table = rich.table.Table(box=SIMPLE, show_header=False)
    for row in rows:
        left, right = row
        table.add_row(left, Text(right, justify="right"))

    output_manager.print(table)


@service_users_cli.command("set-role", no_args_is_help=True)
@click.argument("service_user_name")
@click.argument("environment_name")
@click.argument("role")
def service_users_set_role(
    service_user_name: str,
    environment_name: str,
    role: str,
):
    """Update a Service User's role in a specific Environment.

    Examples:

    ```bash
    # Give Contributor access in Environment `main` to Service User `test-service-user`
    modal workspace service-users set-role test-service-user main contributor

    # Revoke access to Environment `prod` from Service User `test-service-user`
    modal workspace service-users set-role test-service-user prod no-access
    ```
    """

    ws = Workspace.from_context()
    info = ws.service_users.info(service_user_name)

    existing_role = info.environment_roles.get(environment_name) or "no-access"

    ws.service_users.set_role(service_user_name, environment_name, role=role)

    output_manager = OutputManager.get()
    if role == existing_role:
        if role == "no-access":
            output_manager.print(
                f"Service User [bold]{service_user_name}[/bold] does not have access to "
                f"Environment [bold]{environment_name}[/bold] already. Role was not updated."
            )
        else:
            output_manager.print(
                f"Service User [bold]{service_user_name}[/bold] is already a {role.capitalize()} in "
                f"Environment [bold]{environment_name}[/bold]. Role was not updated."
            )
    elif role == "no-access":
        output_manager.print(
            f"[green]✓[/green] Revoked Service User [bold]{service_user_name}[/bold]'s access to "
            f"Environment [bold]{environment_name}[/bold]."
        )
    else:
        output_manager.print(
            f"[green]✓[/green] Updated Service User [bold]{service_user_name}[/bold]'s role in "
            f"Environment [bold]{environment_name}[/bold] to {role.capitalize()}."
        )
