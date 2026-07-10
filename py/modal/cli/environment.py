# Copyright Modal Labs 2023
import json as json_mod

import click
import rich
from click import UsageError
from rich.text import Text

from modal import environments
from modal._environments import _Environment
from modal._utils.async_utils import synchronizer
from modal._utils.name_utils import check_environment_name
from modal._utils.time_utils import format_interval
from modal.cli.billing import DATE_HELP, _validate_and_parse_interval
from modal.cli.utils import confirm_or_suggest_yes, display_table, yes_option
from modal.config import config
from modal.environments import Environment
from modal.output import OutputManager

from ._help import ModalGroup

ENVIRONMENT_HELP_TEXT = """Create and interact with Environments

Environments are sub-divisions of workspaces, allowing you to deploy the same app
in different namespaces. Each environment has their own set of Secrets and any
lookups performed from an app in an environment will by default look for entities
in the same environment.

Typical use cases for environments include having one for development and one for
production, to prevent overwriting production apps when developing new features
while still being able to deploy changes to a live environment.
"""

environment_cli = ModalGroup(name="environment", help=ENVIRONMENT_HELP_TEXT)


class RenderableBool(Text):
    def __init__(self, value: bool):
        self.value = value

    def __rich__(self):
        return repr(self.value)


@environment_cli.command("list", help="List all environments in the current workspace.")
@click.option("--json", is_flag=True, default=False)
def list_(json: bool = False):
    envs = environments.list_environments()

    # determine which environment is currently active, prioritizing the local default
    # over the server default
    active_env = config.get("environment")
    for env in envs:
        if env.default is True and active_env is None:
            active_env = env.name

    table_data = []
    for item in envs:
        is_active = item.name == active_env
        is_active_display: Text | str = str(is_active) if json else RenderableBool(is_active)
        row = [item.name, item.webhook_suffix, is_active_display]
        table_data.append(row)
    display_table(["name", "web suffix", "active"], table_data, json=json)


@environment_cli.command("create", help="Create a new environment in the current workspace.", no_args_is_help=True)
@click.argument("name")
@click.option("--restricted", is_flag=True, default=False, help="Enable RBAC restrictions on the new environment")
def create(name: str, restricted: bool = False):
    check_environment_name(name)
    Environment.objects.create(name, restricted=restricted)
    prefix = "Restricted " if restricted else ""
    rich.print(f"[green]✓[/green] {prefix}Environment created: {name}")


ENVIRONMENT_DELETE_HELP = """Delete an environment in the current workspace.

Deletes all apps in the selected environment and deletes the environment irrevocably.
"""


@environment_cli.command("delete", help=ENVIRONMENT_DELETE_HELP, no_args_is_help=True)
@click.argument("name")
@yes_option
def delete(
    name: str,
    *,
    yes: bool = False,
):
    if not yes:
        # Check if the environment exists before confirming the deletion
        Environment.from_name(name).hydrate()
        message = (
            f"Are you sure you want to irrevocably delete the environment '{name}' and"
            " all its associated Apps, Secrets, Volumes, Dicts and Queues?"
        )
        confirm_or_suggest_yes(message)

    Environment.objects.delete(name)
    rich.print(f"[green]✓[/green] Environment deleted: {name}")


@environment_cli.command("update", help="Update environment-level settings.", no_args_is_help=True)
@click.argument("current_name")
@click.option("--set-name", default=None, help="New name of the environment")
@click.option("--set-web-suffix", default=None, help="New web suffix of environment (empty string is no suffix)")
def update(
    current_name: str,
    set_name: str | None = None,
    set_web_suffix: str | None = None,
):
    if set_name is None and set_web_suffix is None:
        raise UsageError("You need to at least one new property (using --set-name or --set-web-suffix)")

    if set_name:
        check_environment_name(set_name)

    environments.update_environment(current_name, new_name=set_name, new_web_suffix=set_web_suffix)
    rich.print("[green]✓[/green] Environment updated")


MEMBERS_HELP_TEXT = """Manage members and their roles in a restricted Environment.

Restricted Environments use RBAC to limit the actions that can be performed by
users (and service users) based on roles: https://modal.com/docs/guide/rbac.
"""

members_cli = ModalGroup(name="members", help=MEMBERS_HELP_TEXT)
environment_cli.add_command(members_cli)

service_user_option = click.option(
    "--service-user", is_flag=True, default=False, help="Treat MEMBER as the name of a service user"
)


@members_cli.command("list", help="List the members of a restricted Environment", no_args_is_help=True)
@click.argument("environment")
@click.option("--json", is_flag=True, default=False)
def members_list(environment: str, json: bool = False):
    members = Environment.from_name(environment).members.list()
    if json:
        # Mirror the API output shape: a nested dict keyed by users / service_users.
        OutputManager.get().print_json(json_mod.dumps(members))
        return
    rows = []
    for name, role in members.get("users", {}).items():
        rows.append([name, role])
    for name, role in members.get("service_users", {}).items():
        rows.append([f"{name} (service user)", role])
    rows = sorted(rows, key=lambda x: x[0])
    display_table(["name", "role"], rows)


@members_cli.command("update", help="Add or update a member's role in a restricted Environment", no_args_is_help=True)
@click.argument("environment")
@click.argument("member")
@click.option(
    "--role", type=click.Choice(["contributor", "viewer"]), required=True, help="Role to assign to the member"
)
@service_user_option
def members_update(environment: str, member: str, role: str, service_user: bool = False):
    env = Environment.from_name(environment)
    if service_user:
        env.members.update(service_users={member: role})
    else:
        env.members.update(users={member: role})
    kind = "service user" if service_user else "user"
    rich.print(f"[green]✓[/green] Set {kind} {member!r} to role {role!r} in environment {environment!r}")


@members_cli.command("remove", help="Remove a member from a restricted Environment", no_args_is_help=True)
@click.argument("environment")
@click.argument("member")
@service_user_option
def members_remove(environment: str, member: str, service_user: bool = False):
    env = Environment.from_name(environment)
    if service_user:
        env.members.remove(service_users=[member])
    else:
        env.members.remove(users=[member])
    kind = "service user" if service_user else "user"
    rich.print(f"[green]✓[/green] Removed {kind} {member!r} from environment {environment!r}")


billing_cli = ModalGroup(name="billing", help="View billing and usage info for the given Environment.")
environment_cli.add_command(billing_cli)


@billing_cli.command("report", no_args_is_help=True)
@click.argument(
    "environment_name",
    required=False,
    default=None,
)
@click.option("--start", default=None, help=f"Start date. {DATE_HELP}")
@click.option("--end", default=None, help=f"End date. {DATE_HELP} Defaults to now.")
@click.option(
    "--for",
    "for_",
    default=None,
    help="Convenience range: today, yesterday, this week, last week, this month, last month.",
)
@click.option("-r", "--resolution", default="d", help="Time resolution: 'd' (daily) or 'h' (hourly).")
@click.option(
    "--tz",
    default=None,
    help="Timezone for date interpretation: 'local', offset (5, -4, +05:30), or IANA name. Requires hourly resolution.",
)
@click.option("-t", "--tag-names", default=None, help="Comma-separated list of tag names to include.")
@click.option("--show-resources", is_flag=True, default=False, help="Further break down usage by resource type.")
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--csv", "csv", is_flag=True, default=False, help="Output as CSV.")
@synchronizer.create_blocking
async def environment_billing(
    environment_name: str | None,
    start: str | None,
    end: str | None,
    for_: str | None,
    resolution: str,
    tz: str | None,
    tag_names: str | None,
    show_resources: bool,
    json: bool,
    csv: bool,
):
    """Generate a billing report for the specified Environment.

    The report range can be provided by setting `--start` / `--end` dates (`--end` defaults to 'now')
    or by requesting a date range using `--for` (e.g., `--for today`, `--for 'last month'`).

    This command provides a CLI frontend for the
    [`Environment.billing.report`](https://modal.com/docs/sdk/py/latest/modal.Environment#billingreport)
    API.

    Note that, as with the API, the start date is inclusive and the end date is exclusive.
    Data will be reported for full intervals only. Using `--for` is a convenient way to define a
    complete interval.

    In addition, the `--show-resources` option further breaks the cost in each bucket by the resource
    that generated it (CPU, Memory, specific GPU types, etc.). Note that the specific resource types
    included in the report are subject to change as Modal's billing model evolves.

    Examples:

    ```bash
    modal environment billing report --start 2025-12-01 --end 2026-01-01

    modal environment billing report --for "last month" --tag-names team,project

    modal environment billing report test_env --for today --resolution h

    modal environment billing report test_env --for "this month" --show-resources

    modal environment billing report prod_env --for yesterday -r h --tz local

    modal environment billing report main_env --for "last month" --csv > report.csv

    modal environment billing report main_env --start 2025-12-01 --json > report.json
    ```

    """
    if json and csv:
        raise click.UsageError("--json and --csv are mutually exclusive")

    interval = _validate_and_parse_interval(start, end, for_, tz, resolution)

    tags = [t.strip() for t in tag_names.split(",")] if tag_names else None

    if environment_name is None:
        env = _Environment.from_context()
    else:
        env = _Environment.from_name(environment_name)

    rows_data = await env.billing.report(
        start=interval.start,
        end=interval.end,
        resolution=resolution,
        tag_names=tags,
    )

    columns = [
        "Object ID",
        "Description",
        "Environment",
        "Interval Start",
        *(["Resource"] if show_resources else []),
        "Cost",
        *(["Tags"] if tags else []),
    ]

    rows = []
    for item in rows_data:
        row_base = [
            item.object_id,
            item.description,
            item.environment_name,
            format_interval(
                item.interval_start,
                tz=interval.tz,
                isoformat=json or csv,
                show_date_only=resolution == "d",
            ),
        ]

        row_set: list[list[str]] = []
        if show_resources:
            for resource, cost in item.cost_by_resource.items():
                row_set.append([*row_base, resource, str(cost)])
        else:
            row_set.append([*row_base, str(item.cost)])

        if tags:
            for row in row_set:
                row.append(json_mod.dumps(item.tags))

        rows.extend(row_set)

    display_table(columns, rows, json=json, csv=csv)
