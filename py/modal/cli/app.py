# Copyright Modal Labs 2022
import re
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional, Union, get_args

import click
import rich
import typer
from click import UsageError
from rich.table import Column
from rich.text import Text
from typer import Argument

from modal._object import _get_environment_name
from modal._traceback import print_server_warnings
from modal._utils.async_utils import synchronizer
from modal._utils.browser_utils import open_url_and_display
from modal.client import _Client
from modal.environments import ensure_env
from modal.exception import InvalidError, NotFoundError
from modal.output import OutputManager
from modal.runner import DEPLOYMENT_STRATEGY_TYPE, _stop_and_wait_for_containers
from modal_proto import api_pb2

from .._logs import _FETCH_LIMIT, _MAX_FETCH_RANGE, LogsFilters
from .._utils.time_utils import locale_tz, timestamp_to_localized_str
from .utils import (
    ENV_OPTION,
    YES_OPTION,
    confirm_or_suggest_yes,
    display_table,
    fetch_app_logs,
    stream_app_logs,
    tail_app_logs,
)

APP_IDENTIFIER = Argument("", help="App name or ID")
NAME_OPTION = typer.Option("", "-n", "--name", help="Deprecated: Pass App name as a positional argument")

app_cli = typer.Typer(name="app", help="Manage deployed and running apps.", no_args_is_help=True)

APP_STATE_TO_MESSAGE = {
    api_pb2.APP_STATE_DEPLOYED: Text("deployed", style="green"),
    api_pb2.APP_STATE_DETACHED: Text("ephemeral (detached)", style="green"),
    api_pb2.APP_STATE_DETACHED_DISCONNECTED: Text("ephemeral (detached)", style="green"),
    api_pb2.APP_STATE_DISABLED: Text("disabled", style="dim"),
    api_pb2.APP_STATE_EPHEMERAL: Text("ephemeral", style="green"),
    api_pb2.APP_STATE_INITIALIZING: Text("initializing...", style="yellow"),
    api_pb2.APP_STATE_STOPPED: Text("stopped", style="blue"),
    api_pb2.APP_STATE_STOPPING: Text("stopping...", style="blue"),
}


async def resolve_app_identifier(
    app_identifier: str, env: Optional[str], client: Optional[_Client] = None
) -> tuple[str, str, api_pb2.AppLifecycle]:  # Return app_id, environment_name, lifecycle
    """Handle an App ID or an App name and return context about the App it points at.

    When a name is provided, we may retrieve either a currently deployed App or an App that
    was recently stopped (if no other App with that name has been deployed since).
    It is up to callers of this function to decide whether it's valid to use the App ID
    based on the lifecycle returned and their specific operations.

    Can also raise a NotFoundError if the argument matches the App ID regex but the App
    doesn't exist on the backend, or if there is no currently deployed or recently stopped App
    with that name.

    The function also always returns a valid environment name for any name-based lookups,
    which may reflect the server-defined default environment when the provided argument was null.

    """
    if client is None:
        client = await _Client.from_env()
    if re.match(r"^ap-[a-zA-Z0-9]{22}$", app_identifier):
        # Identifier is an App ID. This is unambiguous, so we can make the request and return
        # the lifecycle. AppGetLifecycle will raise NotFoundError if the ID doesn't point at an App.
        # If we return, it's a real App, but it's up to the caller to decide what to do based on
        # the App's current state as conveyed by the lifecycle. We do propagate a NotFoundError
        # from the server if the App ID doesn't actually exist.
        request = api_pb2.AppGetLifecycleRequest(app_id=app_identifier)
        resp = await client.stub.AppGetLifecycle(request)
        return app_identifier, "", resp.lifecycle
    else:
        # Identifier is treated as a name, which may or may not point at a currently deployed App
        # (inside a specific environment)
        request = api_pb2.AppGetByDeploymentNameRequest(name=app_identifier, environment_name=env or "")
        resp = await client.stub.AppGetByDeploymentName(request)
        if resp.app_id:
            # App is currently deployed
            return resp.app_id, resp.environment_name, resp.lifecycle
        elif resp.previous_app_id:
            # An App with this name was recently stopped. Return the ID of the stopped App
            # and let callers decide what to do based on the lifecycle.
            return resp.previous_app_id, resp.environment_name, resp.lifecycle
        else:
            msg = f"No App with name '{app_identifier}' found in the '{resp.environment_name}' environment."
            raise NotFoundError(msg)


@app_cli.command("list")
@synchronizer.create_blocking
async def list_(env: Optional[str] = ENV_OPTION, json: bool = False):
    """List Modal apps that are currently deployed/running or recently stopped."""
    env = ensure_env(env)
    client = await _Client.from_env()

    resp: api_pb2.AppListResponse = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=_get_environment_name(env))
    )

    columns: list[Union[Column, str]] = [
        Column("App ID", min_width=25),  # Ensure that App ID is not truncated in slim terminals
        "Description",
        "State",
        "Tasks",
        "Created at",
        "Stopped at",
    ]
    rows: list[list[Union[Text, str]]] = []
    for app_stats in resp.apps:
        state = APP_STATE_TO_MESSAGE.get(app_stats.state, Text("unknown", style="gray"))
        rows.append(
            [
                app_stats.app_id,
                app_stats.description,
                state,
                str(app_stats.n_running_tasks),
                timestamp_to_localized_str(app_stats.created_at, json),
                timestamp_to_localized_str(app_stats.stopped_at, json),
            ]
        )

    env_part = f" in environment '{env}'" if env else ""
    display_table(columns, rows, json, title=f"Apps{env_part}")


_RELATIVE_TIME_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_time_arg(value: Optional[str], default: datetime) -> datetime:
    """Parse a time argument that can be a relative duration (e.g. '2h', '30m') or ISO 8601 datetime.

    Naive datetime values are interpreted in the user's local timezone.
    Relative durations are always UTC-relative.
    """
    if value is None:
        return default

    # Try relative duration: digits followed by a unit letter
    if len(value) >= 2 and value[-1] in _RELATIVE_TIME_UNITS and value[:-1].isdigit():
        seconds = int(value[:-1]) * _RELATIVE_TIME_UNITS[value[-1]]
        return datetime.now(timezone.utc) - timedelta(seconds=seconds)

    # Try ISO 8601 datetime
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            # Interpret naive datetimes in the user's local timezone
            dt = dt.replace(tzinfo=locale_tz())
        return dt
    except ValueError:
        raise UsageError(f"Invalid time format: '{value}'. Use a relative duration (e.g. '2h') or ISO 8601 datetime.")


_DEFAULT_LOGS_TAIL = 100


_SOURCE_OPTIONS = {
    "stdout": api_pb2.FILE_DESCRIPTOR_STDOUT,
    "stderr": api_pb2.FILE_DESCRIPTOR_STDERR,
    "system": api_pb2.FILE_DESCRIPTOR_INFO,
}


@app_cli.command("logs", no_args_is_help=True)
@synchronizer.create_blocking
async def logs(
    app_identifier: str = APP_IDENTIFIER,
    follow: bool = typer.Option(False, "-f", "--follow", help="Stream log output until App stops"),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help=(
            "Start of time range. Accepts ISO 8601 datetime or relative time, e.g. '1d' (1 day ago), '2h', '30m', etc."
        ),
    ),
    until: Optional[str] = typer.Option(
        None,
        "--until",
        help="End of time range; accepts same argument types as --since",
    ),
    tail: Optional[int] = typer.Option(None, "--tail", "-n", help="Show only the last N log entries"),
    search: Optional[str] = typer.Option(None, "--search", help="Filter by search text"),
    function_id: Optional[str] = typer.Option("", "--function", help="Filter by Function ID (fu-*)"),
    function_call_id: Optional[str] = typer.Option("", "--function-call", help="Filter by FunctionCall ID (fc-*)"),
    container_id: Optional[str] = typer.Option("", "--container", help="Filter by Container ID (ta-*)"),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Filter by source: 'stdout', 'stderr', or 'system'"
    ),
    timestamps: bool = typer.Option(False, "--timestamps", help="Prefix each line with its timestamp"),
    show_function_id: bool = typer.Option(False, "--show-function-id", help="Prefix each line with its Function ID"),
    show_function_call_id: bool = typer.Option(
        False, "--show-function-call-id", help="Prefix each line with its FunctionCall ID"
    ),
    show_container_id: bool = typer.Option(False, "--show-container-id", help="Prefix each line with its Container ID"),
    *,
    env: Optional[str] = ENV_OPTION,
):
    """Fetch or stream App logs.

    By default, this command fetches the last 100 log entries and exits. Use ``-f`` to
    live-stream logs from a running App instead. Fetch and follow are mutually exclusive.

    **Examples:**

    Get recent logs based on an app ID:

    ```
    modal app logs ap-123456
    ```

    Get recent logs for a currently deployed App based on its name:

    ```
    modal app logs my-app
    ```

    Follow (stream) logs from a running App:

    ```
    modal app logs my-app -f
    ```

    Fetch the last 1000 entries:

    ```
    modal app logs my-app --tail 1000
    ```

    Fetch logs from the last 2 hours:

    ```
    modal app logs my-app --since 2h
    ```

    Fetch logs in a specific time range:

    ```
    modal app logs my-app --since 2026-03-01T05:00:00 --until 2026-03-01T08:00:00
    ```

    Filter the logs by source and function:

    ```
    modal app logs my-app --source stderr --function fu-abc123
    ```

    Include timestamps along with Function and Container IDs on each line:

    ```
    modal app logs my-app --timestamps --show-function-id --show-container-id
    ```

    """
    if not app_identifier:
        raise UsageError("Either an App ID or name must be provided.")

    if follow and (since or until or tail):
        raise UsageError("--follow cannot be combined with --since, --until, or --tail.")

    if tail is not None and tail <= 0:
        raise UsageError("--tail value must be positive.")

    if tail is not None and tail > _FETCH_LIMIT:
        raise UsageError(f"--tail value must not exceed {_FETCH_LIMIT}.")

    app_id, _, _ = await resolve_app_identifier(app_identifier, env)

    if source is not None:
        if source not in _SOURCE_OPTIONS:
            raise UsageError(f"Invalid source: '{source}'. Must be 'stdout', 'stderr', or 'system'.")
        source_fd = _SOURCE_OPTIONS[source]
    else:
        source_fd = api_pb2.FILE_DESCRIPTOR_UNSPECIFIED

    prefix_fields: list[str] = []
    if show_function_id:
        prefix_fields.append("fu")
    if show_function_call_id:
        prefix_fields.append("fc")
    if show_container_id:
        prefix_fields.append("ta")

    log_filters = LogsFilters(
        source=source_fd,
        function_id=function_id or "",
        function_call_id=function_call_id or "",
        task_id=container_id or "",
        search_text=search or "",
    )

    if follow:
        await stream_app_logs.aio(
            app_id,
            task_id=container_id or "",
            show_timestamps=timestamps,
            follow=True,
            prefix_fields=prefix_fields,
            filters=log_filters,
        )
    else:
        now = datetime.now(timezone.utc)
        since_dt = _parse_time_arg(since, default=now) if since else None
        until_dt = _parse_time_arg(until, default=now) if until else None

        if since_dt is not None and until_dt is not None and since_dt >= until_dt:
            raise UsageError("--since must be before --until.")

        if since_dt is not None:
            effective_until = until_dt or now
            if effective_until - since_dt > _MAX_FETCH_RANGE:
                raise UsageError(f"Log fetch time range cannot exceed {_MAX_FETCH_RANGE.days} days.")

        if since and tail is None:
            # Range mode: --since without --tail fetches everything in the range.
            await fetch_app_logs.aio(
                app_id,
                since_dt,
                until_dt or now,
                show_timestamps=timestamps,
                prefix_fields=prefix_fields,
                filters=log_filters,
            )
        else:
            # Tail mode: single fetch with limit.
            # --since is a hard floor, --until shifts the anchor.
            effective_tail = tail if tail is not None else _DEFAULT_LOGS_TAIL
            await tail_app_logs.aio(
                app_id,
                effective_tail,
                show_timestamps=timestamps,
                since=since_dt,
                until=until_dt,
                prefix_fields=prefix_fields,
                filters=log_filters,
            )


@app_cli.command("rollback", no_args_is_help=True, context_settings={"ignore_unknown_options": True})
@synchronizer.create_blocking
async def rollback(
    app_identifier: str = APP_IDENTIFIER,
    version: str = typer.Argument("", help="Target version for rollback."),
    *,
    env: Optional[str] = ENV_OPTION,
):
    """Redeploy a previous version of an App.

    Note that the App must currently be in a "deployed" state.
    Rollbacks will appear as a new deployment in the App history, although
    the App state will be reset to the state at the time of the previous deployment.

    **Examples:**

    Rollback an App to its previous version:

    ```
    modal app rollback my-app
    ```

    Rollback an App to a specific version:

    ```
    modal app rollback my-app v3
    ```

    Rollback an App using its App ID instead of its name:

    ```
    modal app rollback ap-abcdefghABCDEFGH123456
    ```

    """
    env = ensure_env(env)
    client = await _Client.from_env()
    app_id, environment_name, lifecycle = await resolve_app_identifier(app_identifier, env, client)
    if lifecycle.app_state != api_pb2.APP_STATE_DEPLOYED:
        env_suffix = f" in the '{environment_name}' environment" if environment_name else ""
        raise InvalidError(f"App '{app_identifier}' is not deployed{env_suffix}.")

    if not version:
        version_number = -1
    else:
        if m := re.match(r"v(\d+)", version):
            version_number = int(m.group(1))
        else:
            raise UsageError(f"Invalid version specifier: {version}")
    req = api_pb2.AppRollbackRequest(app_id=app_id, version=version_number)
    await client.stub.AppRollback(req)
    rich.print("[green]✓[/green] Deployment rollback successful!")


@app_cli.command("rollover", no_args_is_help=True)
@synchronizer.create_blocking
async def rollover(
    app_identifier: str = APP_IDENTIFIER,
    *,
    strategy: str = typer.Option(
        "rolling",
        help="Strategy for rollover",
        click_type=click.Choice(get_args(DEPLOYMENT_STRATEGY_TYPE)),
    ),
    env: Optional[str] = ENV_OPTION,
):
    """Redeploy an App to get new containers without code changes.

    A rollover replaces existing containers with fresh ones built from the same
    App version — useful for refreshing containers without changing your code.
    The rollover appears as a new entry in the App's deployment history.

    **Examples:**

    Rollover an App using a rolling deployment. Running containers are now considered
    outdated and will be gracefully replaced by new ones.

    ```
    modal app rollover my-app
    ```

    Rollover an App by terminating any running containers. Inputs on the queue will
    start new containers.

    ```
    modal app rollover my-app --strategy recreate
    ```
    """
    env = ensure_env(env)
    client = await _Client.from_env()

    app_id, environment_name, lifecycle = await resolve_app_identifier(app_identifier, env, client)
    if lifecycle.app_state != api_pb2.APP_STATE_DEPLOYED:
        env_suffix = f" in the '{environment_name}' environment" if environment_name else ""
        raise InvalidError(f"App '{app_identifier}' is not deployed{env_suffix}.")

    output_mgr = OutputManager.get()
    output_mgr.print(f"🔨 Starting app rollover with {strategy} strategy")
    t0 = time.monotonic()

    req = api_pb2.AppRolloverRequest(app_id=app_id)
    response = await client.stub.AppRollover(req)
    print_server_warnings(response.server_warnings)

    if strategy == "recreate":
        try:
            await _stop_and_wait_for_containers(client, app_id, response.deployed_at, env)
        except Exception as exc:
            warnings.warn(f"App updated successfully, but containers did not all terminate. {exc}", UserWarning)
            output_mgr.print(f"\nView Deployment: [magenta]{response.url}[/magenta]")
            sys.exit(1)

    duration = time.monotonic() - t0
    output_mgr.step_completed(f"Rollover completed in {duration:.3f}s with {strategy} strategy! 🎉")
    output_mgr.print(f"\nView Deployment: [magenta]{response.url}[/magenta]")


@app_cli.command("stop", no_args_is_help=True)
@synchronizer.create_blocking
async def stop(
    app_identifier: str = APP_IDENTIFIER,
    *,
    yes: bool = YES_OPTION,
    env: Optional[str] = ENV_OPTION,
):
    """Permanently stop an App and terminate its running containers."""
    env = ensure_env(env)
    client = await _Client.from_env()
    app_id, environment_name, lifecycle = await resolve_app_identifier(app_identifier, env, client)

    if lifecycle.app_state == api_pb2.APP_STATE_STOPPED:
        msg = "App is already stopped."
        if lifecycle.stopped_at:
            stopped_at = timestamp_to_localized_str(lifecycle.stopped_at)
            verb = "Stopped" if lifecycle.stopped_by else "Finished"
            attribution = f" by '{lifecycle.stopped_by}'" if lifecycle.stopped_by else ""
            msg += f" ({verb} at {stopped_at}{attribution})."
        raise SystemExit(msg)

    if not yes:
        res = await client.stub.TaskList(api_pb2.TaskListRequest(app_id=app_id))
        num_containers = len(res.tasks)

        if environment_name:
            msg = f"Are you sure you want to stop App '{app_identifier}' in the '{environment_name}' environment?"
        else:
            msg = f"Are you sure you want to stop App '{app_identifier}'?"

        if num_containers:
            msg += (
                f" This will immediately terminate {num_containers} running"
                f" container{'s' if num_containers != 1 else ''}."
            )
        else:
            msg += " No containers are currently running."
        confirm_or_suggest_yes(msg)
    req = api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI)
    await client.stub.AppStop(req)


@app_cli.command("history", no_args_is_help=True)
@synchronizer.create_blocking
async def history(
    app_identifier: str = APP_IDENTIFIER,
    *,
    env: Optional[str] = ENV_OPTION,
    json: bool = False,
):
    """Show an App's deployment history.

    **Examples:**

    Get the history based on an app ID:

    ```
    modal app history ap-123456
    ```

    Get the history for an App based on its name:

    ```
    modal app history my-app
    ```

    """
    env = ensure_env(env)
    client = await _Client.from_env()
    app_id, _, _ = await resolve_app_identifier(app_identifier, env, client)
    resp = await client.stub.AppDeploymentHistory(api_pb2.AppDeploymentHistoryRequest(app_id=app_id))

    columns = [
        "Version",
        "Time deployed",
        "Client",
        "Deployed by",
        "Commit",
        "Tag",
    ]
    rows = []
    deployments_with_dirty_commit = False
    for idx, app_stats in enumerate(resp.app_deployment_histories):
        style = "bold green" if idx == 0 else ""

        row = [
            Text(f"v{app_stats.version}", style=style),
            Text(timestamp_to_localized_str(app_stats.deployed_at, json), style=style),
            Text(app_stats.client_version, style=style),
            Text(app_stats.deployed_by, style=style),
        ]

        if app_stats.commit_info.commit_hash:
            short_hash = app_stats.commit_info.commit_hash[:7]
            if app_stats.commit_info.dirty:
                deployments_with_dirty_commit = True
                short_hash = f"{short_hash}*"
            row.append(Text(short_hash, style=style))
        else:
            row.append(None)

        if app_stats.tag:
            row.append(Text(app_stats.tag, style=style))
        else:
            row.append(None)

        rows.append(row)

    # Suppress tag information when no deployments used one
    if not any(row[-1] for row in rows):
        rows = [row[:-1] for row in rows]
        columns = columns[:-1]

    rows = sorted(rows, key=lambda x: int(str(x[0])[1:]), reverse=True)
    display_table(columns, rows, json)

    if deployments_with_dirty_commit and not json:
        rich.print("* - repo had uncommitted changes")


@app_cli.command("dashboard", no_args_is_help=True)
@synchronizer.create_blocking
async def dashboard(
    app_identifier: str = APP_IDENTIFIER,
    *,
    env: Optional[str] = ENV_OPTION,
):
    """Open an App's dashboard page in your web browser.

    **Examples:**

    Open dashboard for an app by name:

    ```
    modal app dashboard my-app
    ```

    Use a specified environment:

    ```
    modal app dashboard my-app --env dev
    ```
    """
    client = await _Client.from_env()
    app_id, _, _ = await resolve_app_identifier(app_identifier, env, client)
    url = f"https://modal.com/id/{app_id}"
    open_url_and_display(url, "App dashboard")
