# Copyright Modal Labs 2022
import re
from typing import Optional, Union

import rich
import typer
from click import UsageError
from rich.table import Column
from rich.text import Text
from typer import Argument

from modal._object import _get_environment_name
from modal._utils.async_utils import synchronizer
from modal._utils.deprecation import deprecation_warning
from modal.client import _Client
from modal.environments import ensure_env
from modal_proto import api_pb2

from .utils import ENV_OPTION, display_table, get_app_id_from_name, stream_app_logs, timestamp_to_local

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


@synchronizer.create_blocking
async def get_app_id(app_identifier: str, env: Optional[str], client: Optional[_Client] = None) -> str:
    """Resolve an app_identifier that may be a name or an ID into an ID."""
    if re.match(r"^ap-[a-zA-Z0-9]{22}$", app_identifier):
        return app_identifier
    return await get_app_id_from_name.aio(app_identifier, env, client)


def warn_on_name_option(command: str, app_identifier: str, name: str) -> str:
    if name:
        message = (
            "Passing an App name using --name is deprecated;"
            " App names can now be passed directly as positional arguments:"
            f"\n\n    modal app {command} {name} ..."
        )
        deprecation_warning((2024, 8, 15), message, show_source=False)
        return name
    return app_identifier


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
                timestamp_to_local(app_stats.created_at, json),
                timestamp_to_local(app_stats.stopped_at, json),
            ]
        )

    env_part = f" in environment '{env}'" if env else ""
    display_table(columns, rows, json, title=f"Apps{env_part}")


@app_cli.command("logs", no_args_is_help=True)
def logs(
    app_identifier: str = APP_IDENTIFIER,
    *,
    name: str = NAME_OPTION,
    env: Optional[str] = ENV_OPTION,
):
    """Show App logs, streaming while active.

    **Examples:**

    Get the logs based on an app ID:

    ```
    modal app logs ap-123456
    ```

    Get the logs for a currently deployed App based on its name:

    ```
    modal app logs my-app
    ```

    """
    app_identifier = warn_on_name_option("logs", app_identifier, name)
    app_id = get_app_id(app_identifier, env)
    stream_app_logs(app_id)


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
    app_id = await get_app_id.aio(app_identifier, env, client)
    if not version:
        version_number = -1
    else:
        if m := re.match(r"v(\d+)", version):
            version_number = int(m.group(1))
        else:
            raise UsageError(f"Invalid version specifer: {version}")
    req = api_pb2.AppRollbackRequest(app_id=app_id, version=version_number)
    await client.stub.AppRollback(req)
    rich.print("[green]✓[/green] Deployment rollback successful!")


@app_cli.command("stop", no_args_is_help=True)
@synchronizer.create_blocking
async def stop(
    app_identifier: str = APP_IDENTIFIER,
    *,
    name: str = NAME_OPTION,
    env: Optional[str] = ENV_OPTION,
):
    """Stop an app."""
    app_identifier = warn_on_name_option("stop", app_identifier, name)
    client = await _Client.from_env()
    app_id = await get_app_id.aio(app_identifier, env)
    req = api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI)
    await client.stub.AppStop(req)


@app_cli.command("history", no_args_is_help=True)
@synchronizer.create_blocking
async def history(
    app_identifier: str = APP_IDENTIFIER,
    *,
    env: Optional[str] = ENV_OPTION,
    name: str = NAME_OPTION,
    json: bool = False,
):
    """Show App deployment history, for a currently deployed app

    **Examples:**

    Get the history based on an app ID:

    ```
    modal app history ap-123456
    ```

    Get the history for a currently deployed App based on its name:

    ```
    modal app history my-app
    ```

    """
    app_identifier = warn_on_name_option("history", app_identifier, name)
    env = ensure_env(env)
    client = await _Client.from_env()
    app_id = await get_app_id.aio(app_identifier, env, client)
    resp = await client.stub.AppDeploymentHistory(api_pb2.AppDeploymentHistoryRequest(app_id=app_id))

    columns = [
        "Version",
        "Time deployed",
        "Client",
        "Deployed by",
    ]
    rows = []
    deployments_with_tags = False
    deployments_with_commit_info = False
    deployments_with_dirty_commit = False
    for idx, app_stats in enumerate(resp.app_deployment_histories):
        style = "bold green" if idx == 0 else ""

        row = [
            Text(f"v{app_stats.version}", style=style),
            Text(timestamp_to_local(app_stats.deployed_at, json), style=style),
            Text(app_stats.client_version, style=style),
            Text(app_stats.deployed_by, style=style),
        ]

        if app_stats.tag:
            deployments_with_tags = True
            row.append(Text(app_stats.tag, style=style))

        if app_stats.commit_info.commit_hash:
            deployments_with_commit_info = True
            short_hash = app_stats.commit_info.commit_hash[:7]
            if app_stats.commit_info.dirty:
                deployments_with_dirty_commit = True
                short_hash = f"{short_hash}*"
            row.append(Text(short_hash, style=style))

        rows.append(row)

    if deployments_with_tags:
        columns.append("Tag")
    if deployments_with_commit_info:
        columns.append("Commit")

    rows = sorted(rows, key=lambda x: int(str(x[0])[1:]), reverse=True)
    display_table(columns, rows, json)

    if deployments_with_dirty_commit and not json:
        rich.print("* - repo had uncommitted changes")
