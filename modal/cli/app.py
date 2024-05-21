# Copyright Modal Labs 2022
import time
from typing import List, Optional, Union

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.table import Column
from rich.text import Text
from typer import Argument, Option

from modal._utils.async_utils import synchronizer
from modal.app_utils import _list_apps
from modal.client import _Client
from modal.environments import ensure_env
from modal.exception import NotFoundError
from modal_proto import api_pb2

from .utils import ENV_OPTION, display_table, stream_app_logs, timestamp_to_local

app_cli = typer.Typer(name="app", help="Manage deployed and running apps.", no_args_is_help=True)

APP_STATE_TO_MESSAGE = {
    api_pb2.APP_STATE_DEPLOYED: Text("deployed", style="green"),
    api_pb2.APP_STATE_DETACHED: Text("running (detached)", style="green"),
    api_pb2.APP_STATE_DISABLED: Text("disabled", style="dim"),
    api_pb2.APP_STATE_EPHEMERAL: Text("running", style="green"),
    api_pb2.APP_STATE_INITIALIZING: Text("initializing...", style="green"),
    api_pb2.APP_STATE_STOPPED: Text("stopped", style="blue"),
    api_pb2.APP_STATE_STOPPING: Text("stopping...", style="blue"),
}


@app_cli.command("list")
@synchronizer.create_blocking
async def list(env: Optional[str] = ENV_OPTION, json: Optional[bool] = False):
    """List Modal apps that are currently deployed/running or recently stopped."""
    env = ensure_env(env)

    columns: List[Union[Column, str]] = [
        Column("App ID", min_width=25),  # Ensure that App ID is not truncated in slim terminals
        "Description",
        "State",
        "Tasks",
        "Created at",
        "Stopped at",
    ]
    rows: List[List[Union[Text, str]]] = []
    apps: List[api_pb2.AppStats] = await _list_apps(env)
    now = time.time()
    for app_stats in apps:
        if (
            # Previously, all deployed objects (Dicts, Volumes, etc.) created an entry in the App table.
            # We are waiting to roll off support for old clients before we can clean up the database.
            # Until then, we filter deployed "single-object apps" from this output based on the object entity.
            (app_stats.object_entity and app_stats.object_entity != "ap")
            # AppList always returns up to the 250 most-recently stopped apps, which is a lot for the CLI
            # (it is also used in the web interface, where apps are organized by tabs and paginated).
            # So we semi-arbitrarily limit the stopped apps to those stopped within the past 2 hours.
            or (
                app_stats.state in {api_pb2.AppState.APP_STATE_STOPPED, api_pb2.AppState.APP_STATE_DERIVED}
                and (now - app_stats.stopped_at) > (2 * 60 * 60)
            )
        ):
            continue

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
    app_id: str = Argument("", help="Look up any App by its ID"),
    *,
    name: Optional[str] = Option(None, "-n", "--name", help="Look up a deployed App by its name"),
    env: Optional[str] = ENV_OPTION,
):
    """Show App logs, streaming while active.

    **Examples:**

    Get the logs based on an app ID:

    ```bash
    modal app logs ap-123456
    ```

    Get the logs for a currently deployed App based on its name:

    ```bash
    modal app logs --name my-app
    ```

    """
    if not bool(app_id) ^ bool(name):
        raise UsageError("Must pass either an ID or a name.")

    if not app_id:

        @synchronizer.create_blocking
        async def get_app_id():
            client = await _Client.from_env()
            env_name = ensure_env(env)
            request = api_pb2.AppGetByDeploymentNameRequest(
                namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, name=name, environment_name=env_name
            )
            resp = await client.stub.AppGetByDeploymentName(request)
            if not resp.app_id:
                env_comment = f" in the '{env_name}' environment" if env_name else ""
                raise NotFoundError(f"Could not find a deployed app named '{name}'{env_comment}.")
            return resp.app_id

        try:
            app_id = get_app_id()
        except GRPCError as exc:
            if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
                raise UsageError(exc.message)
            else:
                raise

    stream_app_logs(app_id)


@app_cli.command("stop")
@synchronizer.create_blocking
async def stop(app_id: str):
    """Stop an app."""
    client = await _Client.from_env()
    req = api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI)
    await client.stub.AppStop(req)
