# Copyright Modal Labs 2022
import asyncio
from typing import Optional, List, Union

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.text import Text

from modal.cli.environment import ENV_OPTION_HELP, ensure_env
from modal._output import OutputManager, get_app_logs_loop
from modal.cli.utils import timestamp_to_local, display_table
from modal.client import _Client
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer

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
async def list(
    env: Optional[str] = typer.Option(default=None, help=ENV_OPTION_HELP, hidden=True), json: Optional[bool] = False
):
    """List all running or recently running Modal apps for the current account"""
    client = await _Client.from_env()
    env = ensure_env(env)
    res: api_pb2.AppListResponse = await client.stub.AppList(api_pb2.AppListRequest(environment_name=env))
    console = Console()

    column_names = ["App ID", "Name", "State", "Creation time", "Stop time"]
    rows: List[List[Union[Text, str]]] = []
    for app_stats in res.apps:
        state = APP_STATE_TO_MESSAGE.get(app_stats.state, Text("unknown", style="gray"))

        rows.append(
            [
                app_stats.app_id,
                app_stats.description,
                state,
                timestamp_to_local(app_stats.created_at),
                timestamp_to_local(app_stats.stopped_at),
            ]
        )

    env_part = f" in environment '{env}'" if env else ""
    console.print(f"Listing apps{env_part}")
    display_table(column_names, rows, json, console)


@app_cli.command("logs")
def app_logs(app_id: str):
    """Output logs for a running app."""

    @synchronizer.create_blocking
    async def sync_command():
        client = await _Client.from_env()
        output_mgr = OutputManager(None, None, "Tailing logs for {app_id}")
        try:
            with output_mgr.show_status_spinner():
                await get_app_logs_loop(app_id, client, output_mgr)
        except asyncio.CancelledError:
            pass

    try:
        sync_command()
    except GRPCError as exc:
        if exc.status in (Status.INVALID_ARGUMENT, Status.NOT_FOUND):
            raise UsageError(exc.message)
        else:
            raise
    except KeyboardInterrupt:
        pass


@app_cli.command("stop")
@synchronizer.create_blocking
async def stop(app_id: str):
    """Stop an app."""
    client = await _Client.from_env()
    req = api_pb2.AppStopRequest(app_id=app_id)
    await client.stub.AppStop(req)
