# Copyright Modal Labs 2022
import asyncio
from typing import Optional

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.table import Table

from modal.cli.environment import ENV_OPTION_HELP
from modal.config import config
from modal._output import OutputManager, get_app_logs_loop
from modal.cli.utils import timestamp_to_local
from modal.client import AioClient, _Client
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer

app_cli = typer.Typer(name="app", help="Manage deployed and running apps.", no_args_is_help=True)


@app_cli.command("list")
@synchronizer.create_blocking
async def list_apps(env: Optional[str] = typer.Option(default=None, help=ENV_OPTION_HELP)):
    """List all running or recently running Modal apps for the current account"""
    if env is None:
        env = config.get("environment")

    aio_client = await AioClient.from_env()
    res: api_pb2.AppListResponse = await aio_client.stub.AppList(api_pb2.AppListRequest(environment_name=env))
    console = Console()

    table = Table("App ID", "Description", "State", "Creation time", "Stop time")
    for app_stats in res.apps:
        if app_stats.state == api_pb2.AppState.APP_STATE_DETACHED:
            state = "[green]running (detached)[/green]"
        elif app_stats.state == api_pb2.AppState.APP_STATE_EPHEMERAL:
            state = "[green]running[/green]"
        elif app_stats.state == api_pb2.AppState.APP_STATE_INITIALIZING:
            state = "[green]initializing...[/green]"
        elif app_stats.state == api_pb2.AppState.APP_STATE_DEPLOYED:
            state = "[green]deployed[/green]"
        elif app_stats.state == api_pb2.AppState.APP_STATE_STOPPING:
            state = "[blue]stopping...[/blue]"
        elif app_stats.state == api_pb2.AppState.APP_STATE_STOPPED:
            state = "[blue]stopped[/blue]"
        else:
            state = "[grey]unknown[/grey]"

        table.add_row(
            app_stats.app_id,
            app_stats.description,
            state,
            timestamp_to_local(app_stats.created_at),
            timestamp_to_local(app_stats.stopped_at),
        )

    console.print(f"Listing apps in environment '{res.environment_name}'")
    console.print(table)


@app_cli.command("logs")
def app_logs(app_id: str):
    """Output logs for a running app."""

    @synchronizer.create_blocking
    async def sync_command():
        aio_client = await _Client.from_env()
        output_mgr = OutputManager(None, None, "Tailing logs for {app_id}")
        try:
            with output_mgr.show_status_spinner():
                await get_app_logs_loop(app_id, aio_client, output_mgr)
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
    aio_client = await AioClient.from_env()
    req = api_pb2.AppStopRequest(app_id=app_id)
    await aio_client.stub.AppStop(req)
