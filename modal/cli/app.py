# Copyright Modal Labs 2022
import asyncio
from typing import Optional

import typer
from click import UsageError
from grpclib import GRPCError, Status
from rich.console import Console
from rich.table import Table

from modal.cli.environment import ENV_OPTION_HELP, ensure_env
from modal._output import OutputManager, get_app_logs_loop
from modal.cli.utils import timestamp_to_local
from modal.client import _Client
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer

app_cli = typer.Typer(name="app", help="Manage deployed and running apps.", no_args_is_help=True)


@app_cli.command("list")
@synchronizer.create_blocking
async def list(env: Optional[str] = typer.Option(default=None, help=ENV_OPTION_HELP, hidden=True)):
    """List all running or recently running Modal apps for the current account"""
    client = await _Client.from_env()
    env = ensure_env(env)
    res: api_pb2.AppListResponse = await client.stub.AppList(api_pb2.AppListRequest(environment_name=env))
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

    env_part = f" in environment '{env}'" if env else ""
    console.print(f"Listing apps{env_part}")
    console.print(table)


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
