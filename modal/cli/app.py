# Copyright Modal Labs 2022
import asyncio

import typer
from click import ClickException, UsageError
from google.protobuf import empty_pb2
from grpclib import GRPCError, Status
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.tree import Tree

from modal._output import OutputManager, step_progress
from modal.cli.utils import timestamp_to_local
from modal.client import AioClient
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer
from modal_utils.package_utils import DEFAULT_STUB_NAME, StubRef

app_cli = typer.Typer(name="app", help="Manage deployed and running apps.", no_args_is_help=True)


@app_cli.command(
    "run",
    help="[Moved] Run a Modal function.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run():
    raise ClickException("Use the `modal run ...` command instead (no longer nested under `app`)")


@app_cli.command(
    "deploy",
    help="[Moved] Deploy a Modal stub as an application.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def deploy():
    raise ClickException("Use the `modal deploy ...` command instead (no longer nested under `app`)")


@app_cli.command(
    "shell",
    help="[Moved] Start a shell session in a Modal container",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def shell():
    raise ClickException("Use the `modal shell ...` command instead (no longer nested under `app`)")


@app_cli.command("list")
@synchronizer
async def list_apps():
    """List all running or recently running Modal apps for the current account"""
    aio_client = await AioClient.from_env()
    res: api_pb2.AppListResponse = await aio_client.stub.AppList(empty_pb2.Empty())
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

    console.print(table)


@app_cli.command("logs")
def app_logs(app_id: str):
    """Output logs for a running app."""

    @synchronizer
    async def sync_command():
        aio_client = await AioClient.from_env()
        output_manager = OutputManager(None, None)
        tree = Tree(step_progress(f"Tailing logs for {app_id}"), guide_style="gray50")
        status_spinner = step_progress()
        tree.add(tree)
        try:
            with output_manager.ctx_if_visible(output_manager.make_live(status_spinner)):
                await output_manager.get_logs_loop(app_id, aio_client, status_spinner, "")
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
@synchronizer
async def list_stops(app_id: str):
    """Stop an app."""
    aio_client = await AioClient.from_env()
    req = api_pb2.AppStopRequest(app_id=app_id)
    await aio_client.stub.AppStop(req)


def _show_stub_ref_failure_help(stub_ref: StubRef) -> None:
    stub_name = stub_ref.stub_name
    import_path = stub_ref.file_or_module
    error_console = Console(stderr=True)
    guidance_msg = (
        (
            f"Expected to find a stub variable named **`{DEFAULT_STUB_NAME}`** (the default stub name). If your `modal.Stub` is named differently, "
            "you must specify it in the stub ref argument. "
            f"For example a stub variable `app_stub = modal.Stub()` in `{import_path}` would "
            f"be specified as `{import_path}::app_stub`."
        )
        if stub_name is None
        else (
            f"Expected to find a stub variable named **`{stub_name}`**. "
            f"Check the name of your stub variable in `{import_path}`.\n"
            f"""It should look like:

    {stub_name} = modal.Stub(…)
"""
        )
    )
    error_console.print(f"[bold red]Could not locate stub variable in {import_path}.")
    md = Markdown(guidance_msg)
    error_console.print(md)
