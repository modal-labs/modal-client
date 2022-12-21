# Copyright Modal Labs 2022
import asyncio
from typing import List, Optional, Tuple

import typer
from click import ClickException
from google.protobuf import empty_pb2
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

from modal._output import OutputManager, step_progress
from modal.cli.utils import timestamp_to_local
from modal.client import AioClient
from modal.functions import _Function
from modal.stub import _Stub
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer

DEFAULT_STUB_NAME = "stub"

app_cli = typer.Typer(name="app", help="Manage deployed and running apps.", no_args_is_help=True)


@app_cli.command(
    "run",
    help="[Moved] Run a Modal function.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run():
    raise ClickException("Use the `modal run ...` command instead (no longer nested under `app`)")


@app_cli.command("deploy", help="[Moved] Deploy a Modal stub as an application.")
def deploy(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    name: str = typer.Option(None, help="Name of the deployment."),
):
    raise ClickException("Use the `modal deploy ...` command instead (no longer nested under `app`)")


def make_function_panel(idx: int, tag: str, function: _Function, stub: _Stub) -> Panel:
    items = [
        f"- {i}"
        for i in [*function._mounts, function._image, *function._secrets, *function._shared_volumes.values()]
        if i not in [stub._client_mount, *stub._function_mounts.values()]
    ]
    if function._gpu:
        items.append("- GPU")
    return Panel(
        Markdown("\n".join(items)),
        title=f"[bright_magenta]{idx}. [/bright_magenta][bold]{tag}[/bold]",
        title_align="left",
    )


def choose_function(stub: _Stub, functions: List[Tuple[str, _Function]], console: Console):
    if len(functions) == 0:
        return None
    elif len(functions) == 1:
        return functions[0][1]

    function_panels = [make_function_panel(idx, tag, obj, stub) for idx, (tag, obj) in enumerate(functions)]

    renderable = Panel(Group(*function_panels))
    console.print(renderable)

    choice = Prompt.ask(
        "[yellow] Pick a function definition to create a corresponding shell: [/yellow]",
        choices=[str(i) for i in range(len(functions))],
        default="0",
        show_default=False,
    )

    return functions[int(choice)][1]


@app_cli.command("shell", no_args_is_help=True, help="[Moved] Start a shell session in a Modal container")
def shell(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    function_name: Optional[str] = typer.Option(
        default=None,
        help="Name of the Modal function to run. If unspecified, Modal will prompt you for a function if running in interactive mode.",
    ),
    cmd: str = typer.Option(default="/bin/bash", help="Command to run inside the Modal image."),
):
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
    except KeyboardInterrupt:
        pass


@app_cli.command("stop")
@synchronizer
async def list_stops(app_id: str):
    """Stop an app."""
    aio_client = await AioClient.from_env()
    req = api_pb2.AppStopRequest(app_id=app_id)
    await aio_client.stub.AppStop(req)


def _show_stub_ref_failure_help(import_path: str, stub_name: str) -> None:
    error_console = Console(stderr=True)
    guidance_msg = (
        (
            f"Expected to find a stub variable named **`{stub_name}`** (the default stub name). If your `modal.Stub` is named differently, "
            "you must specify it in the stub ref argument. "
            f"For example a stub variable `app_stub = modal.Stub()` in `{import_path}` would "
            f"be specified as `{import_path}::app_stub`."
        )
        if stub_name == DEFAULT_STUB_NAME
        else (
            f"Expected to find a stub variable named **`{stub_name}`**. "
            f"Check the name of your stub variable in `{import_path}`. "
            f"It should look like `{stub_name} = modal.Stub(…)`."
        )
    )
    error_console.print(f"[bold red]Could not locate stub variable in {import_path}.")
    md = Markdown(guidance_msg)
    error_console.print(md)
