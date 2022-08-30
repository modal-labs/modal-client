import asyncio
import inspect
import sys
import traceback

import typer
from rich.console import Console

from modal_utils.package_utils import import_stub_by_ref

app_cli = typer.Typer(no_args_is_help=True)


@app_cli.command("deploy", help="Deploy a Modal stub as an application.")
def deploy(
    stub_ref: str = typer.Argument(..., help="Path to a Modal stub."),
    name: str = typer.Option(None, help="Name of the deployment."),
):
    try:
        stub = import_stub_by_ref(stub_ref)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    if name is None:
        name = stub.name

    res = stub.deploy(name=name)
    if inspect.iscoroutine(res):
        asyncio.run(res)

    console = Console()
    console.print(f"\nView Deployment: [magenta]https://modal.com/deployments/{name}[/magenta]")
