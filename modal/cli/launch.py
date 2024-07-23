# Copyright Modal Labs 2023
import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from typer import Typer

from ..app import App
from ..exception import _CliUserExecutionError
from ..runner import run_app
from .import_refs import import_function

launch_cli = Typer(
    name="launch",
    no_args_is_help=True,
    help="""
    [Preview] Open a serverless app instance on Modal.

    This command is in preview and may change in the future.
    """,
)


def _launch_program(name: str, filename: str, args: Dict[str, Any]) -> None:
    os.environ["MODAL_LAUNCH_ARGS"] = json.dumps(args)

    program_path = str(Path(__file__).parent / "programs" / filename)
    entrypoint = import_function(program_path, "modal launch")
    app: App = entrypoint.app
    app.set_description(f"modal launch {name}")

    # `launch/` scripts must have a `local_entrypoint()` with no args, for simplicity here.
    func = entrypoint.info.raw_f
    isasync = inspect.iscoroutinefunction(func)
    with run_app(app):
        try:
            if isasync:
                asyncio.run(func())
            else:
                func()
        except Exception as exc:
            raise _CliUserExecutionError(inspect.getsourcefile(func)) from exc


@launch_cli.command(name="jupyter", help="Start Jupyter Lab on Modal.")
def jupyter(
    cpu: int = 8,
    memory: int = 32768,
    gpu: Optional[str] = None,
    timeout: int = 3600,
    image: str = "ubuntu:22.04",
    add_python: Optional[str] = "3.11",
    mount: Optional[str] = None,  # Create a `modal.Mount` from a local directory.
    volume: Optional[str] = None,  # Attach a persisted `modal.Volume` by name (creating if missing).
):
    args = {
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "timeout": timeout,
        "image": image,
        "add_python": add_python,
        "mount": mount,
        "volume": volume,
    }
    _launch_program("jupyter", "run_jupyter.py", args)


@launch_cli.command(name="vscode", help="Start Visual Studio Code on Modal.")
def vscode(
    cpu: int = 8,
    memory: int = 32768,
    gpu: Optional[str] = None,
    timeout: int = 3600,
    mount: Optional[str] = None,  # Create a `modal.Mount` from a local directory.
    volume: Optional[str] = None,  # Attach a persisted `modal.Volume` by name (creating if missing).
):
    args = {
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "timeout": timeout,
        "mount": mount,
        "volume": volume,
    }
    _launch_program("vscode", "vscode.py", args)
