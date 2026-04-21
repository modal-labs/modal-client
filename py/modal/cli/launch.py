# Copyright Modal Labs 2023
import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any, Optional

import click
import rich.panel
from rich.markdown import Markdown

from ..exception import _CliUserExecutionError
from ..output import OutputManager
from ..runner import run_app
from ._help import ModalGroup
from .import_refs import ImportRef, _get_runnable_app, import_file_or_module

launch_cli = ModalGroup(
    name="launch",
    help="[Experimental] Open a serverless app instance on Modal.",
)


def _launch_program(
    name: str, filename: str, detach: bool, args: dict[str, Any], *, description: Optional[str] = None
) -> None:
    OutputManager.get().print(
        rich.panel.Panel(
            Markdown(f"⚠️  `modal launch {name}` is **experimental** and may change in the future."),
            border_style="yellow",
        ),
    )

    os.environ["MODAL_LAUNCH_ARGS"] = json.dumps(args)

    program_path = str(Path(__file__).parent / "programs" / filename)
    base_cmd = f"modal launch {name}"
    module = import_file_or_module(ImportRef(program_path, use_module_mode=False), base_cmd=base_cmd)
    entrypoint = module.main

    app = _get_runnable_app(entrypoint)
    app.set_description(description if description else base_cmd)

    # `launch/` scripts must have a `local_entrypoint()` with no args, for simplicity here.
    func = entrypoint.info.raw_f
    isasync = inspect.iscoroutinefunction(func)
    with run_app(app, detach=detach):
        try:
            if isasync:
                asyncio.run(func())
            else:
                func()
        except Exception as exc:
            raise _CliUserExecutionError(inspect.getsourcefile(func)) from exc


@launch_cli.command("jupyter", help="Start Jupyter Lab on Modal.")
@click.option("--cpu", default=8, type=int)
@click.option("--memory", default=32768, type=int)
@click.option("--gpu", default=None)
@click.option("--timeout", default=3600, type=int)
@click.option("--image", default="ubuntu:22.04")
@click.option("--add-python", default="3.11")
@click.option("--mount", default=None, help="Adds a local directory to the jupyter container")
@click.option("--volume", default=None, help="Attach a persisted modal.Volume by name (creating if missing).")
@click.option(
    "--detach",
    is_flag=True,
    default=False,
    help="Run the app in 'detached' mode to persist after local client disconnects",
)
def jupyter(
    cpu: int = 8,
    memory: int = 32768,
    gpu: Optional[str] = None,
    timeout: int = 3600,
    image: str = "ubuntu:22.04",
    add_python: Optional[str] = "3.11",
    mount: Optional[str] = None,
    volume: Optional[str] = None,
    detach: bool = False,
):
    OutputManager.get().print(
        rich.panel.Panel(
            (
                "[link=https://modal.com/notebooks]Try Modal Notebooks! "
                "modal.com/notebooks[/link]\n"
                "Notebooks have a new UI, saved content, real-time collaboration and more."
            ),
        ),
        highlight=False,
        style="bold cyan",
    )
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
    _launch_program("jupyter", "run_jupyter.py", detach, args)


@launch_cli.command("vscode", help="Start Visual Studio Code on Modal.")
@click.option("--cpu", default=8, type=int)
@click.option("--memory", default=32768, type=int)
@click.option("--gpu", default=None)
@click.option("--image", default="debian:12")
@click.option("--timeout", default=3600, type=int)
@click.option("--mount", default=None, help="Create a modal.Mount from a local directory.")
@click.option("--volume", default=None, help="Attach a persisted modal.Volume by name (creating if missing).")
@click.option(
    "--detach",
    is_flag=True,
    default=False,
    help="Run the app in 'detached' mode to persist after local client disconnects",
)
def vscode(
    cpu: int = 8,
    memory: int = 32768,
    gpu: Optional[str] = None,
    image: str = "debian:12",
    timeout: int = 3600,
    mount: Optional[str] = None,
    volume: Optional[str] = None,
    detach: bool = False,
):
    args = {
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "image": image,
        "timeout": timeout,
        "mount": mount,
        "volume": volume,
    }
    _launch_program("vscode", "vscode.py", detach, args)
