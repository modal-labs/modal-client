# Copyright Modal Labs 2023
import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any, Optional

from typer import Argument, Option, Typer

from ..app import App
from ..exception import _CliUserExecutionError
from ..output import enable_output
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


def _launch_program(name: str, filename: str, detach: bool, args: dict[str, Any]) -> None:
    os.environ["MODAL_LAUNCH_ARGS"] = json.dumps(args)

    program_path = str(Path(__file__).parent / "programs" / filename)
    entrypoint = import_function(program_path, "modal launch")
    app: App = entrypoint.app
    app.set_description(f"modal launch {name}")

    # `launch/` scripts must have a `local_entrypoint()` with no args, for simplicity here.
    func = entrypoint.info.raw_f
    isasync = inspect.iscoroutinefunction(func)
    with enable_output():
        with run_app(app, detach=detach):
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
    detach: bool = False,  # Run the app in "detached" mode to persist after local client disconnects
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
    _launch_program("jupyter", "run_jupyter.py", detach, args)


@launch_cli.command(name="vscode", help="Start Visual Studio Code on Modal.")
def vscode(
    cpu: int = 8,
    memory: int = 32768,
    gpu: Optional[str] = None,
    image: str = "debian:12",
    timeout: int = 3600,
    mount: Optional[str] = None,  # Create a `modal.Mount` from a local directory.
    volume: Optional[str] = None,  # Attach a persisted `modal.Volume` by name (creating if missing).
    detach: bool = False,  # Run the app in "detached" mode to persist after local client disconnects
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


@launch_cli.command(name="hf-download")
def hf_download(
    repo_id: str = Argument(help="The Hugging Face repository ID"),
    volume: str = Argument(help="The name of the Modal volume to use for caching"),
    secret: Optional[str] = Option(None, help="The name of a Modal secret with Hugging Face credentials"),
    timeout: int = Option(600, help="Maximum time the download is allowed to run (in seconds)"),
    type: Optional[str] = Option(None, help="The repository type(e.g. 'model' or 'dataset')"),
    revision: Optional[str] = Option(None, help="A specific revision to download"),
    ignore: list[str] = Option([], help="Ignore patterns to skip downloading matching files"),
    allow: list[str] = Option([], help="Allow patterns to selectively download matching files"),
    detatch: bool = Option(False, "--detach", help="Allow the download to continue if the local client disconnects"),
):
    """Download a snapshot from the Hugging Face Hub.

    This command uses Hugging Face's `snapshot_download` function to download a snapshot from a repository
    on Hugging Face's Hub and cache it in a Modal volume. In your Modal applications, if you mount the
    Volume at a location corresponding to the `HF_HUB_CACHE` environment variable, Hugging Face will load
    data from the cache instead of downloading it.

    \b```
    modal launch hf-download microsoft/Phi-3-mini hf-hub-cache --secret "hf-secret" --timeout 600 --detach
    ```

    Then in your Modal App:

    \b```
    volume = modal.Volume.from_name("hf-hub-cache")
    @app.function(volumes={HF_HUB_CACHE: volume})
    def f():
        model = ModelClass.from_pretrained(model_name, cache_dir=HF_HUB_CACHE)
    ```

    """
    args = {
        "volume": volume,
        "secret": secret,
        "timeout": timeout,
        "repo_id": repo_id,
        "type": type,
        "revision": revision,
        "ignore": ignore,
        "allow": allow,
    }
    _launch_program("hf-download", "hf_download.py", detatch, args)
