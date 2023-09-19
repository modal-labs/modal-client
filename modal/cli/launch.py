# Copyright Modal Labs 2023
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from typer import Typer

launch_cli = Typer(
    name="launch",
    no_args_is_help=True,
    help="""
    [Preview] Open a serverless app instance on Modal.

    This command is in preview and may change in the future.
    """,
)


def _launch_program(name: str, args) -> None:
    contents = (Path(__file__).parent / "programs" / name).read_text()
    contents = contents.replace("args: Any = {}", f"args: Any = {repr(args)}")

    # TODO: This is a big hack and can break for unexpected $PATH reasons. Make an actual code path
    # for correctly setting up and running a program in the CLI.
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / name
        f.write_text(contents)
        subprocess.run(["modal", "run", f])


@launch_cli.command(name="jupyter", help="Start Jupyter Lab on Modal.")
def jupyter(
    cpu: int = 8,
    memory: int = 32768,
    gpu: Optional[str] = None,
    timeout: int = 3600,
    image: str = "ubuntu:22.04",
    add_python: Optional[str] = "3.11",
):
    args = {
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "timeout": timeout,
        "image": image,
        "add_python": add_python,
    }
    _launch_program("jupyterlab.py", args)


@launch_cli.command(name="vscode", help="Start VS Code on Modal.")
def vscode(
    cpu: int = 8,
    memory: int = 32768,
    gpu: Optional[str] = None,
    timeout: int = 3600,
):
    args = {
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "timeout": timeout,
    }
    _launch_program("vscode.py", args)
