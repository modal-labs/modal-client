# Copyright Modal Labs 2023
import subprocess
import tempfile
from pathlib import Path

from typer import Typer

launch_cli = Typer(
    name="launch",
    no_args_is_help=True,
    help="""
    [Preview] Open a serverless app instance on Modal.

    This command is in preview and may change in the future.
    """,
)


def _launch_program(name: str) -> None:
    contents = (Path(__file__).parent / "programs" / name).read_text()

    # TODO: This is a big hack and can break for unexpected $PATH reasons. Make an actual code path
    # for correctly setting up and running a program in the CLI.
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "_main.py"
        f.write_text(contents)
        subprocess.run(["modal", "run", f])


@launch_cli.command(name="jupyter", help="Start Jupyter Lab on Modal.")
def jupyter():
    _launch_program("jupyterlab.py")


@launch_cli.command(name="vscode", help="Start VS Code on Modal.")
def vscode():
    _launch_program("vscode.py")
