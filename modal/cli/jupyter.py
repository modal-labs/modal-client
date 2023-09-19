# Copyright Modal Labs 2023
import subprocess
import tempfile
from pathlib import Path

from typer import Typer

jupyter_cli = Typer(
    name="jupyter",
    no_args_is_help=True,
    help="""
    [Preview] Open a serverless Jupyter instance on Modal.

    This command is in preview and may change in the future.
    """,
)


@jupyter_cli.command(name="lab", help="Start Jupyter Lab on Modal.")
def lab():
    contents = (Path(__file__).parent / "programs" / "jupyterlab.py").read_text()

    # TODO: This is a big hack and can break for unexpected $PATH reasons. Make an actual code path
    # for correctly setting up and running a program in the CLI.
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "_main.py"
        f.write_text(contents)
        subprocess.run(["modal", "run", f])
