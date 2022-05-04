import contextlib
import os
import platform
import sys
from typing import AsyncIterator, Optional, TextIO

from rich.abc import RichRenderable
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from modal._output_capture import capture, nullcapture

if platform.system() == "Windows":
    default_spinner = "line"
else:
    default_spinner = "dots"


def step_progress(message: str) -> RichRenderable:
    """Returns the element to be rendered when a step is in progress."""
    return Spinner(default_spinner, Text(message, style="blue"), style="default")


def step_completed(message: str, is_substep: bool = False) -> RichRenderable:
    """Returns the element to be rendered when a step is completed."""

    STEP_COMPLETED = "✓"
    SUBSTEP_COMPLETED = "🔨"

    symbol = SUBSTEP_COMPLETED if is_substep else STEP_COMPLETED
    return f"[green]{symbol}[/green] " + message


@contextlib.asynccontextmanager
async def live_capture(
    renderable: RichRenderable,
    stdout: Optional[TextIO],
    stderr: Optional[TextIO],
) -> AsyncIterator[Live]:
    """Captures standard output and standard error during a Rich Live session.

    Returns the live session as a context manager. Note that this method allows
    ANSI escape codes to be processed, unlike Rich's default capture.
    """

    live = Live(renderable, transient=True, redirect_stdout=False, redirect_stderr=False, auto_refresh=False)

    with live:

        def write_callback(line: str, out: TextIO) -> None:
            live.stop()
            out.write(line)
            live.start()

        # capture stdout/err unless they have been customized
        if stdout is None or stdout == sys.stdout:
            capstdout = capture(sys.stdout, write_callback)
        else:
            capstdout = nullcapture(stdout or sys.stdout)

        if stderr is None or stderr == sys.stderr:
            capstderr = capture(sys.stderr, write_callback)
        else:
            capstderr = nullcapture(stderr or sys.stderr)

        async with capstdout as stdout:
            async with capstderr as stderr:
                if platform.system() == "Windows":
                    # See https://github.com/modal-labs/modal/pull/1440 for why this is necessary.
                    os.system("")
                yield live
