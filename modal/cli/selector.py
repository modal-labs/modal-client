# Copyright Modal Labs 2026
import contextlib
import os
import select
import sys

from rich.console import Console
from rich.live import Live
from rich.text import Text


@contextlib.contextmanager
def _cbreak_terminal():
    """Put the terminal in cbreak mode for character-by-character input."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd, termios.TCSADRAIN)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _has_pending_input(fd: int, timeout: float = 0.05) -> bool:
    """Check if *fd* has data ready within *timeout* seconds."""
    ready, _, _ = select.select([fd], [], [], timeout)
    return bool(ready)


class Selector:
    """Interactive arrow-key selector that doubles as a Rich renderable.

    Can be used standalone via :meth:`run`, or embedded in an external
    Rich ``Live`` display by passing the instance directly as a renderable
    and driving it with :meth:`move_up` / :meth:`move_down` from your own
    input loop.

    Callers are encouraged to catch all exceptions (including interrupts)
    and handle them gracefully. This element will not work on Windows, and
    CLIs should gracefully degrade to presenting a list of options and
    exiting.
    """

    def __init__(self, options: list[str], title: str = "Select an option"):
        if not options:
            raise ValueError("options must not be empty")
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise RuntimeError("Interactive selection requires a TTY")
        # Fail fast on platforms without termios (e.g. Windows).
        import termios  # noqa: F401

        self.options = options
        self.title = title
        self.selected = 0

    # -- state manipulation ---------------------------------------------------

    def move_up(self):
        self.selected = (self.selected - 1) % len(self.options)

    def move_down(self):
        self.selected = (self.selected + 1) % len(self.options)

    @property
    def value(self) -> str:
        return self.options[self.selected]

    # -- Rich renderable protocol ---------------------------------------------

    def __rich__(self) -> Text:
        text = Text()
        text.append(f"{self.title}\n\n", style="bold")
        for i, opt in enumerate(self.options):
            if i == self.selected:
                text.append("  > ", style="bold green")
                text.append(f"{opt}\n", style="bold green")
            else:
                text.append(f"    {opt}\n")
        return text

    # -- interactive input loop -----------------------------------------------

    def run(self, live: Live | None = None) -> str:
        """Run an interactive selection loop. Returns the selected option.

        If *live* is provided, the selector uses the caller's ``Live``
        context (useful when the selector is composed into a larger
        layout).  Otherwise a standalone ``Live`` is created automatically.
        """
        if live is not None:
            with _cbreak_terminal():
                return self._input_loop(live)

        console = Console()
        with _cbreak_terminal(), Live(self, console=console, refresh_per_second=30) as own_live:
            return self._input_loop(own_live)

    def _input_loop(self, live: Live) -> str:
        fd = sys.stdin.fileno()
        while True:
            b = os.read(fd, 1)
            if not b:
                return self.value
            if b == b"\x1b":
                if not _has_pending_input(fd):
                    continue
                b2 = os.read(fd, 1)
                if b2 == b"[" and _has_pending_input(fd):
                    b3 = os.read(fd, 1)
                    if b3 == b"A":
                        # `\x1b[A` sequence
                        self.move_up()
                    elif b3 == b"B":
                        # `\x1b[B` sequence
                        self.move_down()
            elif b in (b"\r", b"\n"):
                return self.value
            elif b == b"\x03":
                raise KeyboardInterrupt

            live.refresh()
