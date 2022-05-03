import asyncio
import contextlib
import io
import os
import platform
import random
import sys
import threading
from typing import List, Optional

from rich.console import Console

from modal._output_capture import capture, nullcapture
from modal_utils.async_utils import TaskContext, synchronizer

from ._terminfo import term_seq_str

if platform.system() == "Windows":
    default_frames = "-\\|/"
else:
    default_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Symbols:
    STEP_COMPLETED = "✓"
    SUBSTEP_COMPLETED = "🔨"


class StepState:
    def __init__(self, frames: str, ongoing_message: str, is_substep: bool, done_message: Optional[str] = None) -> None:
        self.frames = frames
        self.ongoing_message = ongoing_message
        self.done_message = done_message
        self.idx = random.randint(0, len(frames) - 1)
        self.done = False
        self.is_substep = is_substep

    def tick(self) -> None:
        self.idx = (self.idx + 1) % len(self.frames)

    def set_done(self, done_message: Optional[str] = None) -> None:
        if done_message is not None:
            self.done_message = done_message
        self.done = True

    def update_frames(self, frames: str) -> None:
        self.frames = frames
        self.idx = random.randint(0, len(frames) - 1)

    @property
    def message(self) -> Optional[str]:
        if self.done and self.done_message:
            return self.done_message
        else:
            return self.ongoing_message

    @property
    def symbol(self) -> str:
        if self.done:
            if self.is_substep:
                return "  " + Symbols.SUBSTEP_COMPLETED
            else:
                return Symbols.STEP_COMPLETED
        else:
            return self.frames[self.idx]


class NoProgress:
    def step(self, message, done_message):
        pass

    def substep(self, message, clear=True):
        pass

    def complete_substep(self, step_no, done_message):
        pass

    def is_stopped(self):
        return True

    @contextlib.contextmanager
    def suspend(self):
        yield


class ProgressSpinner:
    looptime = 1.0  # seconds

    def __init__(self, stdout: io.TextIOBase, frames: str = default_frames) -> None:
        self._console = Console(file=stdout, highlight=False)

        self._frames = frames

        self._stopped = True
        self._suspended = 0
        self._lock = threading.Lock()
        self._lines_printed = 0
        self._time_per_frame = self.looptime / len(self._frames)
        self._ongoing_steps: List[StepState] = []

    def _print(self) -> None:
        if self._lines_printed > 0 or not self._ongoing_steps:
            return

        for step in self._ongoing_steps:
            if step.is_substep and not step.done:
                self._console.print(f"{step.symbol} [blue]{step.message}[/blue]")
            else:
                self._console.print(f"{step.symbol} {step.message}")
            self._lines_printed += 1

        self._console.out(term_seq_str("cuu", 1), end="")  # move cursor up 1 line.

    def _persist_done(self) -> None:
        if len(self._ongoing_steps) == 0:
            return

        assert not self._ongoing_steps[0].is_substep
        self._ongoing_steps[0].set_done()
        with self._lock:
            self._clear()
            for step in self._ongoing_steps:
                if step.done:
                    self._console.print(f"[green]{step.symbol}[/green] {step.message}")
            self._lines_printed += 1
            self._ongoing_steps = []

    def _hide_cursor(self) -> None:
        self._console.out(term_seq_str("civis"), end="")  # cursor invisible.

    def _show_cursor(self) -> None:
        self._console.out(term_seq_str("cnorm"), end="")  # cursor reset.

    def _clear(self) -> None:
        if self._lines_printed > 1:
            self._console.out(term_seq_str("cr"), end="")  # carriage return.
            self._console.out(term_seq_str("cuu", self._lines_printed - 1), end="")  # move cursor up n lines.
            self._console.out(term_seq_str("ed"), end="")  # clear to end of display.
        if self._lines_printed >= 1:
            self._console.out(term_seq_str("el"), end="")  # erase line.

        self._lines_printed = 0

    def _tick(self) -> None:
        with self._lock:
            self._clear()
            self._print()
            for step in self._ongoing_steps:
                step.tick()

    async def _loop(self) -> None:
        try:
            self._hide_cursor()
            self._stopped = False
            self._suspended = 0
            while not self._stopped:
                if not self._suspended:
                    self._tick()
                await asyncio.sleep(self._time_per_frame)
        finally:
            self._show_cursor()

    def is_stopped(self) -> bool:
        return self._stopped

    def _stop(self) -> None:
        self._stopped = True
        self._persist_done()

    def step(self, message: str, done_message: str) -> None:
        self._persist_done()
        self._ongoing_steps = [StepState(self._frames, message, False, done_message)]

    def substep(self, message: str, clear=True) -> int:
        if clear:
            self._ongoing_steps = self._ongoing_steps[:1]

        step_no = len(self._ongoing_steps)
        assert step_no > 0

        self._ongoing_steps[0].update_frames("-")  # Make parent step spinner static.
        self._ongoing_steps.append(StepState(self._frames, message, True))
        return step_no

    def complete_substep(self, step_no, done_message):
        if step_no >= len(self._ongoing_steps):
            return
        self._ongoing_steps[step_no].set_done(done_message)

    @contextlib.contextmanager
    def suspend(self):
        with self._lock:
            if self._suspended == 0:
                self._clear()
            self._suspended += 1

        yield

        with self._lock:
            self._suspended -= 1
            if self._suspended == 0:
                self._print()


@synchronizer.asynccontextmanager
async def safe_progress(
    task_context: TaskContext,
    stdout: io.TextIOBase,
    stderr: io.TextIOBase,
    visible: bool = True,
):
    if not visible:
        yield NoProgress()
        return

    progress = None
    pending_cr = False

    def write_callback(line, out):
        nonlocal progress, pending_cr

        if progress:
            with progress.suspend():
                # `output_capture` guarantees `line` to end in `\r` or `\n`
                # Without that guarantee, we would have to know the column position of the cursor,
                # which basically requires you to emulate a terminal. (terminal save/restore doesn't work:
                # https://unix.stackexchange.com/questions/278884/save-cursor-position-and-restore-it-in-terminal/278888#278888 )
                if pending_cr:
                    out.write(f"{term_seq_str('cuu1')}\r")

                out.write(line)

                if line.endswith("\r"):
                    out.write("\n")
                    pending_cr = True
                else:
                    pending_cr = False

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
                os.system("")

            progress = ProgressSpinner(stdout)
            t = task_context.create_task(progress._loop())
            try:
                yield progress
            finally:
                progress._stop()
                await t


if __name__ == "__main__":

    async def printstuff():
        async with TaskContext(grace=1.0) as tc:
            p: ProgressSpinner
            async with safe_progress(tc, sys.stdout, sys.stderr) as p:
                p.step("Making pastaz", "Pasta done")
                p.substep("boiling water.............")
                await asyncio.sleep(1)
                print("kids start to shout")
                await asyncio.sleep(1)
                print("grandpa slips", file=sys.stderr)
                with p.suspend():
                    await asyncio.sleep(1)
                p.substep("putting pasta in water")
                await asyncio.sleep(1)
                p.substep("rinsing pasta")
                await asyncio.sleep(1)
                p.step("Making sauce", "Sauce done")
                await asyncio.sleep(1)
                p.substep("frying onions")
                await asyncio.sleep(1)
                p.substep("adding tomatoes")
                p.substep("adding tomatoes 2", clear=False)
                await asyncio.sleep(1)
                p.step("Eating", "Ate", clear=False)
                await asyncio.sleep(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(printstuff())
