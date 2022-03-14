import asyncio
import contextlib
import io
import sys
from typing import Optional, Tuple

import colorama
from modal_utils.async_utils import TaskContext, synchronizer

from modal._output_capture import nullcapture, thread_capture

default_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Symbols:
    DONE = "✓"
    ONGOING = "-"


class NoProgress:
    def step(self, status, completion_status):
        pass

    def set_substep_text(self, status):
        pass

    @contextlib.contextmanager
    def suspend(self):
        yield


class ProgressSpinner:
    looptime = 1.0  # seconds

    def __init__(self, stdout: io.TextIOBase, frames=default_frames, use_color=True):
        self._stdout = stdout

        self._frames = frames
        self._frame_i = 0

        self._stopped = True
        self._suspended = 0
        self._lines_printed = 0
        self._time_per_frame = self.looptime / len(self._frames)
        self._status_message = ""

        self.colors = {
            "status": colorama.Fore.BLUE,
            "success": colorama.Fore.GREEN,
            "reset": colorama.Style.RESET_ALL,
        }
        if not use_color:
            self.colors = {k: "" for k in self.colors.keys()}

        self._active_step: Optional[Tuple[str, str]] = None
        self._ongoing_parent_step: Optional[str] = None

    def _step(self):
        self._frame_i = (self._frame_i + 1) % len(self._frames)

    def _print(self):
        if self._lines_printed > 0:
            return

        frame = self._frames[self._frame_i]
        if self._ongoing_parent_step:
            self._lines_printed += 1
            self._stdout.write(f"{Symbols.ONGOING} {self._ongoing_parent_step}\n")

        self._stdout.write(f"{frame} {self._status_message}\r")
        self._lines_printed += 1

        self._stdout.flush()

    def _set_status_message(self, status_message):
        self._status_message = self.colors["status"] + status_message + self.colors["reset"]

    def _persist_done(self, final_message):
        self._clear()
        self._stdout.write(f"{self.colors['success']}{Symbols.DONE}{self.colors['reset']} {final_message}\n")
        self._active_step = None
        self._ongoing_parent_step = None

    # borrowed control sequences from yaspin
    def _hide_cursor(self):
        self._stdout.write("\033[?25l")
        self._stdout.flush()

    def _show_cursor(self):
        self._stdout.write("\033[?25h")
        self._stdout.flush()

    def _clear(self):
        if self._lines_printed == 1:
            self._stdout.write("\033[K")
        elif self._lines_printed > 1:
            self._stdout.write(f"\r\033[{self._lines_printed - 1}A")
            self._stdout.write("\033[J")

        self._lines_printed = 0

    # end yaspin

    def _tick(self):
        self._clear()
        self._print()
        self._step()

    async def _loop(self):
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

    def _stop(self):
        self._stopped = True
        if self._active_step:
            self._persist_done(self._active_step[1])

    def step(self, status: str, completion_status: str):
        if self._active_step:
            self._persist_done(self._active_step[1])

        self._set_status_message(status)
        self._active_step = (status, completion_status)

    def set_substep_text(self, status):
        if self._active_step and not self._ongoing_parent_step:
            self._ongoing_parent_step = self._active_step[0]

        self._set_status_message(status)

    @contextlib.contextmanager
    def suspend(self):
        if self._suspended == 0:
            self._clear()
            self._stdout.flush()
        self._suspended += 1
        yield
        self._suspended -= 1
        if self._suspended == 0:
            self._print()


@synchronizer.asynccontextmanager
async def safe_progress(task_context, stdout, stderr, visible=True):
    if not visible:
        yield NoProgress(), stdout, stderr
        return

    progress = None

    def write_callback(line, out):
        nonlocal progress

        if not line.endswith("\n"):
            line += "\n"  # only write full lines
        if progress:
            with progress.suspend():
                out.write(line)

    # capture stdout/err unless they have been customized
    if stdout is None or stdout == sys.stdout:
        capstdout = thread_capture(sys.stdout, write_callback)
    else:
        capstdout = nullcapture(stdout)

    if stderr is None or stderr == sys.stderr:
        capstderr = thread_capture(sys.stderr, write_callback)
    else:
        capstderr = nullcapture(stderr)

    async with capstdout as stdout:
        async with capstderr as stderr:
            progress = ProgressSpinner(stdout)
            t = task_context.create_task(progress._loop())
            try:
                yield progress, stdout, stderr
            finally:
                progress._stop()
                await t


if __name__ == "__main__":

    async def printstuff():
        async with TaskContext(grace=1.0) as tc:
            async with safe_progress(tc, sys.stdout, sys.stderr) as (p, stdout, stderr):
                p.step("Making pastaz", "Pasta done")
                p.set_substep_text("boiling water.............")
                await asyncio.sleep(1)
                print("kids start to shout")
                await asyncio.sleep(1)
                print("grandpa slips", file=sys.stderr)
                with p.suspend():
                    await asyncio.sleep(1)
                p.set_substep_text("putting pasta in water")
                await asyncio.sleep(1)
                p.set_substep_text("rinsing pasta")
                await asyncio.sleep(1)
                p.step("Making sauce", "Sauce done")
                await asyncio.sleep(1)
                p.set_substep_text("frying onions")
                await asyncio.sleep(1)
                p.set_substep_text("adding tomatoes")
                await asyncio.sleep(1)
                p.step("Eating", "Ate")
                await asyncio.sleep(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(printstuff())
