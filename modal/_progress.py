import asyncio
import contextlib
import io
import random
import sys
import threading
from typing import List, Optional, Tuple

import colorama  # TODO: maybe use _terminfo for this

from modal._output_capture import can_capture, nullcapture, thread_capture
from modal_utils.async_utils import TaskContext, synchronizer

from ._terminfo import term_seq_str

default_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Symbols:
    DONE = "✓"
    ONGOING = "-"


class StepState:
    def __init__(self, frames: str, message: str, is_substep: bool):
        self.frames = frames
        self.message = message
        self.idx = random.randint(0, len(frames) - 1)
        self.done = False
        self.is_substep = is_substep

    def tick(self):
        self.idx = (self.idx + 1) % len(self.frames)

    def set_done(self, new_message):
        self.done = True
        self.message = new_message

    def value(self):
        if self.done:
            return Symbols.DONE
        else:
            return self.frames[self.idx]


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

        self._stopped = True
        self._suspended = 0
        self._lock = threading.Lock()
        self._lines_printed = 0
        self._time_per_frame = self.looptime / len(self._frames)

        self.colors = {
            "status": colorama.Fore.BLUE,
            "success": colorama.Fore.GREEN,
            "reset": colorama.Style.RESET_ALL,
        }
        if not use_color:
            self.colors = {k: "" for k in self.colors.keys()}

        self._active_step: Optional[Tuple[str, str]] = None
        self._ongoing_parent_step: Optional[str] = None

        self._ongoing_steps: List[StepState] = []

    def _print(self):
        if self._lines_printed > 0:
            return

        self._stdout.write(self.colors["reset"])

        if self._ongoing_parent_step:
            self._lines_printed += 1
            self._stdout.write(f"{Symbols.ONGOING} {self._ongoing_parent_step}\n")

        for step in self._ongoing_steps:
            self._stdout.write(f"{step.value()} {self.colors['status']}{step.message}{self.colors['reset']}\n")
            self._lines_printed += 1

        self._stdout.write(term_seq_str("cuu", 1))  # move cursor up 1 line.
        self._stdout.flush()

    def _persist_done(self, final_message):
        with self._lock:
            self._clear()
            self._stdout.write(f"{self.colors['success']}{Symbols.DONE}{self.colors['reset']} {final_message}\n")
            for step in self._ongoing_steps:
                if step.done:
                    self._stdout.write(f"  {self.colors['success']}🔨{self.colors['reset']} {step.message}\n")
            self._lines_printed += 1
            self._active_step = None
            self._ongoing_parent_step = None

    def _hide_cursor(self):
        self._stdout.write(term_seq_str("civis"))  # cursor invisible.
        self._stdout.flush()

    def _show_cursor(self):
        self._stdout.write(term_seq_str("cvvis"))  # cursor visible.
        self._stdout.flush()

    def _clear(self):
        if self._lines_printed > 1:
            self._stdout.write(term_seq_str("cr"))  # carriage return.
            self._stdout.write(term_seq_str("cuu", self._lines_printed - 1))  # move cursor up n lines.
            self._stdout.write(term_seq_str("ed"))  # clear to end of display.
        if self._lines_printed >= 1:
            self._stdout.write(term_seq_str("el"))  # erase line.
            self._stdout.flush()

        self._lines_printed = 0

    def _tick(self):
        with self._lock:
            self._clear()
            self._print()
            for step in self._ongoing_steps:
                step.tick()

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

        self._ongoing_steps = [StepState(self._frames, status)]
        self._active_step = (status, completion_status)

    def set_substep_text(self, status, clear=True):
        if self._active_step and not self._ongoing_parent_step:
            self._ongoing_parent_step = self._active_step[0]
            self._ongoing_steps = []

        if clear:
            self._ongoing_steps = []

        step_no = len(self._ongoing_steps)
        self._ongoing_steps.append(StepState(self._frames, status))
        return step_no

    def set_substep_done(self, step_no, new_message):
        if step_no >= len(self._ongoing_steps):
            return
        self._ongoing_steps[step_no].set_done(new_message)

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
async def safe_progress(task_context, stdout, stderr, visible=True):
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
    if (stdout is None or stdout == sys.stdout) and can_capture(sys.stdout):
        capstdout = thread_capture(sys.stdout, write_callback)
    else:
        capstdout = nullcapture(stdout or sys.stdout)

    if (stderr is None or stderr == sys.stderr) and can_capture(sys.stderr):
        capstderr = thread_capture(sys.stderr, write_callback)
    else:
        capstderr = nullcapture(stderr or sys.stderr)

    async with capstdout as stdout:
        async with capstderr as stderr:
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
            async with safe_progress(tc, sys.stdout, sys.stderr) as p:
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
