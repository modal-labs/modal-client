import contextlib
import sys

import colorama
from yaspin import yaspin

from .proto import api_pb2


# contextlib.nullcontext is not present in 3.6
@contextlib.contextmanager
def nullcontext():
    yield


class ProgressSpinner:
    def __init__(self, visible):
        self._spinner = None
        self._last_tag = None
        self._substeps = {}
        self._task_states = {}

        if not sys.stdout.isatty():
            visible = False

        self._visible = visible

    # TODO: move this somewhere else?
    def update_task_state(self, task_id, state):
        self._task_states[task_id] = state

        # Recompute task status string.

        all_states = self._task_states.values()
        max_state = max(all_states)
        tasks_at_max = len(list(filter(lambda x: x == max_state, all_states)))

        if max_state == api_pb2.TaskState.CREATED:
            msg = f"Tasks created..."
        if max_state == api_pb2.TaskState.QUEUED:
            msg = f"Tasks queued..."
        elif max_state == api_pb2.TaskState.LOADING_IMAGE:
            msg = f"Loading images ({tasks_at_max} containers initializing)..."
        elif max_state == api_pb2.TaskState.RUNNING:
            tasks_loading = len(list(filter(lambda x: x == api_pb2.TaskState.LOADING_IMAGE, all_states)))
            msg = f"Running ({tasks_at_max}/{tasks_at_max + tasks_loading} containers in use)..."
        self.set_substep_text("task", msg)

    def set_substep_text(self, tag, text):
        if not self._visible:
            return

        text = colorama.Fore.BLUE + "\t" + text + colorama.Style.RESET_ALL

        if not tag in self._substeps:
            self._create_substep(tag, text)
        else:
            self._substeps[tag].text = text

    def set_step_text(self, text):
        if not self._visible:
            return

        self._spinner.text = colorama.Fore.WHITE + text + colorama.Style.RESET_ALL

    def _ok_prev(self):
        num_lines = len(self._substeps)
        if num_lines:
            # Clear multiple lines if there are substeps.
            sys.stdout.write(f"\r\033[{num_lines}A")
            sys.stdout.write("\033[J")

        if self._done_text:
            self.set_step_text(self._done_text)

        self._spinner.ok("✓")
        for substep in self._substeps.values():
            substep.stop()

    def _create_substep(self, tag, text):
        if self._substeps:
            prev_substep = self._substeps[self._last_tag]
            prev_substep.ok(" ")
        else:
            self._spinner.ok("✓")
        substep = yaspin(color="blue")

        self._last_tag = tag
        self._substeps[tag] = substep
        self._substeps[tag].text = text
        substep.start()

    def step(self, text, done_text=None):
        """OK the previous stage of the spinner and start a new one."""
        if not self._visible:
            return

        if self._spinner:
            self._ok_prev()
            self._last_tag = None
            self._substeps = {}
        self._done_text = done_text
        self._spinner = yaspin(color="white")
        self._spinner.start()
        self.set_step_text(text)

    def hidden(self):
        if not self._visible:
            return nullcontext()

        if self._last_tag:
            return self._substeps[self._last_tag].hidden()
        return self._spinner.hidden()

    def stop(self):
        if not self._visible:
            return

        self._ok_prev()
