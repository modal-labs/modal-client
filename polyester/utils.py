import io
import sys

import colorama
from yaspin import yaspin


class ProgressSpinner:
    """Just a wrapper around yaspin."""

    def __init__(self):
        self._spinner = None
        self._last_tag = None
        self._substeps = {}

    def set_substep_text(self, tag, text):
        text = colorama.Fore.BLUE + "\t" + text + colorama.Style.RESET_ALL

        if not tag in self._substeps:
            self._create_substep(tag, text)
        else:
            self._substeps[tag].text = text

    def set_step_text(self, text):
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
            substep.ok(" ")

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
        if self._spinner:
            self._ok_prev()
            self._last_tag = None
            self._substeps = {}
        self._done_text = done_text
        self._spinner = yaspin(color="white", timer=True)
        self._spinner.start()
        self.set_step_text(text)

    def hidden(self):
        if self._last_tag:
            return self._substeps[self._last_tag].hidden()
        return self._spinner.hidden()

    def stop(self):
        self._ok_prev()


def get_buffer(handle):
    # HACK: Jupyter notebooks have sys.stdout point to an OutStream object,
    # which doesn't have a buffer attribute.
    if hasattr(handle, "buffer"):
        return handle.buffer
    else:
        return handle


def print_logs(output: bytes, fd: str, stdout=None, stderr=None):
    if fd == "stdout":
        buf = stdout or get_buffer(sys.stdout)
        color = colorama.Fore.BLUE.encode()
    elif fd == "stderr":
        buf = stderr or get_buffer(sys.stderr)
        color = colorama.Fore.RED.encode()
    elif fd == "server":
        buf = stderr or get_buffer(sys.stderr)
        color = colorama.Fore.YELLOW.encode()
    else:
        raise Exception(f"weird fd {fd} for log output")

    if buf.isatty():
        buf.write(color)

    if isinstance(buf, (io.RawIOBase, io.BufferedIOBase)):
        buf.write(output)
    else:
        buf.write(output.decode("utf-8"))

    if buf.isatty():
        buf.write(colorama.Style.RESET_ALL.encode())
        buf.flush()
