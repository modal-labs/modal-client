import io
import sys

import colorama
from yaspin import yaspin


class ProgressSpinner:
    """Just a wrapper around yaspin."""

    def __init__(self):
        self._spinner = None
        self._substeps = []
        self.step("Starting up...")

    def update(self, text):
        self._spinner.text = colorama.Fore.WHITE + text + colorama.Style.RESET_ALL

    def _ok_prev(self):
        num_lines = len(self._substeps)
        sys.stdout.write(f"\r\033[{num_lines}A")
        sys.stdout.write("\033[J")

        self._spinner.ok()
        for substep in self._substeps:
            substep.ok(" ")

    def substep(self, text):
        if self._substeps:
            prev_substep = self._substeps[-1]
            prev_substep.ok(" ")
        else:
            self._spinner.ok()
        text = colorama.Fore.BLUE + "\t" + text + colorama.Style.RESET_ALL
        substep = yaspin(color="blue", text=text)
        substep.start()
        self._substeps.append(substep)

    def step(self, text, prev_text=None):
        """OK the previous stage of the spinner and start a new one."""
        if self._spinner:
            if prev_text:
                self.update(prev_text)
            self._ok_prev()
            self._substeps = []
        self._spinner = yaspin(color="white", timer=True)
        self._spinner.start()
        self.update(text)

    def hidden(self):
        return self._spinner.hidden()

    def stop(self, text):
        self.update(text)
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
