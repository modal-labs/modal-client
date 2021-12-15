import io
import sys

import colorama


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
