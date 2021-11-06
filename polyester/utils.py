import sys

import colorama


def is_ipykernel_outstream(handle):
    return type(handle).__name__ == "OutStream"


def get_buffer(handle):
    # HACK: Jupyter notebooks have sys.stdout point to an OutStream object,
    # which doesn't have a buffer attribute.
    if is_ipykernel_outstream(handle):
        return handle
    else:
        return handle.buffer


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
        raise Exception("weird fd for log output")

    is_ipykernel = is_ipykernel_outstream(buf)

    if buf.isatty():
        buf.write(color)
    if is_ipykernel:
        buf.write(output.decode("utf-8"))
    else:
        buf.write(output)
    if buf.isatty():
        buf.write(colorama.Style.RESET_ALL.encode())
        buf.flush()
