import sys

import colorama

from .proto import api_pb2


def print_logs(output: str, fd, stdout=None, stderr=None):
    if fd == api_pb2.FILE_DESCRIPTOR_STDOUT:
        buf = stdout or sys.stdout
        color = colorama.Fore.BLUE
    elif fd == api_pb2.FILE_DESCRIPTOR_STDERR:
        buf = stderr or sys.stderr
        color = colorama.Fore.RED
    elif fd == api_pb2.FILE_DESCRIPTOR_INFO:
        buf = stderr or sys.stderr
        color = colorama.Fore.YELLOW
    else:
        raise Exception(f"weird fd {fd} for log output")

    if buf.isatty():
        buf.write(color)

    buf.write(output)

    if buf.isatty():
        buf.write(colorama.Style.RESET_ALL)
        buf.flush()
