import colorama
import sys


def print_logs(output: bytes, fd: str, stdout=None, stderr=None):
    if fd == "stdout":
        buf = stdout or sys.stdout.buffer
        color = colorama.Fore.BLUE.encode()
    elif fd == "stderr":
        buf = stderr or sys.stderr.buffer
        color = colorama.Fore.RED.encode()
    elif fd == "server":
        buf = sys.stderr.buffer
        color = colorama.Fore.YELLOW.encode()
    else:
        raise Exception("weird fd for log output")
    if buf.isatty():
        buf.write(color)
    buf.write(output)
    if buf.isatty():
        buf.write(colorama.Style.RESET_ALL.encode())
        buf.flush()
