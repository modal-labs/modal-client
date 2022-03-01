import sys

import colorama

from .proto import web_pb2


class LogPrinter:
    def __init__(self):
        self.add_newline = False

    def feed(self, log: web_pb2.TaskLogs, stdout=None, stderr=None):
        stdout_buf = stdout or sys.stdout
        stderr_buf = stderr or sys.stderr
        if log.file_descriptor == web_pb2.FILE_DESCRIPTOR_STDOUT:
            buf = stdout_buf
            color = colorama.Fore.BLUE
        elif log.file_descriptor == web_pb2.FILE_DESCRIPTOR_STDERR:
            buf = stderr_buf
            color = colorama.Fore.RED
        elif log.file_descriptor == web_pb2.FILE_DESCRIPTOR_INFO:
            buf = stderr_buf
            color = colorama.Fore.YELLOW
        else:
            raise Exception(f"Weird file descriptor {log.file_descriptor} for log output")

        if buf.isatty():
            buf.write(color)

        buf.write(log.data)

        if buf.isatty():
            buf.write(colorama.Style.RESET_ALL)
            buf.flush()
