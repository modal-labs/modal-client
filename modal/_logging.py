import sys

import colorama

from .proto import api_pb2


class LogPrinter:
    def __init__(self):
        self.add_newline = False

    def feed(self, log: api_pb2.TaskLogs, stdout=None, stderr=None):
        stdout_buf = stdout or sys.stdout
        stderr_buf = stderr or sys.stderr
        if log.file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            buf = stdout_buf
            color = colorama.Fore.BLUE
        elif log.file_descriptor == api_pb2.FILE_DESCRIPTOR_STDERR:
            buf = stderr_buf
            color = colorama.Fore.RED
        elif log.file_descriptor == api_pb2.FILE_DESCRIPTOR_INFO:
            buf = stderr_buf
            color = colorama.Fore.YELLOW
        else:
            raise Exception(f"Weird file descriptor {log.file_descriptor} for log output")

        if buf.isatty():
            if self.add_newline:
                # HACK: to make partial line outputs (like when using a progress bar that uses
                # ANSI escape chars) work. If the last log line doesn't end with a newline,
                # add one manually, and take it back the next time we print something.
                # TODO: this can cause problems if there are partial lines being printed as logs, and the user is also
                # printing to stdout directly. Can be solved if we can print directly here and rely on the redirection
                # (without calling suspend()), and then add the newline logic to `write_callback`.
                stdout_buf.write("\033[A\r")
            self.add_newline = not log.data.endswith("\n")

            buf.write(color)

        buf.write(log.data)

        if buf.isatty():
            buf.write(colorama.Style.RESET_ALL)
            if self.add_newline:
                stdout_buf.write("\n")
            buf.flush()
