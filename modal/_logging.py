import re
import sys

import colorama

from modal_proto import api_pb2


class LogPrinter:
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
            colored_chunks = [color + chunk + colorama.Style.RESET_ALL for chunk in re.split("(\r\n|\r|\n)", log.data)]
            output = "".join(colored_chunks)
        else:
            output = log.data

        buf.write(output)
        buf.flush()
