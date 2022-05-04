import re
import sys
from typing import TextIO

from rich.color import Color

from modal_proto import api_pb2


def escape_color_code(color: Color) -> str:
    return "".join(["\033[", *color.get_ansi_codes(), "m"])


def print_log(log: api_pb2.TaskLogs, stdout: TextIO, stderr: TextIO) -> None:
    stdout_buf = stdout or sys.stdout
    stderr_buf = stderr or sys.stderr

    if log.file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
        buf = stdout_buf
        color = Color.parse("blue")
    elif log.file_descriptor == api_pb2.FILE_DESCRIPTOR_STDERR:
        buf = stderr_buf
        color = Color.parse("red")
    elif log.file_descriptor == api_pb2.FILE_DESCRIPTOR_INFO:
        buf = stderr_buf
        color = Color.parse("yellow")
    else:
        raise Exception(f"Weird file descriptor {log.file_descriptor} for log output")

    if buf.isatty():
        colored_chunks = [
            escape_color_code(color) + chunk + escape_color_code(Color.default())
            for chunk in re.split("(\r\n|\r|\n)", log.data)
        ]
        output = "".join(colored_chunks)
    else:
        output = log.data

    buf.write(output)
    buf.flush()
