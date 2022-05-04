import asyncio
import codecs
import contextlib
import errno
import io
import os
import platform
import re
import sys
import typing
from asyncio import TimeoutError
from typing import Callable

from modal.config import logger
from modal_utils.async_utils import synchronizer


@synchronizer.asynccontextmanager
async def nullcapture(stream: io.IOBase):
    yield stream


def can_capture_as_file(stream: typing.IO[str]):
    if platform.system() == "Windows":
        return False

    try:
        stream.fileno()
    except io.UnsupportedOperation:
        return False
    return True


class LineBufferedOutput(io.TextIOBase, typing.IO[str]):  # type: ignore
    def __init__(self, original_stream: typing.IO[str], callback: Callable[[str, typing.IO[str]], None]):
        self._original_stream = original_stream
        self._callback = callback
        self._buf = ""

    def write(self, data: str):
        chunks = re.split("(\r\n|\r|\n)", self._buf + data)

        # re.split("(<exp>)") returns the matched groups, and also the separators.
        # e.g. re.split("(+)", "a+b") returns ["a", "+", "b"].
        # This means that chunks is guaranteed to be odd in length.
        for i in range(int(len(chunks) / 2)):
            # piece together chunk back with separator.
            line = chunks[2 * i] + chunks[2 * i + 1]
            self._callback(line, self._original_stream)

        self._buf = chunks[-1]

    def flush(self):
        pass

    def finalize(self):
        if self._buf:
            self._callback(self._buf, self._original_stream)
            self._buf = ""


@synchronizer.asynccontextmanager
async def capture(stream: typing.IO[str], callback: Callable[[str, typing.IO[str]], None]):
    if not can_capture_as_file(stream):
        async with sysmonkeypatch_capture(stream, callback) as orig:
            yield orig
    else:
        async with thread_capture(stream, callback) as orig:
            yield orig


@synchronizer.asynccontextmanager
async def sysmonkeypatch_capture(stream: typing.IO[str], callback: Callable[[str, typing.IO[str]], None]):
    # used on Windows
    if stream == sys.stdout:
        orig = sys.stdout
        buf = LineBufferedOutput(sys.stdout, callback)
        with contextlib.redirect_stdout(buf):
            yield orig
        buf.finalize()
    elif stream == sys.stderr:
        orig = sys.stderr
        buf = LineBufferedOutput(sys.stderr, callback)
        with contextlib.redirect_stderr(buf):
            yield orig
        buf.finalize()
    else:
        raise ValueError("can't redirect streams other than stdout and stderr")


@synchronizer.asynccontextmanager
async def thread_capture(stream: typing.IO[str], callback: Callable[[str, typing.IO[str]], None]):
    """Intercept writes on a stream (typically stderr or stdout)"""
    fd = stream.fileno()
    dup_fd = os.dup(fd)
    orig_writer = os.fdopen(dup_fd, "w")

    assert platform.system() != "Windows"
    if stream.isatty():
        import pty

        read_fd, write_fd = pty.openpty()
    else:
        read_fd, write_fd = os.pipe()  # untested branch

    os.dup2(write_fd, fd)

    decoder = codecs.getincrementaldecoder("utf8")()

    def capture_thread():
        buffered_writer = LineBufferedOutput(orig_writer, callback)

        while 1:
            try:
                raw_data = os.read(read_fd, 50)
            except OSError as err:
                if err.errno == errno.EIO:
                    # Input/Output error - triggered on linux when the write pipe is closed
                    raw_data = b""
                else:
                    raise

            if not raw_data:
                buffered_writer.finalize()
                return
            data = decoder.decode(raw_data)
            buffered_writer.write(data)

    # start thread but don't await it
    print_task = asyncio.get_event_loop().run_in_executor(None, capture_thread)
    try:
        yield orig_writer
    finally:
        stream.flush()  # flush any remaining writes on fake output
        os.close(write_fd)  # this should trigger eof in the capture thread
        os.dup2(dup_fd, fd)  # restore stdout
        try:
            await asyncio.wait_for(print_task, 3)  # wait for thread to empty the read buffer
        except TimeoutError:
            # TODO: this doesn't actually kill the thread, but since the pipe is closed it shouldn't
            #       capture more user output and eventually end when eof is reached on the read pipe
            logger.warn("Could not empty user output buffer. Some user output might be missing at this time")
        os.close(read_fd)
