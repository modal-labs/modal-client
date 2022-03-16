import asyncio
import codecs
import io
import os
import platform
import pty
import re
from asyncio import TimeoutError
from typing import Callable

from modal.config import logger
from modal_utils.async_utils import synchronizer


@synchronizer.asynccontextmanager
async def nullcapture(stream: io.IOBase):
    yield stream


@synchronizer.asynccontextmanager
async def thread_capture(stream: io.IOBase, callback: Callable[[str, io.TextIOBase], None]):
    """Intercept writes on a stream (typically stderr or stdout)"""
    fd = stream.fileno()
    dup_fd = os.dup(fd)
    orig_writer = os.fdopen(dup_fd, "w")

    if platform.system() != "Windows" and stream.isatty():
        read_fd, write_fd = pty.openpty()
    else:
        # pty doesn't work on Windows.
        # TODO: this branch has not been tested.
        read_fd, write_fd = os.pipe()

    os.dup2(write_fd, fd)

    decoder = codecs.getincrementaldecoder("utf8")()

    def capture_thread():
        buf = ""

        while 1:
            raw_data = os.read(read_fd, 5)
            if not raw_data:
                return
            data = decoder.decode(raw_data)

            # Only send back lines that end in \n or \r.
            # This is needed to make progress bars and the like work well.
            # TODO: maybe write a custom IncrementalDecoder?
            chunks = re.split("([\r\n])", buf + data)

            # chunks is guaranteed to be odd in length.
            for i in range(int(len(chunks) / 2)):
                # piece together chunk back with delimiter.
                line = chunks[2 * i] + chunks[2 * i + 1]
                callback(line, orig_writer)

            buf = chunks[-1]

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
