import asyncio
import io
import os
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
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fd)
    read_file = os.fdopen(read_fd, "r")

    def capture_thread():
        while 1:
            line = read_file.readline()
            if not line:
                return
            callback(line, orig_writer)

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
