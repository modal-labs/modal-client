# Copyright Modal Labs 2024

import asyncio
import contextlib
import errno
import os
import select
import sys
from collections.abc import Coroutine
from typing import Callable, Optional

from .async_utils import asyncify

# =============================================================================
# Low-level terminal utilities
# =============================================================================


def get_winsz(fd=None) -> tuple[Optional[int], Optional[int]]:
    """Get the window size of a terminal."""
    try:
        if fd is None:
            fd = sys.stdin.fileno()

        import fcntl
        import struct
        import termios

        return struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))  # type: ignore
    except Exception:
        return None, None


def set_nonblocking(fd: int):
    """Set a file descriptor to non-blocking mode."""
    import fcntl

    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


@contextlib.contextmanager
def raw_terminal():
    """Context manager that puts the terminal in raw mode."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd, termios.TCSADRAIN)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# =============================================================================
# Async shell utilities
# =============================================================================


def write_to_fd(fd: int, data: bytes):
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    def try_write():
        nonlocal data
        try:
            nbytes = os.write(fd, data)
            data = data[nbytes:]
            if not data:
                loop.remove_writer(fd)
                future.set_result(None)
        except OSError as e:
            if e.errno == errno.EAGAIN:
                # Wait for the next write notification
                return
            # Fail if it's not EAGAIN
            loop.remove_writer(fd)
            future.set_exception(e)

    loop.add_writer(fd, try_write)
    return future


@contextlib.asynccontextmanager
async def stream_from_stdin(handle_input: Callable[[bytes, int], Coroutine], use_raw_terminal=False):
    """Stream from terminal stdin to the handle_input provided by the method"""
    quit_pipe_read, quit_pipe_write = os.pipe()

    set_nonblocking(sys.stdin.fileno())

    @asyncify
    def _read_stdin() -> Optional[bytes]:
        nonlocal quit_pipe_read
        # TODO: Windows support.
        (readable, _, _) = select.select([sys.stdin.buffer, quit_pipe_read], [], [], 5)
        if quit_pipe_read in readable:
            return None
        if sys.stdin.buffer in readable:
            return sys.stdin.buffer.read()
        # we had 5 seconds of no input. send an empty string as a "heartbeat" to the server.
        return b""

    async def _write():
        message_index = 1
        while True:
            data = await _read_stdin()
            if data is None:
                return

            await handle_input(data, message_index)

            message_index += 1

    write_task = asyncio.create_task(_write())

    if use_raw_terminal:
        with raw_terminal():
            yield
    else:
        yield
    os.write(quit_pipe_write, b"\n")
    write_task.cancel()
