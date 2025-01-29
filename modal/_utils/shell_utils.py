# Copyright Modal Labs 2024

import asyncio
import contextlib
import errno
import os
import select
import shutil
import signal
import sys
import threading
from collections.abc import Coroutine
from types import FrameType
from typing import Callable, Optional

from modal._pty import raw_terminal, set_nonblocking

from .async_utils import asyncify


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


class WindowSizeHandler:
    """Handles terminal window resize events."""

    def __init__(self):
        """Initialize window size handler. It is a system limitation that we can only set signals from the main thread.

        In case this is invoked from a thread that is not the main thread, e.g. in tests, the context manager
        becomes a no-op.
        """
        self._is_main_thread = threading.current_thread() is threading.main_thread()

        self._current_size: tuple[int, int] = self._get_terminal_size()
        self._size_changed = asyncio.Event()

        if self._is_main_thread and hasattr(signal, "SIGWINCH"):
            signal.signal(signal.SIGWINCH, self._signal_resize_event)

    def _get_terminal_size(self) -> tuple[int, int]:
        c, r = shutil.get_terminal_size()
        return r, c

    def _signal_resize_event(self, signum: Optional[int] = None, frame: Optional[FrameType] = None) -> None:
        """Custom signal handler for SIGWINCH."""
        try:
            self._current_size = self._get_terminal_size()
            self._size_changed.set()
        except Exception:
            # ignore failed window size reads
            pass

    @contextlib.asynccontextmanager
    async def watch_window_size(self, handler: Callable[[int, int], Coroutine]):
        """Context manager that processes window resize events from the queue.

        Can be run from any thread. If the window manager was initialized from a thread that is not the main thread,
        e.g. in tests, this context manager is a no-op.
        """
        if not self._is_main_thread:
            yield
            return

        async def process_events():
            while True:
                await self._size_changed.wait()
                await handler(*self._current_size)
                self._size_changed.clear()

        event_task = asyncio.create_task(process_events())
        try:
            yield
        finally:
            event_task.cancel()
