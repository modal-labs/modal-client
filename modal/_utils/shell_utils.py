# Copyright Modal Labs 2024
import asyncio
import contextlib
import errno
import os
import select
import sys
from typing import Callable, Coroutine, Optional

import rich.status

from modal._pty import raw_terminal, set_nonblocking
from modal.exception import ExecutionError, InteractiveTimeoutError

from .async_utils import TaskContext, asyncify


def write_to_fd(fd: int, data: bytes):
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    def try_write():
        try:
            nbytes = os.write(fd, data)
            loop.remove_writer(fd)
            future.set_result(nbytes)
        except OSError as e:
            if e.errno != errno.EAGAIN:
                future.set_exception(e)
                raise

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


async def connect_to_terminal(
    # Handles data read from stdin. Inputs are the stdin data and message index.
    handle_stdin: Callable[[bytes, int], Coroutine],
    # Creates a coroutine that streams data to stdout/stderr. Returns the exit status.
    stream_to_stdio: Callable[[asyncio.Event], Coroutine[None, None, int]],
    pty: bool = False,
    connecting_status: Optional[rich.status.Status] = None,
) -> None:
    """
    Connect to the current terminal by streaming data from terminal's stdin to the running process
    and streaming output from running process into terminal's stdout.

    If connecting_status is given, this function will stop the status spinner upon connection or error.
    """

    def stop_connecting_status():
        if connecting_status:
            connecting_status.stop()

    on_connect = asyncio.Event()
    async with TaskContext() as tc:
        exec_output_task = tc.create_task(stream_to_stdio(on_connect))
        try:
            # time out if we can't connect to the server fast enough
            await asyncio.wait_for(on_connect.wait(), timeout=15)
            stop_connecting_status()

            async with stream_from_stdin(handle_stdin, use_raw_terminal=pty):
                exit_status = await exec_output_task

            if exit_status != 0:
                raise ExecutionError(f"Process exited with status code {exit_status}")

        except (asyncio.TimeoutError, TimeoutError):
            stop_connecting_status()
            exec_output_task.cancel()
            raise InteractiveTimeoutError("Failed to establish connection to container. Please try again.")
