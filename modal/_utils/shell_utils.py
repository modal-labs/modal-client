# Copyright Modal Labs 2024
import asyncio
import contextlib
import errno
import os
from async_utils import asyncify, TaskContext
from modal._pty import get_pty_info, raw_terminal, set_nonblocking
import sys
import select
from typing import List, Optional, Callable, Coroutine
from exception import ExecutionError, InteractiveTimeoutError, NotFoundError
import rich.status

def _write_to_fd(fd: int, data: bytes):
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
async def _stream_stdin(handle_input: Callable[[bytes, int], Coroutine], use_raw_terminal=False):
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
    handle_input: Callable[[bytes, int], Coroutine], 
    handle_output: Callable[[asyncio.Event], Coroutine], 
    pty: bool = False, 
    connecting_status: Optional[rich.status.Status] = None
):
    """
    Connect to the current terminal by streaming inputs into stdin and streaming 
    stdout to handler.

    If connecting_status is given, this function will stop the status spinner upon connection or error.
    """

    def stop_connecting_status():
        if connecting_status:
            connecting_status.stop()

    on_connect = asyncio.Event()
    async with TaskContext() as tc:
        exec_output_task = tc.create_task(handle_output(on_connect))
        try:
            # time out if we can't connect to the server fast enough
            await asyncio.wait_for(on_connect.wait(), timeout=15)
            stop_connecting_status()

            async with _stream_stdin(handle_input, use_raw_terminal=pty):
                exit_status = await exec_output_task

            if exit_status != 0:
                raise ExecutionError(f"Process exited with status code {exit_status}")

        except (asyncio.TimeoutError, TimeoutError):
            stop_connecting_status()
            exec_output_task.cancel()
            raise InteractiveTimeoutError("Failed to establish connection to container.")
