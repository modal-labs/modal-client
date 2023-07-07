# Copyright Modal Labs 2022
import asyncio
import contextlib
import functools
import os
import platform
import select
import sys
import traceback
from typing import Optional, Tuple, no_type_check

import rich

from modal.queue import _QueueHandle
from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext, asyncify

from .exception import InvalidError


def get_winsz(fd) -> Tuple[Optional[int], Optional[int]]:
    try:
        import fcntl
        import struct
        import termios

        return struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))  # type: ignore
    except Exception:
        return None, None


def set_winsz(fd, rows, cols):
    try:
        import fcntl
        import struct
        import termios

        s = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, s)
    except Exception:
        pass


def set_nonblocking(fd: int):
    import fcntl

    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


@contextlib.contextmanager
def raw_terminal():
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd, termios.TCSADRAIN)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


@no_type_check
def _pty_spawn(pty_info: api_pb2.PTYInfo, fn, args, kwargs):
    """Modified from pty.spawn, so we can set the window size on the forked FD
    and run a custom function in the forked child process."""

    import pty
    import tty

    pid, master_fd = pty.fork()
    if pid == pty.CHILD:
        try:
            res = fn(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            os._exit(1)
        if res is not None:
            print(
                "Return values from interactive functions are currently ignored. Ignoring result of %s." % fn.__name__
            )
        os._exit(0)

    if pty_info.winsz_rows or pty_info.winsz_cols:
        set_winsz(master_fd, pty_info.winsz_rows, pty_info.winsz_cols)

    try:
        mode = tty.tcgetattr(pty.STDIN_FILENO)
        tty.setraw(pty.STDIN_FILENO)
        restore = 1
    except tty.error:  # This is the same as termios.error
        restore = 0
    try:
        pty._copy(master_fd, pty._read, pty._read)
    except OSError:
        if restore:
            tty.tcsetattr(pty.STDIN_FILENO, tty.TCSAFLUSH, mode)

    os.close(master_fd)
    return os.waitpid(pid, 0)[1]


def run_in_pty(fn, queue, pty_info: api_pb2.PTYInfo):
    import pty
    import threading

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        write_fd, read_fd = pty.openpty()
        os.dup2(read_fd, sys.stdin.fileno())
        writer = os.fdopen(write_fd, "wb")

        def _read():
            while True:
                try:
                    char = queue.get()
                    if char is None:
                        return
                    writer.write(char)
                    writer.flush()
                except asyncio.CancelledError:
                    return

        t = threading.Thread(target=_read, daemon=True)
        t.start()

        os.environ.update(
            {"TERM": pty_info.env_term, "COLORTERM": pty_info.env_colorterm, "TERM_PROGRAM": pty_info.env_term_program}
        )

        _pty_spawn(pty_info, fn, args, kwargs)
        queue.put(None)
        t.join()
        writer.close()

    return wrapped_fn


def get_pty_info() -> api_pb2.PTYInfo:
    rows, cols = get_winsz(sys.stdin.fileno())
    return api_pb2.PTYInfo(
        enabled=True,
        winsz_rows=rows,
        winsz_cols=cols,
        env_term=os.environ.get("TERM"),
        env_colorterm=os.environ.get("COLORTERM"),
        env_term_program=os.environ.get("TERM_PROGRAM"),
    )


@contextlib.asynccontextmanager
async def write_stdin_to_pty_stream(queue: _QueueHandle):
    if platform.system() == "Windows":
        raise InvalidError("Interactive mode is not currently supported on Windows.")

    quit_pipe_read, quit_pipe_write = os.pipe()

    set_nonblocking(sys.stdin.fileno())

    @asyncify
    def _read_char() -> Optional[bytes]:
        nonlocal quit_pipe_read
        # TODO: Windows support.
        (readable, _, _) = select.select([sys.stdin.buffer, quit_pipe_read], [], [])
        if quit_pipe_read in readable:
            return None
        return sys.stdin.buffer.read()

    async def _write():
        while True:
            char = await _read_char()
            if char is None:
                return
            await queue.put(char)

    async with TaskContext(grace=0.1) as tc:
        write_task = tc.create_task(_write())
        with raw_terminal():
            yield
        os.write(quit_pipe_write, b"\n")
        write_task.cancel()


def exec_cmd(cmd: str = None):
    run_cmd = cmd or os.environ.get("SHELL", "sh")

    rich.print(f"[yellow]Spawning [bold]{run_cmd}[/bold][/yellow]")

    # TODO: support args.
    argv = [run_cmd]

    os.execvp(argv[0], argv)
