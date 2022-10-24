# Copyright Modal Labs 2022
import asyncio
import contextlib
import os
import select
import sys
from typing import Optional, Tuple, no_type_check

import modal
from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext


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


async def _pty(
    fn, queue: modal.queue._QueueHandle, winsz: Optional[Tuple[int, int]], term_env: dict
):  # queue is an AioQueue, but mypy doesn't like that
    import pty
    import tty

    @no_type_check
    def spawn(fn, *args, **kwargs):
        """Modified from pty.spawn, so we can set the window size on the forked FD
        and run a custom function in the forked child process."""

        pid, master_fd = pty.fork()
        if pid == pty.CHILD:
            fn(*args, **kwargs)

        if winsz:
            rows, cols = winsz
            set_winsz(master_fd, rows, cols)

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

    write_fd, read_fd = pty.openpty()
    os.dup2(read_fd, sys.stdin.fileno())
    writer = os.fdopen(write_fd, "wb")

    async def _read():
        while True:
            try:
                char = await queue.get()
                if char is None:
                    return
                writer.write(char)
                writer.flush()
            except asyncio.CancelledError:
                return

    run_cmd = cmd or os.environ.get("SHELL", "sh")

    print(f"Spawning {run_cmd}. Type 'exit' to exit. ")

    async with TaskContext(grace=0.01) as tc:
        t = tc.create_task(_read())

        for key, value in term_env.items():
            if value is not None:
                os.environ[key] = value

        await asyncio.get_event_loop().run_in_executor(None, spawn, run_cmd)
        await queue.put(None)
        t.cancel()
        writer.close()


def _set_nonblocking(fd: int):
    import fcntl

    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


def get_pty_info(queue_id: str) -> api_pb2.PTYInfo:
    rows, cols = get_winsz(sys.stdin.fileno())
    return api_pb2.PTYInfo(
        queue_id=queue_id,
        winsz_rows=rows,
        winsz_cols=cols,
        env_term=os.environ.get("TERM"),
        env_colorterm=os.environ.get("COLORTERM"),
        env_term_program=os.environ.get("TERM_PROGRAM"),
    )


@contextlib.contextmanager
async def image_pty(image, app, cmd=None, **kwargs):
    queue = running_app["queue"]
    quit_pipe_read, quit_pipe_write = os.pipe()

    _set_nonblocking(sys.stdin.fileno())

    def _read_char() -> Optional[bytes]:
        nonlocal quit_pipe_read
        # TODO: Windows support.
        (readable, _, _) = select.select([sys.stdin.buffer, quit_pipe_read], [], [])
        if quit_pipe_read in readable:
            return None
        return sys.stdin.buffer.read()

    async def _write():
        await queue.put(b"\n")
        while True:
            char = await asyncio.get_event_loop().run_in_executor(None, _read_char)
            if char is None:
                return
            await queue.put(char)

    async with TaskContext(grace=0.1) as tc:
        write_task = tc.create_task(_write())
        with raw_terminal():
            yield
        os.write(quit_pipe_write, b"\n")
        write_task.cancel()
