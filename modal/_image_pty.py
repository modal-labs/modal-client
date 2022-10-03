import asyncio
import contextlib
import os
import select
import sys
from typing import Optional, Tuple, no_type_check

import modal
from modal.queue import _Queue
from modal_utils.async_utils import TaskContext


def get_winsz(fd):
    try:
        import fcntl
        import struct
        import termios

        cr = struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))  # type: ignore
        return cr
    except Exception:
        return None


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
    cmd: Optional[str], queue: modal.queue._QueueHandle, winsz: Optional[Tuple[int, int]], term_env: dict
):  # queue is an AioQueue, but mypy doesn't like that
    import pty
    import threading
    import tty

    @no_type_check
    def spawn(argv, master_read=pty._read, stdin_read=pty._read):
        """Fork of pty.spawn, so we can set the window size on the forked FD"""

        if not isinstance(argv, tuple):
            argv = (argv,)

        pid, master_fd = pty.fork()
        if pid == pty.CHILD:
            os.execlp(argv[0], *argv)

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
            pty._copy(master_fd, master_read, stdin_read)
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
                writer.write(char.encode("utf-8"))
                writer.flush()
            except asyncio.CancelledError:
                return

    run_cmd = cmd or os.environ.get("SHELL", "sh")

    print(f"Spawning {run_cmd}. Type 'exit' to exit. ")

    # TODO use TaskContext and async task for this (runs into a weird synchroncity error on exit for now).
    t = threading.Thread(target=asyncio.run, args=(_read(),))
    t.start()

    for key, value in term_env.items():
        if value is not None:
            os.environ[key] = value

    spawn(run_cmd)
    await queue.put(None)
    t.join()
    writer.close()


async def image_pty(image, app, cmd=None, **kwargs):
    _pty_wrapped = app.function(image=image, **kwargs)(_pty)
    app["queue"] = _Queue()

    async with app.run(show_progress=False) as running_app:
        queue = running_app["queue"]
        quit_pipe_read, quit_pipe_write = os.pipe()

        def _read_char() -> Optional[str]:
            nonlocal quit_pipe_read
            # TODO: Windows support.
            (readable, _, _) = select.select([sys.stdin, quit_pipe_read], [], [])
            if quit_pipe_read in readable:
                return None
            return sys.stdin.read(1)

        async def _write():
            await queue.put("\n")
            while True:
                char = await asyncio.get_event_loop().run_in_executor(None, _read_char)
                if char is None:
                    return
                await queue.put(char)

        async with TaskContext(grace=0.1) as tc:
            write_task = tc.create_task(_write())
            winsz = get_winsz(sys.stdin.fileno())
            term_env = {
                "TERM": os.environ.get("TERM"),
                "COLORTERM": os.environ.get("COLORTERM"),
                "TERM_PROGRAM": os.environ.get("TERM_PROGRAM"),
            }

            with raw_terminal():
                await _pty_wrapped(cmd, queue, winsz, term_env)
            os.write(quit_pipe_write, b"\n")
            write_task.cancel()
