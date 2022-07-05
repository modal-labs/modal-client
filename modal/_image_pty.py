import asyncio
import contextlib
import pty
import sys
import termios
import tty
from typing import Optional

from modal.queue import _Queue
from modal_utils.async_utils import TaskContext


@contextlib.contextmanager
def raw_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd, termios.TCSADRAIN)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


async def _pty(cmd: Optional[str], queue):  # queue is an AioQueue, but mypy doesn't like that
    import os
    import threading

    write_fd, read_fd = pty.openpty()
    os.dup2(read_fd, sys.stdin.fileno())
    writer = os.fdopen(write_fd, "wb")

    async def _read():
        while True:
            try:
                char = await queue.get()
                writer.write(char.encode("utf-8"))
                writer.flush()
            except asyncio.CancelledError:
                return

    threading.Thread(target=asyncio.run, args=(_read(),), daemon=True).start()

    run_cmd = cmd or os.environ.get("SHELL", "sh")

    print(f"Spawning {run_cmd}. Type 'exit' to exit. ")

    pty.spawn(run_cmd)
    writer.close()


async def image_pty(image, app, cmd=None, mounts=[], secrets=[], shared_volumes={}):
    _pty_wrapped = app.function(image=image, mounts=mounts, secrets=secrets, shared_volumes=shared_volumes)(_pty)
    app["queue"] = _Queue()

    async with app.run(show_progress=False) as running_app:
        queue = running_app["queue"]

        async def _write():
            while True:
                char = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.read, 1)
                await queue.put(char)

        async with TaskContext(grace=0) as tc:
            tc.create_task(_write())

            # TODO: figure out keyboard interrupts
            with raw_terminal():
                await _pty_wrapped(cmd, queue)
