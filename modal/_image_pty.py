import asyncio
import sys
from typing import Optional

from modal.queue import _Queue
from modal_utils.async_utils import TaskContext


async def _pty(cmd: Optional[str], queue):  # queue is an AioQueue, but mypy doesn't like that
    import os
    import pty
    import threading

    write_fd, read_fd = pty.openpty()
    os.dup2(read_fd, sys.stdin.fileno())
    writer = os.fdopen(write_fd, "wb")

    run_cmd = cmd or os.environ.get("SHELL", "sh")

    print(f"Spawning {run_cmd}. Type 'exit' to exit. ")

    threading.Thread(target=pty.spawn, args=(run_cmd,), daemon=True).start()

    while True:
        line = await queue.get()

        if line is None:
            return

        writer.write(line.encode("utf-8"))
        writer.flush()


async def image_pty(image, app, cmd=None, mounts=[], secrets=[]):
    _pty_wrapped = app.function(image=image, mounts=mounts, secrets=secrets)(_pty)
    app["queue"] = _Queue()

    async with app.run(show_progress=False) as running_app:
        queue = running_app["queue"]

        async with TaskContext(grace=0) as tc:
            tc.create_task(_pty_wrapped(cmd, queue))

            # TODO: figure out keyboard interrupts
            while True:
                loop = asyncio.get_event_loop()
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if line == "exit\n":
                    await queue.put(None)
                    return

                await queue.put(line)
