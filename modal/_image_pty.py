import asyncio
import sys
from typing import Optional

from modal.queue import AioQueue
from modal_utils.async_utils import TaskContext


async def _pty(cmd: Optional[str], queue: AioQueue):
    import os
    import pty
    import threading

    write_fd, read_fd = pty.openpty()
    os.dup2(read_fd, sys.stdin.fileno())
    writer = os.fdopen(write_fd, "wb")

    run_cmd = cmd or os.environ.get("SHELL", "sh")

    print(f"Spawning {run_cmd}")

    threading.Thread(target=pty.spawn, args=(run_cmd,), daemon=True).start()

    while True:
        line = await queue.get()

        if line is None:
            return

        writer.write(line.encode("ascii"))
        writer.flush()


async def image_pty(image, app, cmd=None):
    _pty_wrapped = app.function(image=image)(_pty)

    async with app.run(show_progress=False):
        queue = await AioQueue.create(app)

        async with TaskContext(grace=0) as tc:
            tc.create_task(_pty_wrapped(cmd, queue))

            try:
                while True:
                    loop = asyncio.get_event_loop()
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    await queue.put(line)
            except KeyboardInterrupt:
                # TODO: synchronicity doesn't seem to propagate KeyboardInterrupts correctly.
                await queue.put(None)
