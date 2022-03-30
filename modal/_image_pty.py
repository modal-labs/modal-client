import asyncio
import sys


async def image_pty(image, app, cmd=None):
    initialized = False
    writer = None

    @app.function(image=image)
    def send_line(line: bytes):
        nonlocal initialized, writer

        if not initialized:
            import os
            import pty
            import threading

            write_fd, read_fd = pty.openpty()
            os.dup2(read_fd, sys.stdin.fileno())
            writer = os.fdopen(write_fd, "wb")

            run_cmd = cmd or os.environ.get("SHELL", "sh")

            print(f"Spawning {run_cmd}")

            threading.Thread(target=lambda: pty.spawn(run_cmd), daemon=True).start()
            initialized = True

        writer.write(line)
        writer.flush()

    async with app.run():
        await send_line(b"")
        while True:
            loop = asyncio.get_event_loop()
            line = await loop.run_in_executor(None, sys.stdin.readline)
            await send_line(f"{line}".encode("ascii"))
