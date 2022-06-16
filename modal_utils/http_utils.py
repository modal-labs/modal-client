import contextlib
import socket

import aiohttp.web
import aiohttp.web_runner


@contextlib.asynccontextmanager
async def run_temporary_http_server(app: aiohttp.web.Application):
    # Allocates a random port, runs a server in a context manager
    # This is used in various tests
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    host = f"http://127.0.0.1:{port}"

    runner = aiohttp.web_runner.AppRunner(app)
    await runner.setup()
    site = aiohttp.web_runner.SockSite(runner, sock=sock)
    await site.start()
    try:
        yield host
    finally:
        await runner.cleanup()
