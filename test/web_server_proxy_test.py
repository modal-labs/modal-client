# Copyright Modal Labs 2024
import asyncio
import contextlib
import pytest
import socket
from dataclasses import dataclass
from typing import List

import pytest_asyncio
from aiohttp.web import Application
from aiohttp.web_runner import AppRunner, SockSite

import modal._asgi

# TODO: add more tests


@dataclass
class DummyHttpServer:
    host: str
    port: int
    event: asyncio.Event
    assertion_log: List[str]


@contextlib.asynccontextmanager
async def run_temporary_http_server(app: Application):
    # Allocates a random port, runs a server in a context manager
    sock = socket.socket()
    host = "127.0.0.1"
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    runner = AppRunner(app)
    await runner.setup()
    site = SockSite(runner, sock=sock)
    await site.start()
    try:
        yield host, port
    finally:
        await runner.cleanup()


@pytest_asyncio.fixture()
async def http_dummy_server():
    from aiohttp import web

    assertion_log = []
    event = asyncio.Event()

    async def hello(request):
        assertion_log.append("request")
        try:
            await request.read()
        except asyncio.CancelledError:
            assertion_log.append("cancelled")
            raise
        except OSError:
            # disconnect
            assertion_log.append("disconnect")
            event.set()
            return

        return web.Response(text="Hello, world")

    app = web.Application()
    app.add_routes(([web.post("/", hello)]))
    async with run_temporary_http_server(app) as (host, port):
        yield DummyHttpServer(host=host, port=port, event=event, assertion_log=assertion_log)


@pytest.mark.asyncio
async def test_web_server_wrapper_immediate_disconnect(http_dummy_server: DummyHttpServer):
    proxy_asgi_app = modal._asgi.web_server_proxy(http_dummy_server.host, http_dummy_server.port)

    async def recv():
        return {"type": "http.disconnect"}

    async def send(msg):
        print("msg", msg)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": []}
    await proxy_asgi_app(scope, recv, send)
    await http_dummy_server.event.wait()
    assert http_dummy_server.assertion_log == ["request", "disconnect"]
