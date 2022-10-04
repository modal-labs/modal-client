import socket
import ssl
from typing import Optional

import certifi
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiohttp.web import Application
from aiohttp.web_runner import AppRunner, SockSite

from modal_utils.async_utils import synchronizer


def http_client_with_tls(timeout: Optional[float]) -> ClientSession:
    """Create a new HTTP client session with standard, bundled TLS certificates.

    This is necessary to prevent client issues on some system where Python does
    not come pre-installed with specific TLS certificates that are necessary to
    connect to AWS S3 bucket URLs.

    Specifically: the error "unable to get local issuer certificate" when making
    an aiohttp request.
    """
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = TCPConnector(ssl=ssl_context)
    return ClientSession(connector=connector, timeout=ClientTimeout(total=timeout))


@synchronizer.asynccontextmanager
async def run_temporary_http_server(app: Application):
    # Allocates a random port, runs a server in a context manager
    # This is used in various tests
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    host = f"http://127.0.0.1:{port}"

    runner = AppRunner(app)
    await runner.setup()
    site = SockSite(runner, sock=sock)
    await site.start()
    try:
        yield host
    finally:
        await runner.cleanup()
