# Copyright Modal Labs 2022
import contextlib
from typing import TYPE_CHECKING, ClassVar

# Note: importing aiohttp seems to take about 100ms, and it's not really necessarily,
# unless we need to work with blobs. So that's why we import it lazily instead.

if TYPE_CHECKING:
    from aiohttp import ClientSession
    from aiohttp.web import Application

from .async_utils import on_shutdown


def _http_client_with_tls(timeout: float | None) -> "ClientSession":
    """Create a new HTTP client session with standard, bundled TLS certificates.

    This is necessary to prevent client issues on some system where Python does
    not come pre-installed with specific TLS certificates that are necessary to
    connect to AWS S3 bucket URLs.

    Specifically: the error "unable to get local issuer certificate" when making
    an aiohttp request.
    """
    import ssl

    import certifi
    from aiohttp import ClientSession, ClientTimeout, TCPConnector

    from .proxy_support import get_proxy_url

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    connector: TCPConnector
    proxy_url = get_proxy_url(None, use_ssl=True)
    if proxy_url:
        try:
            from aiohttp_socks import ProxyConnector
        except ImportError:
            raise ImportError(
                f"A proxy is configured ({proxy_url}) but the 'aiohttp-socks' package "
                "is not installed. Install it with: pip install 'modal[api-proxy-support]'"
            ) from None
        from .proxy_support import _normalize_proxy_url

        normalized_url, rdns = _normalize_proxy_url(proxy_url)
        connector = ProxyConnector.from_url(normalized_url, rdns=rdns, ssl=ssl_context)
    else:
        connector = TCPConnector(ssl=ssl_context)

    return ClientSession(connector=connector, timeout=ClientTimeout(total=timeout))


class ClientSessionRegistry:
    _client_session: "ClientSession"
    _client_session_active: bool = False
    _shutdown_unregister: "ClassVar[staticmethod[[], None] | None]" = None

    @staticmethod
    def get_session():
        if not ClientSessionRegistry._client_session_active:
            ClientSessionRegistry._client_session = _http_client_with_tls(timeout=None)
            ClientSessionRegistry._client_session_active = True
            ClientSessionRegistry._shutdown_unregister = staticmethod(on_shutdown(ClientSessionRegistry.close_session))
        return ClientSessionRegistry._client_session

    @staticmethod
    async def close_session():
        if ClientSessionRegistry._shutdown_unregister is not None:
            ClientSessionRegistry._shutdown_unregister()
            ClientSessionRegistry._shutdown_unregister = None
        if ClientSessionRegistry._client_session_active:
            session = ClientSessionRegistry._client_session
            del ClientSessionRegistry._client_session
            ClientSessionRegistry._client_session_active = False
            await session.close()


@contextlib.asynccontextmanager
async def run_temporary_http_server(app: "Application"):
    # Allocates a random port, runs a server in a context manager
    # This is used in various tests
    import socket

    from aiohttp.web_runner import AppRunner, SockSite

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
