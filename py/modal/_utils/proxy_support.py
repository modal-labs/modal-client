# Copyright Modal Labs 2026
"""Proxy support for gRPC (grpclib) and HTTP (aiohttp) connections.

By default, the Modal client reads standard proxy environment variables
(HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, NO_PROXY and their lowercase
variants).  Set ``MODAL_DISABLE_API_PROXY=1`` to opt out.

Supported proxy types:
- HTTP CONNECT (``http://host:port``)
- SOCKS4/4a/5/5h (``socks5://``, ``socks5h://``, etc.)

All proxy types require the optional ``python-socks[asyncio]`` package.
Install it via ``pip install modal[api-proxy-support]``.
"""

from __future__ import annotations

import asyncio
import ssl
import urllib.parse
import urllib.request

from grpclib.protocol import H2Protocol


def get_proxy_url(host: str | None, *, use_ssl: bool) -> str | None:
    """Return the proxy URL for the given target, respecting NO_PROXY when host is known."""
    from ..config import config

    if config.get("disable_api_proxy"):
        return None

    if host is not None and urllib.request.proxy_bypass(host):
        return None

    proxies = urllib.request.getproxies()
    if not proxies:
        return None

    scheme = "https" if use_ssl else "http"
    proxy_url = proxies.get(scheme) or proxies.get("all")
    return proxy_url or None


def _normalize_proxy_url(proxy_url: str) -> tuple[str, bool]:
    """Normalize a proxy URL for python-socks, which doesn't handle socks5h/socks4a.

    Returns (normalized_url, rdns) where rdns=True for remote-DNS schemes.
    """
    parsed = urllib.parse.urlparse(proxy_url)
    scheme = parsed.scheme.lower()
    if scheme == "socks5h":
        return urllib.parse.urlunparse(
            ("socks5", parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        ), True
    elif scheme == "socks4a":
        return urllib.parse.urlunparse(
            ("socks4", parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        ), True
    return proxy_url, False


async def create_proxied_connection(
    protocol_factory,
    host: str,
    port: int,
    *,
    proxy_url: str,
    ssl_context: ssl.SSLContext | None = None,
) -> H2Protocol:
    """Create a gRPC connection through the configured proxy."""

    try:
        from python_socks.async_.asyncio import Proxy
    except ImportError:
        raise ImportError(
            f"A proxy is configured ({proxy_url}) but the 'python-socks' package is not installed. "
            "Install it with: pip install 'modal[api-proxy-support]'"
        ) from None

    normalized_url, rdns = _normalize_proxy_url(proxy_url)
    proxy = Proxy.from_url(normalized_url, rdns=rdns)
    sock = await proxy.connect(host, port)

    loop = asyncio.get_running_loop()
    try:
        if ssl_context:
            _, protocol = await loop.create_connection(
                protocol_factory,
                sock=sock,
                ssl=ssl_context,
                server_hostname=host,
            )
        else:
            _, protocol = await loop.create_connection(
                protocol_factory,
                sock=sock,
            )
    except BaseException:
        sock.close()
        raise
    return protocol
