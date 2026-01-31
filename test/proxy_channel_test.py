# Copyright Modal Labs 2025
import asyncio
import base64
import pytest

import grpclib.client

from modal._utils.grpc_utils import ProxyChannel, create_channel
from modal.config import Config
from modal.exception import ConnectionError as ModalConnectionError

# --- Config priority tests ---


def test_grpc_proxy_modal_env_var_highest_priority(monkeypatch):
    """MODAL_GRPC_PROXY takes precedence over HTTPS_PROXY and https_proxy."""
    monkeypatch.setenv("MODAL_GRPC_PROXY", "http://modal-proxy:1111")
    monkeypatch.setenv("HTTPS_PROXY", "http://env-proxy:2222")
    monkeypatch.setenv("https_proxy", "http://lower-proxy:3333")
    assert Config().get("grpc_proxy") == "http://modal-proxy:1111"


def test_grpc_proxy_https_proxy_fallback(monkeypatch):
    """HTTPS_PROXY is used when MODAL_GRPC_PROXY is not set."""
    monkeypatch.delenv("MODAL_GRPC_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://env-proxy:2222")
    monkeypatch.setenv("https_proxy", "http://lower-proxy:3333")
    assert Config().get("grpc_proxy") == "http://env-proxy:2222"


def test_grpc_proxy_lowercase_https_proxy_fallback(monkeypatch):
    """https_proxy is used when both MODAL_GRPC_PROXY and HTTPS_PROXY are unset."""
    monkeypatch.delenv("MODAL_GRPC_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("https_proxy", "http://lower-proxy:3333")
    assert Config().get("grpc_proxy") == "http://lower-proxy:3333"


def test_grpc_proxy_none_when_no_env_vars(monkeypatch):
    """Returns None when no proxy env vars are set."""
    monkeypatch.delenv("MODAL_GRPC_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    assert Config().get("grpc_proxy") is None


# --- create_channel routing tests ---


def test_create_channel_uses_proxy_for_https(monkeypatch):
    """create_channel returns ProxyChannel when proxy is configured and scheme is https."""
    monkeypatch.setenv("MODAL_GRPC_PROXY", "http://proxy:3128")
    channel = create_channel("https://api.modal.com:443")
    try:
        assert isinstance(channel, ProxyChannel)
    finally:
        channel.close()


def test_create_channel_standard_for_https_without_proxy(monkeypatch):
    """create_channel returns standard Channel when no proxy is configured."""
    monkeypatch.delenv("MODAL_GRPC_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    channel = create_channel("https://api.modal.com:443")
    try:
        assert type(channel) is grpclib.client.Channel
    finally:
        channel.close()


def test_create_channel_standard_for_http_even_with_proxy(monkeypatch):
    """create_channel never proxies HTTP (non-TLS) connections."""
    monkeypatch.setenv("MODAL_GRPC_PROXY", "http://proxy:3128")
    channel = create_channel("http://localhost:50051")
    try:
        assert type(channel) is grpclib.client.Channel
    finally:
        channel.close()


def test_create_channel_standard_for_unix_even_with_proxy(monkeypatch):
    """create_channel never proxies Unix socket connections."""
    monkeypatch.setenv("MODAL_GRPC_PROXY", "http://proxy:3128")
    channel = create_channel("unix:///tmp/modal.sock")
    try:
        assert type(channel) is grpclib.client.Channel
    finally:
        channel.close()


# --- ProxyChannel construction tests ---


def test_proxy_channel_parses_url():
    """ProxyChannel correctly parses proxy host and port."""
    ch = ProxyChannel("api.modal.com", 443, proxy_url="http://myproxy:8080", ssl=True)
    try:
        assert ch._proxy_host == "myproxy"
        assert ch._proxy_port == 8080
        assert ch._proxy_auth is None
    finally:
        ch.close()


def test_proxy_channel_parses_auth():
    """ProxyChannel correctly parses and encodes basic auth credentials."""
    ch = ProxyChannel("api.modal.com", 443, proxy_url="http://user:s3cret@myproxy:8080", ssl=True)
    try:
        expected = base64.b64encode(b"user:s3cret").decode()
        assert ch._proxy_auth == expected
    finally:
        ch.close()


def test_proxy_channel_default_port():
    """ProxyChannel defaults to port 3128 when not specified."""
    ch = ProxyChannel("api.modal.com", 443, proxy_url="http://myproxy", ssl=True)
    try:
        assert ch._proxy_port == 3128
    finally:
        ch.close()


def test_proxy_channel_rejects_socks_scheme():
    """ProxyChannel rejects non-http proxy URL schemes."""
    with pytest.raises(ValueError, match="Unsupported proxy scheme"):
        ProxyChannel("api.modal.com", 443, proxy_url="socks5://myproxy:1080", ssl=True)


def test_proxy_channel_rejects_crlf_in_url():
    """ProxyChannel rejects proxy URLs containing CRLF (header injection prevention)."""
    with pytest.raises(ValueError, match="invalid characters"):
        ProxyChannel("api.modal.com", 443, proxy_url="http://evil\r\nhost:3128", ssl=True)
    with pytest.raises(ValueError, match="invalid characters"):
        ProxyChannel("api.modal.com", 443, proxy_url="http://user:pass@proxy:3128\r\n", ssl=True)


# --- Config edge case tests ---


def test_grpc_proxy_whitespace_only_https_proxy_treated_as_none(monkeypatch):
    """Whitespace-only HTTPS_PROXY is treated as unset."""
    monkeypatch.delenv("MODAL_GRPC_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "   ")
    monkeypatch.delenv("https_proxy", raising=False)
    assert Config().get("grpc_proxy") is None


# --- CONNECT handshake tests ---


async def _run_mock_proxy(host, port, response_bytes):
    """Start a mock TCP server that captures the CONNECT request and sends a response."""
    received = bytearray()
    done = asyncio.Event()

    async def handle_client(reader, writer):
        while True:
            data = await reader.read(4096)
            if not data:
                break
            received.extend(data)
            if b"\r\n\r\n" in received:
                writer.write(response_bytes)
                await writer.drain()
                # Close after sending response — don't wait for TLS handshake
                writer.close()
                await writer.wait_closed()
                done.set()
                return
        done.set()

    server = await asyncio.start_server(handle_client, host, port)
    return server, received, done


@pytest.mark.asyncio
async def test_connect_handshake_sends_correct_request():
    """Verify the CONNECT request format sent to the proxy."""
    # Find a free port for the mock proxy
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        proxy_port = s.getsockname()[1]

    # Respond with 200 then close — TLS handshake will fail, but we can inspect the CONNECT request
    server, received, done = await _run_mock_proxy(
        "127.0.0.1", proxy_port, b"HTTP/1.1 200 Connection established\r\n\r\n"
    )

    ch = ProxyChannel(
        "api.modal.com",
        443,
        proxy_url=f"http://127.0.0.1:{proxy_port}",
        ssl=True,
    )
    try:
        # The TLS handshake will fail because our mock proxy just closes,
        # but the CONNECT request should have been sent correctly
        with pytest.raises(Exception):
            await ch.__connect__()
    finally:
        ch.close()

    await asyncio.wait_for(done.wait(), timeout=5)
    server.close()
    await server.wait_closed()

    request_text = received.decode()
    assert request_text.startswith("CONNECT api.modal.com:443 HTTP/1.1\r\n")
    assert "Host: api.modal.com:443\r\n" in request_text
    assert "Proxy-Authorization" not in request_text


@pytest.mark.asyncio
async def test_connect_handshake_sends_auth_header():
    """Verify Proxy-Authorization header is sent for authenticated proxies."""
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        proxy_port = s.getsockname()[1]

    server, received, done = await _run_mock_proxy(
        "127.0.0.1", proxy_port, b"HTTP/1.1 200 Connection established\r\n\r\n"
    )

    ch = ProxyChannel(
        "api.modal.com",
        443,
        proxy_url=f"http://myuser:mypass@127.0.0.1:{proxy_port}",
        ssl=True,
    )
    try:
        with pytest.raises(Exception):
            await ch.__connect__()
    finally:
        ch.close()

    await asyncio.wait_for(done.wait(), timeout=5)
    server.close()
    await server.wait_closed()

    request_text = received.decode()
    expected_auth = base64.b64encode(b"myuser:mypass").decode()
    assert f"Proxy-Authorization: Basic {expected_auth}\r\n" in request_text


@pytest.mark.asyncio
async def test_connect_handshake_rejects_non_200():
    """Verify ConnectionError is raised when proxy returns non-2xx status."""
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        proxy_port = s.getsockname()[1]

    server, received, done = await _run_mock_proxy(
        "127.0.0.1",
        proxy_port,
        b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n",
    )

    ch = ProxyChannel(
        "api.modal.com",
        443,
        proxy_url=f"http://127.0.0.1:{proxy_port}",
        ssl=True,
    )
    try:
        with pytest.raises(ModalConnectionError, match="Proxy CONNECT failed.*407"):
            await ch.__connect__()
    finally:
        ch.close()

    server.close()
    await server.wait_closed()


@pytest.mark.asyncio
async def test_connect_handshake_handles_closed_connection():
    """Verify ConnectionError is raised when proxy closes connection during handshake."""
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        proxy_port = s.getsockname()[1]

    async def handle_and_close(reader, writer):
        await reader.read(4096)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_and_close, "127.0.0.1", proxy_port)

    ch = ProxyChannel(
        "api.modal.com",
        443,
        proxy_url=f"http://127.0.0.1:{proxy_port}",
        ssl=True,
    )
    try:
        with pytest.raises(ModalConnectionError, match="Proxy closed connection"):
            await ch.__connect__()
    finally:
        ch.close()

    server.close()
    await server.wait_closed()
