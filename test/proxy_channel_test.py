# Copyright Modal Labs 2025
import asyncio
import base64
import pytest
import socket

import grpclib.client

from modal._utils.grpc_utils import ProxyChannel, _should_bypass_proxy, create_channel
from modal.config import Config

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
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
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


def test_proxy_channel_rejects_crlf_in_host():
    """ProxyChannel rejects target host containing CRLF (CONNECT request injection prevention)."""
    with pytest.raises(ValueError, match="invalid characters"):
        ProxyChannel("evil\r\nhost", 443, proxy_url="http://proxy:3128", ssl=True)
    with pytest.raises(ValueError, match="invalid characters"):
        ProxyChannel("api.modal.com\r\n", 443, proxy_url="http://proxy:3128", ssl=True)


def test_proxy_channel_no_scheme_url():
    """ProxyChannel rejects a URL without an http:// scheme."""
    with pytest.raises(ValueError, match="Unsupported proxy scheme"):
        ProxyChannel("api.modal.com", 443, proxy_url="myproxy:3128", ssl=True)


def test_proxy_channel_rejects_schemeless_double_slash_url():
    """ProxyChannel rejects //proxy:port URLs (empty scheme not accepted)."""
    with pytest.raises(ValueError, match="Unsupported proxy scheme"):
        ProxyChannel("api.modal.com", 443, proxy_url="//proxy:3128", ssl=True)


def test_proxy_channel_parses_percent_encoded_auth():
    """ProxyChannel correctly decodes percent-encoded credentials."""
    ch = ProxyChannel("api.modal.com", 443, proxy_url="http://user%40corp:p%40ss@proxy:3128", ssl=True)
    try:
        expected = base64.b64encode(b"user@corp:p@ss").decode()
        assert ch._proxy_auth == expected
    finally:
        ch.close()


# --- Config edge case tests ---


def test_grpc_proxy_whitespace_only_https_proxy_treated_as_none(monkeypatch):
    """Whitespace-only HTTPS_PROXY is treated as unset."""
    monkeypatch.delenv("MODAL_GRPC_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "   ")
    monkeypatch.delenv("https_proxy", raising=False)
    assert Config().get("grpc_proxy") is None


# --- NO_PROXY tests ---


def test_no_proxy_bypasses_proxy(monkeypatch):
    """create_channel returns standard Channel when target host matches NO_PROXY."""
    monkeypatch.setenv("MODAL_GRPC_PROXY", "http://proxy:3128")
    monkeypatch.setenv("NO_PROXY", "*.modal.com")
    monkeypatch.delenv("no_proxy", raising=False)
    channel = create_channel("https://api.modal.com:443")
    try:
        assert type(channel) is grpclib.client.Channel
    finally:
        channel.close()


def test_no_proxy_does_not_bypass_unrelated_host(monkeypatch):
    """create_channel returns ProxyChannel when target host does not match NO_PROXY."""
    monkeypatch.setenv("MODAL_GRPC_PROXY", "http://proxy:3128")
    monkeypatch.setenv("NO_PROXY", "*.other.com")
    monkeypatch.delenv("no_proxy", raising=False)
    channel = create_channel("https://api.modal.com:443")
    try:
        assert isinstance(channel, ProxyChannel)
    finally:
        channel.close()


# --- _should_bypass_proxy edge cases ---


def test_should_bypass_proxy_bracketed_ipv6(monkeypatch):
    """Bracketed IPv6 target host matches unbracketed NO_PROXY entry."""
    monkeypatch.setenv("NO_PROXY", "::1")
    monkeypatch.delenv("no_proxy", raising=False)
    assert _should_bypass_proxy("[::1]") is True


def test_should_bypass_proxy_bracketed_ipv6_entry(monkeypatch):
    """NO_PROXY entry with brackets matches plain IPv6 host."""
    monkeypatch.setenv("NO_PROXY", "[::1]")
    monkeypatch.delenv("no_proxy", raising=False)
    assert _should_bypass_proxy("::1") is True


def test_should_bypass_proxy_port_match(monkeypatch):
    """NO_PROXY entry with port matches when target port matches."""
    monkeypatch.setenv("NO_PROXY", "example.com:443")
    monkeypatch.delenv("no_proxy", raising=False)
    assert _should_bypass_proxy("example.com", port=443) is True


def test_should_bypass_proxy_port_mismatch(monkeypatch):
    """NO_PROXY entry with port does not match when target port differs."""
    monkeypatch.setenv("NO_PROXY", "example.com:8443")
    monkeypatch.delenv("no_proxy", raising=False)
    assert _should_bypass_proxy("example.com", port=443) is False


def test_should_bypass_proxy_no_port_in_entry_matches_any_port(monkeypatch):
    """NO_PROXY entry without port matches regardless of target port."""
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.delenv("no_proxy", raising=False)
    assert _should_bypass_proxy("example.com", port=443) is True
    assert _should_bypass_proxy("example.com", port=8080) is True


def test_should_bypass_proxy_bracketed_ipv6_with_port(monkeypatch):
    """NO_PROXY entry [::1]:8443 matches only when port matches."""
    monkeypatch.setenv("NO_PROXY", "[::1]:8443")
    monkeypatch.delenv("no_proxy", raising=False)
    assert _should_bypass_proxy("::1", port=8443) is True
    assert _should_bypass_proxy("[::1]", port=8443) is True
    assert _should_bypass_proxy("::1", port=443) is False


# --- CONNECT handshake tests ---


async def _run_mock_proxy(host, response_bytes):
    """Start a mock TCP server on an OS-assigned port that captures the CONNECT request and sends a response."""
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

    server = await asyncio.start_server(handle_client, host, 0)
    port = server.sockets[0].getsockname()[1]
    return server, port, received, done


@pytest.mark.asyncio
async def test_connect_handshake_sends_correct_request():
    """Verify the CONNECT request format sent to the proxy."""
    server, proxy_port, received, done = await _run_mock_proxy(
        "127.0.0.1", b"HTTP/1.1 200 Connection established\r\n\r\n"
    )
    try:
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

        request_text = received.decode()
        assert request_text.startswith("CONNECT api.modal.com:443 HTTP/1.1\r\n")
        assert "Host: api.modal.com:443\r\n" in request_text
        assert "Proxy-Authorization" not in request_text
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_connect_handshake_sends_auth_header():
    """Verify Proxy-Authorization header is sent for authenticated proxies."""
    server, proxy_port, received, done = await _run_mock_proxy(
        "127.0.0.1", b"HTTP/1.1 200 Connection established\r\n\r\n"
    )
    try:
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

        request_text = received.decode()
        expected_auth = base64.b64encode(b"myuser:mypass").decode()
        assert f"Proxy-Authorization: Basic {expected_auth}\r\n" in request_text
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_connect_handshake_rejects_non_200():
    """Verify ConnectionError is raised when proxy returns non-2xx status."""
    server, proxy_port, received, done = await _run_mock_proxy(
        "127.0.0.1",
        b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n",
    )
    try:
        ch = ProxyChannel(
            "api.modal.com",
            443,
            proxy_url=f"http://127.0.0.1:{proxy_port}",
            ssl=True,
        )
        try:
            with pytest.raises(OSError, match="Proxy CONNECT failed.*407"):
                await ch.__connect__()
        finally:
            ch.close()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_connect_handshake_handles_closed_connection():
    """Verify ConnectionError is raised when proxy closes connection during handshake."""

    async def handle_and_close(reader, writer):
        await reader.read(4096)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_and_close, "127.0.0.1", 0)
    proxy_port = server.sockets[0].getsockname()[1]
    try:
        ch = ProxyChannel(
            "api.modal.com",
            443,
            proxy_url=f"http://127.0.0.1:{proxy_port}",
            ssl=True,
        )
        try:
            with pytest.raises(OSError, match="Proxy closed connection"):
                await ch.__connect__()
        finally:
            ch.close()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_connect_rejects_oversized_response():
    """Verify ConnectionError when proxy sends >16KiB without terminating headers."""
    oversized = b"X" * (16 * 1024 + 1)

    async def handle_client(reader, writer):
        await reader.read(4096)
        writer.write(oversized)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    proxy_port = server.sockets[0].getsockname()[1]
    try:
        ch = ProxyChannel(
            "api.modal.com",
            443,
            proxy_url=f"http://127.0.0.1:{proxy_port}",
            ssl=True,
        )
        try:
            with pytest.raises(OSError, match="exceeded"):
                await ch.__connect__()
        finally:
            ch.close()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_connect_timeout(monkeypatch):
    """Verify OSError with timeout message when proxy never responds."""
    hang_forever = asyncio.Event()

    async def handle_client(reader, writer):
        # Accept TCP connection but never send a response
        await hang_forever.wait()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    proxy_port = server.sockets[0].getsockname()[1]
    try:
        ch = ProxyChannel(
            "api.modal.com",
            443,
            proxy_url=f"http://127.0.0.1:{proxy_port}",
            ssl=True,
        )
        monkeypatch.setattr(ProxyChannel, "_CONNECT_TIMEOUT", 0.5)
        try:
            with pytest.raises(OSError, match="timed out"):
                await ch.__connect__()
        finally:
            ch.close()
    finally:
        hang_forever.set()
        server.close()
        await server.wait_closed()


# --- DNS resolution failure test ---


@pytest.mark.asyncio
async def test_connect_getaddrinfo_failure(monkeypatch):
    """Verify OSError propagates when proxy hostname cannot be resolved."""
    ch = ProxyChannel(
        "api.modal.com",
        443,
        proxy_url="http://nonexistent-proxy.invalid:3128",
        ssl=True,
    )

    async def fake_getaddrinfo(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(asyncio.get_event_loop().__class__, "getaddrinfo", fake_getaddrinfo)
    try:
        with pytest.raises(OSError):
            await ch.__connect__()
    finally:
        ch.close()


# --- IPv6 proxy URL test ---


def test_proxy_channel_parses_ipv6_url():
    """ProxyChannel correctly parses bracketed IPv6 proxy URL."""
    ch = ProxyChannel("api.modal.com", 443, proxy_url="http://[::1]:3128", ssl=True)
    try:
        assert ch._proxy_host == "::1"
        assert ch._proxy_port == 3128
        assert ch._proxy_auth is None
    finally:
        ch.close()


# --- __repr__ tests ---


def test_proxy_channel_repr_without_auth():
    """__repr__ shows proxy host/port and target without auth."""
    ch = ProxyChannel("api.modal.com", 443, proxy_url="http://myproxy:8080", ssl=True)
    try:
        r = repr(ch)
        assert "myproxy:8080" in r
        assert "api.modal.com:443" in r
        assert "auth" not in r
    finally:
        ch.close()


def test_proxy_channel_repr_masks_auth():
    """__repr__ masks auth credentials."""
    ch = ProxyChannel("api.modal.com", 443, proxy_url="http://user:secret@myproxy:8080", ssl=True)
    try:
        r = repr(ch)
        assert "myproxy:8080" in r
        assert "auth=***" in r
        assert "secret" not in r
        assert "user" not in r
    finally:
        ch.close()


# --- Input validation tests ---


def test_proxy_channel_rejects_whitespace_in_host():
    """ProxyChannel rejects target host containing spaces or tabs."""
    with pytest.raises(ValueError, match="whitespace"):
        ProxyChannel("api modal.com", 443, proxy_url="http://proxy:3128", ssl=True)
    with pytest.raises(ValueError, match="whitespace"):
        ProxyChannel("api\tmodal.com", 443, proxy_url="http://proxy:3128", ssl=True)


def test_proxy_channel_rejects_port_out_of_range():
    """ProxyChannel rejects target port outside 1..65535."""
    with pytest.raises(ValueError, match="port out of range"):
        ProxyChannel("api.modal.com", 0, proxy_url="http://proxy:3128", ssl=True)
    with pytest.raises(ValueError, match="port out of range"):
        ProxyChannel("api.modal.com", 65536, proxy_url="http://proxy:3128", ssl=True)
    with pytest.raises(ValueError, match="port out of range"):
        ProxyChannel("api.modal.com", -1, proxy_url="http://proxy:3128", ssl=True)


def test_proxy_channel_rejects_missing_hostname():
    """ProxyChannel rejects proxy URL with no hostname."""
    with pytest.raises(ValueError, match="missing a hostname"):
        ProxyChannel("api.modal.com", 443, proxy_url="http://:3128", ssl=True)


# --- Config credential redaction tests ---


def test_redact_url_credentials_strips_userinfo():
    """_redact_url_credentials replaces user:pass with ***."""
    from modal.cli.config import _redact_url_credentials

    assert _redact_url_credentials("http://user:pass@proxy:3128") == "http://***@proxy:3128"


def test_redact_url_credentials_preserves_no_auth_url():
    """_redact_url_credentials returns URL unchanged when no userinfo present."""
    from modal.cli.config import _redact_url_credentials

    assert _redact_url_credentials("http://proxy:3128") == "http://proxy:3128"


def test_redact_url_credentials_handles_username_only():
    """_redact_url_credentials redacts even username-only URLs."""
    from modal.cli.config import _redact_url_credentials

    result = _redact_url_credentials("http://admin@proxy:3128")
    assert "admin" not in result
    assert "***@proxy:3128" in result


# --- DNS timeout coverage test ---


@pytest.mark.asyncio
async def test_connect_dns_timeout(monkeypatch):
    """Verify timeout covers DNS resolution (getaddrinfo), not just handshake."""

    async def slow_getaddrinfo(*args, **kwargs):
        await asyncio.sleep(10)  # Simulate a hanging resolver
        return []

    ch = ProxyChannel(
        "api.modal.com",
        443,
        proxy_url="http://slow-proxy.invalid:3128",
        ssl=True,
    )
    monkeypatch.setattr(ProxyChannel, "_CONNECT_TIMEOUT", 0.3)
    monkeypatch.setattr(asyncio.get_event_loop().__class__, "getaddrinfo", slow_getaddrinfo)
    try:
        with pytest.raises(OSError, match="timed out"):
            await ch.__connect__()
    finally:
        ch.close()
