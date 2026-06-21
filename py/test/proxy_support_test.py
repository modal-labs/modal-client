# Copyright Modal Labs 2026
"""Tests that the Modal client routes traffic through HTTP CONNECT proxies.

These tests spin up a minimal HTTP CONNECT proxy alongside the existing mock
servicer infrastructure and verify that gRPC (control plane + task command
router) and HTTP (blob uploads) traffic is tunneled through the proxy when
the standard HTTPS_PROXY / HTTP_PROXY environment variables are set.
"""

import asyncio
import os
import pytest
import socket
import threading
import urllib.parse

from modal import App, Image, Sandbox
from modal._utils.blob_utils import LARGE_FILE_LIMIT
from modal._utils.proxy_support import get_proxy_url
from modal.client import Client
from modal_proto import api_pb2

from .supports.skip import skip_windows

skip_non_subprocess = skip_windows("Needs subprocess support")


class HttpConnectProxy:
    """A minimal HTTP CONNECT proxy that logs tunneled destinations.

    On receiving a CONNECT request, the proxy opens a TCP connection to the
    requested target, replies with ``200 Connection established``, then blindly
    relays bytes in both directions.  All CONNECT targets are recorded in
    ``self.connect_targets`` as ``(host, port)`` tuples.
    """

    def __init__(self):
        self.connect_targets: list[tuple[str, int]] = []
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.host = "127.0.0.1"
        self.port: int = 0  # assigned after start()

    def start(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, 0))
        self.port = self._server_socket.getsockname()[1]
        self._server_socket.listen(32)
        self._server_socket.settimeout(0.5)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._server_socket:
            self._server_socket.close()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _accept_loop(self):
        while not self._stop_event.is_set():
            try:
                client_sock, _ = self._server_socket.accept()
            except (TimeoutError, OSError):
                continue
            t = threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True)
            t.start()

    def _handle_client(self, client_sock: socket.socket):
        try:
            client_sock.settimeout(30)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = client_sock.recv(4096)
                if not chunk:
                    return
                data += chunk

            request_line = data.split(b"\r\n")[0].decode("utf-8", errors="replace")
            parts = request_line.split()
            if len(parts) < 2:
                client_sock.close()
                return

            method = parts[0]
            if method == "CONNECT":
                target = parts[1]  # host:port
                host, _, port_str = target.rpartition(":")
                port = int(port_str)
                self.connect_targets.append((host, port))

                try:
                    remote_sock = socket.create_connection((host, port), timeout=10)
                except Exception:
                    client_sock.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                    client_sock.close()
                    return

                client_sock.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                self._relay(client_sock, remote_sock)
            else:
                # Forward proxy: absolute-URI request (e.g. PUT http://host:port/path)
                self._handle_forward_proxy(client_sock, data, parts)
        except Exception:
            pass
        finally:
            try:
                client_sock.close()
            except Exception:
                pass

    def _handle_forward_proxy(self, client_sock: socket.socket, data: bytes, parts: list[str]):
        """Forward-proxy an absolute-URI request to the origin server."""
        url = parts[1]
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        self.connect_targets.append((host, port))

        # Rewrite the request line to use a relative path
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query
        new_request_line = f"{parts[0]} {path} {parts[2]}\r\n"
        # Reconstruct the request with the rewritten first line
        header_end = data.index(b"\r\n\r\n") + 4
        headers_block = data[:header_end]
        remainder = data[header_end:]
        lines = headers_block.split(b"\r\n")
        lines[0] = new_request_line.rstrip("\r\n").encode()
        rewritten = b"\r\n".join(lines)

        try:
            remote_sock = socket.create_connection((host, port), timeout=10)
        except Exception:
            client_sock.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return

        remote_sock.sendall(rewritten + remainder)
        self._relay(client_sock, remote_sock)
        remote_sock.close()

    def _relay(self, sock_a: socket.socket, sock_b: socket.socket):
        def forward(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except Exception:
                    pass

        t1 = threading.Thread(target=forward, args=(sock_a, sock_b), daemon=True)
        t2 = threading.Thread(target=forward, args=(sock_b, sock_a), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        try:
            sock_b.close()
        except Exception:
            pass


class Socks5Proxy:
    """A minimal SOCKS5 proxy that logs tunneled destinations.

    Implements just enough of RFC 1928 to accept unauthenticated CONNECT
    requests, open a TCP connection to the target, and relay bytes.
    """

    def __init__(self):
        self.connect_targets: list[tuple[str, int]] = []
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.host = "127.0.0.1"
        self.port: int = 0

    def start(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, 0))
        self.port = self._server_socket.getsockname()[1]
        self._server_socket.listen(32)
        self._server_socket.settimeout(0.5)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._server_socket:
            self._server_socket.close()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"socks5://{self.host}:{self.port}"

    def _accept_loop(self):
        while not self._stop_event.is_set():
            try:
                client_sock, _ = self._server_socket.accept()
            except (TimeoutError, OSError):
                continue
            t = threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True)
            t.start()

    def _handle_client(self, client_sock: socket.socket):
        import struct

        try:
            client_sock.settimeout(30)

            # Greeting: client sends VER, NMETHODS, METHODS
            data = client_sock.recv(258)
            if len(data) < 3 or data[0] != 0x05:
                return
            # Reply: no authentication required
            client_sock.sendall(b"\x05\x00")

            # Request: VER, CMD, RSV, ATYP, DST.ADDR, DST.PORT
            data = client_sock.recv(4)
            if len(data) < 4 or data[0] != 0x05 or data[1] != 0x01:
                client_sock.sendall(b"\x05\x07\x00\x01" + b"\x00" * 6)
                return

            atyp = data[3]
            if atyp == 0x01:  # IPv4
                addr_data = client_sock.recv(4)
                target_host = socket.inet_ntoa(addr_data)
            elif atyp == 0x03:  # Domain name
                length = client_sock.recv(1)[0]
                target_host = client_sock.recv(length).decode()
            elif atyp == 0x04:  # IPv6
                addr_data = client_sock.recv(16)
                target_host = socket.inet_ntop(socket.AF_INET6, addr_data)
            else:
                client_sock.sendall(b"\x05\x08\x00\x01" + b"\x00" * 6)
                return

            port_data = client_sock.recv(2)
            target_port = struct.unpack("!H", port_data)[0]
            self.connect_targets.append((target_host, target_port))

            try:
                remote_sock = socket.create_connection((target_host, target_port), timeout=10)
            except Exception:
                client_sock.sendall(b"\x05\x05\x00\x01" + b"\x00" * 6)
                return

            # Success reply
            bind_addr = remote_sock.getsockname()
            bind_ip = socket.inet_aton(bind_addr[0])
            bind_port = struct.pack("!H", bind_addr[1])
            client_sock.sendall(b"\x05\x00\x00\x01" + bind_ip + bind_port)

            self._relay(client_sock, remote_sock)
        except Exception:
            pass
        finally:
            try:
                client_sock.close()
            except Exception:
                pass

    def _relay(self, sock_a: socket.socket, sock_b: socket.socket):
        def forward(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except Exception:
                    pass

        t1 = threading.Thread(target=forward, args=(sock_a, sock_b), daemon=True)
        t2 = threading.Thread(target=forward, args=(sock_b, sock_a), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        try:
            sock_b.close()
        except Exception:
            pass


def _close_http_session_registry():
    from modal._utils.http_utils import ClientSessionRegistry

    if not ClientSessionRegistry._client_session_active:
        return

    session_loop = ClientSessionRegistry._client_session._loop
    if session_loop.is_closed():
        ClientSessionRegistry._client_session_active = False
        return

    close_session = ClientSessionRegistry.close_session()
    if session_loop.is_running():
        asyncio.run_coroutine_threadsafe(close_session, session_loop).result(timeout=10)
    else:
        session_loop.run_until_complete(close_session)


@pytest.fixture(autouse=True)
def _reset_http_session_registry():
    """Reset the HTTP session singleton after each test.

    ProxyConnector-based sessions bake the proxy into the connector.
    Without this reset, a session created during a proxy test would
    persist in the ClientSessionRegistry and pollute later tests in
    the same Bazel shard (e.g. volume_test.py).
    """
    _close_http_session_registry()
    yield
    _close_http_session_registry()


@pytest.fixture
def http_proxy():
    proxy = HttpConnectProxy()
    proxy.start()
    yield proxy
    proxy.stop()


@pytest.fixture
def socks5_proxy():
    proxy = Socks5Proxy()
    proxy.start()
    yield proxy
    proxy.stop()


def _parse_host_port(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _assert_proxy_saw(proxy: HttpConnectProxy | Socks5Proxy, url: str, label: str):
    host, port = _parse_host_port(url)
    assert any(h == host and p == port for h, p in proxy.connect_targets), (
        f"Proxy did not see CONNECT to {label} ({host}:{port}). Targets seen: {proxy.connect_targets}"
    )


def _clear_proxy_env(monkeypatch):
    for env_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "MODAL_DISABLE_API_PROXY",
    ):
        monkeypatch.delenv(env_var, raising=False)


def test_https_proxy_lookup_does_not_fallback_to_http_proxy(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://http-proxy.example:3128")

    assert get_proxy_url("api.modal.com", use_ssl=True) is None
    assert get_proxy_url("api.modal.com", use_ssl=False) == "http://http-proxy.example:3128"


def test_https_proxy_lookup_uses_all_proxy(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("ALL_PROXY", "socks5://all-proxy.example:1080")

    assert get_proxy_url("api.modal.com", use_ssl=True) == "socks5://all-proxy.example:1080"


def test_grpc_control_plane_uses_proxy(servicer, credentials, http_proxy, monkeypatch):
    """gRPC calls to the control plane should be tunneled through the proxy."""
    monkeypatch.setenv("HTTP_PROXY", http_proxy.url)
    monkeypatch.setenv("HTTPS_PROXY", http_proxy.url)
    # Ensure 127.0.0.1 is NOT in NO_PROXY (the test mock servicer runs on loopback)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
        app = App()
        with app.run(client=client):
            sb = Sandbox.create("echo", "hi", app=app)
            sb.wait()

    _assert_proxy_saw(http_proxy, servicer.client_addr, "control plane (servicer)")


@skip_non_subprocess
def test_grpc_task_command_router_uses_proxy(servicer, credentials, http_proxy, monkeypatch):
    """gRPC calls to the task command router should be tunneled through the proxy."""
    monkeypatch.setenv("HTTP_PROXY", http_proxy.url)
    monkeypatch.setenv("HTTPS_PROXY", http_proxy.url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
        app = App()
        with app.run(client=client):
            sb = Sandbox.create("sleep", "infinity", app=app)
            # exec triggers a gRPC connection to the task command router
            sb.exec("echo", "hello from proxy test")

    _assert_proxy_saw(http_proxy, servicer.task_command_router_url, "task command router")


def test_blob_upload_uses_proxy(servicer, credentials, http_proxy, monkeypatch, tmp_path):
    """Blob uploads via aiohttp should be tunneled through the proxy.

    Creates a file larger than LARGE_FILE_LIMIT (4 MiB) and adds it to an
    image with copy=True, which triggers a blob upload to the mock S3 server.
    """
    monkeypatch.setenv("HTTP_PROXY", http_proxy.url)
    monkeypatch.setenv("HTTPS_PROXY", http_proxy.url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    # Create a file larger than LARGE_FILE_LIMIT to trigger blob upload
    large_file = tmp_path / "large_payload.bin"
    large_file.write_bytes(os.urandom(LARGE_FILE_LIMIT + 1024 * 1024))  # LARGE_FILE_LIMIT + 1 MiB

    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
        app = App()
        with app.run(client=client):
            image = Image.debian_slim().add_local_file(str(large_file), remote_path="/data/payload.bin", copy=True)
            Sandbox.create("echo", "hi", image=image, app=app)

    _assert_proxy_saw(http_proxy, servicer.blob_host, "blob server")


def test_socks5_proxy_via_all_proxy(servicer, credentials, socks5_proxy, monkeypatch):
    """gRPC calls should be tunneled through a SOCKS5 proxy when ALL_PROXY is set."""
    monkeypatch.setenv("ALL_PROXY", socks5_proxy.url)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
        app = App()
        with app.run(client=client):
            sb = Sandbox.create("echo", "hi", app=app)
            sb.wait()

    _assert_proxy_saw(socks5_proxy, servicer.client_addr, "control plane via SOCKS5")


def test_proxy_disabled_via_config(servicer, credentials, http_proxy, monkeypatch):
    """When MODAL_DISABLE_API_PROXY=1, proxy env vars should be ignored."""
    monkeypatch.setenv("HTTP_PROXY", http_proxy.url)
    monkeypatch.setenv("HTTPS_PROXY", http_proxy.url)
    monkeypatch.setenv("MODAL_DISABLE_API_PROXY", "1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
        app = App()
        with app.run(client=client):
            sb = Sandbox.create("echo", "hi", app=app)
            sb.wait()

    # Proxy should NOT have seen any traffic
    assert len(http_proxy.connect_targets) == 0, (
        f"Proxy should not have been used when disabled, but saw: {http_proxy.connect_targets}"
    )
