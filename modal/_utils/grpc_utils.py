# Copyright Modal Labs 2022
import asyncio
import base64
import contextlib
import os
import platform
import socket
import time
import typing
import urllib.parse
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Optional, Sequence, TypeVar

import grpclib.client
import grpclib.config
import grpclib.events
from google.protobuf.message import Message
from google.protobuf.symbol_database import SymbolDatabase
from grpclib import GRPCError, Status
from grpclib.encoding.base import StatusDetailsCodecBase
from grpclib.exceptions import StreamTerminatedError
from grpclib.protocol import H2Protocol

from modal.exception import ConnectionError
from modal_proto import api_pb2
from modal_version import __version__

from .._traceback import suppress_tb_frame
from ..config import config
from .async_utils import retry
from .logger import logger

RequestType = TypeVar("RequestType", bound=Message)
ResponseType = TypeVar("ResponseType", bound=Message)

if typing.TYPE_CHECKING:
    import modal._grpc_client
    import modal.client

# Monkey patches grpclib to have a Modal User Agent header.
grpclib.client.USER_AGENT = "modal-client/{version} ({sys}; {py}/{py_ver})'".format(
    version=__version__,
    sys=platform.system(),
    py=platform.python_implementation(),
    py_ver=platform.python_version(),
).lower()


class ProxyChannel(grpclib.client.Channel):
    """A gRPC channel that tunnels HTTPS gRPC connections through an HTTP CONNECT proxy.

    Subclasses grpclib's Channel to override connection creation. Instead of
    connecting directly to the target host, it first establishes a TCP
    connection to the proxy, performs an HTTP CONNECT handshake, and then
    upgrades the tunneled connection to TLS + HTTP/2.

    Validated against grpclib 0.4.7–0.4.9. Accesses private attributes:
    _loop, _host, _port, _ssl, _config, _protocol_factory (from Channel.__init__).
    """

    _CONNECT_TIMEOUT = 30  # seconds for proxy connect and handshake operations
    _MAX_RESPONSE_BYTES = 16 * 1024  # 16 KiB max for proxy CONNECT response

    def __init__(
        self,
        host: str,
        port: int,
        *,
        proxy_url: str,
        ssl: Any = None,
        config: Optional[grpclib.config.Configuration] = None,
        status_details_codec: Any = None,
    ):
        super().__init__(
            host,
            port,
            ssl=ssl,
            config=config,
            status_details_codec=status_details_codec,
        )
        # Validate target host to prevent CRLF/whitespace injection into CONNECT request
        if "\r" in host or "\n" in host:
            raise ValueError("Target host contains invalid characters (CR/LF)")
        if " " in host or "\t" in host:
            raise ValueError("Target host contains whitespace")
        if not (1 <= port <= 65535):
            raise ValueError(f"Target port out of range: {port}")
        # Validate raw URL before parsing to prevent header injection (urlparse strips CR/LF)
        if "\r" in proxy_url or "\n" in proxy_url:
            raise ValueError("Proxy URL contains invalid characters (CR/LF)")

        parsed = urllib.parse.urlparse(proxy_url)
        if parsed.scheme != "http":
            raise ValueError(f"Unsupported proxy scheme: {parsed.scheme!r} (only http:// is supported)")
        if not parsed.hostname:
            raise ValueError("Proxy URL is missing a hostname")

        self._proxy_host = parsed.hostname
        self._proxy_port = parsed.port or 3128

        self._proxy_auth: Optional[str] = None
        if parsed.username:
            username = urllib.parse.unquote(parsed.username)
            password = urllib.parse.unquote(parsed.password or "")
            credentials = f"{username}:{password}"
            self._proxy_auth = base64.b64encode(credentials.encode()).decode()

    def __repr__(self) -> str:
        auth = " auth=***" if self._proxy_auth else ""
        return f"ProxyChannel(proxy={self._proxy_host}:{self._proxy_port}{auth} -> {self._host}:{self._port})"

    async def _create_connection(self) -> H2Protocol:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError(
                "ProxyChannel was created in a different event loop than the one currently running"
            )
        try:
            return await asyncio.wait_for(
                self._resolve_and_connect(),
                timeout=self._CONNECT_TIMEOUT,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise OSError(
                f"Proxy connect/handshake timed out after {self._CONNECT_TIMEOUT}s "
                f"({self._proxy_host}:{self._proxy_port})"
            ) from exc

    async def _resolve_and_connect(self) -> H2Protocol:
        """Resolve proxy address, open socket, and perform CONNECT handshake.

        Called inside wait_for so DNS resolution is also covered by the timeout.
        """
        loop = asyncio.get_running_loop()
        # Resolve proxy address (supports both IPv4 and IPv6)
        infos = await loop.getaddrinfo(self._proxy_host, self._proxy_port, type=socket.SOCK_STREAM)
        if not infos:
            raise OSError(f"Could not resolve proxy host: {self._proxy_host}")
        family, _, _, _, sockaddr = infos[0]

        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.setblocking(False)
            protocol = await self._proxy_handshake(sock, sockaddr)
            sock = None  # type: ignore[assignment]  # Transport now owns the socket
            return protocol
        finally:
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()

    async def _proxy_handshake(self, sock: socket.socket, sockaddr: tuple) -> H2Protocol:
        loop = asyncio.get_running_loop()
        await loop.sock_connect(sock, sockaddr)

        # Bracket-wrap IPv6 literal hosts in the CONNECT request (avoid double-bracketing)
        host = self._host if self._host.startswith("[") else (f"[{self._host}]" if ":" in self._host else self._host)

        connect_req = f"CONNECT {host}:{self._port} HTTP/1.1\r\n"
        connect_req += f"Host: {host}:{self._port}\r\n"
        if self._proxy_auth:
            connect_req += f"Proxy-Authorization: Basic {self._proxy_auth}\r\n"
        connect_req += "\r\n"

        await loop.sock_sendall(sock, connect_req.encode())

        # Read proxy response one byte at a time to avoid consuming data
        # beyond the header delimiter that belongs to the tunneled connection.
        # CONNECT responses are typically <200 bytes so this is efficient enough.
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = await loop.sock_recv(sock, 1)
            if not chunk:
                raise OSError("Proxy closed connection during CONNECT handshake")
            response += chunk
            if len(response) > self._MAX_RESPONSE_BYTES:
                raise OSError(f"Proxy response exceeded {self._MAX_RESPONSE_BYTES} bytes")

        status_line = response.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
        # Expected: "HTTP/1.1 200 Connection established" (or similar 2xx)
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or not parts[1].startswith("2"):
            raise OSError(f"Proxy CONNECT failed: {status_line}")

        # server_hostname is required explicitly when using sock= mode with
        # create_connection, unlike the base class which passes host= and lets
        # asyncio handle hostname resolution and TLS server name indication.
        server_hostname = (
            getattr(self._config, "ssl_target_name_override", None) or self._host
        ) if self._ssl is not None else None

        # Upgrade tunneled socket to TLS + H2
        _, protocol = await loop.create_connection(
            self._protocol_factory,
            ssl=self._ssl,
            sock=sock,
            server_hostname=server_hostname,
        )
        return protocol


class Subchannel:
    protocol: H2Protocol
    created_at: float
    requests: int

    def __init__(self, protocol: H2Protocol) -> None:
        self.protocol = protocol
        self.created_at = time.time()
        self.requests = 0

    def connected(self):
        if hasattr(self.protocol.handler, "connection_lost"):
            # AbstractHandler doesn't have connection_lost, but Handler does
            return not self.protocol.handler.connection_lost  # type: ignore
        return True


RETRYABLE_GRPC_STATUS_CODES = [
    Status.DEADLINE_EXCEEDED,
    Status.UNAVAILABLE,
    Status.CANCELLED,
    Status.INTERNAL,
    Status.UNKNOWN,
]
SERVER_RETRY_WARNING_TIME_INTERVAL = 30.0


@dataclass
class RetryWarningMessage:
    message: str
    warning_interval: int
    errors_to_warn_for: typing.List[Status]


class ConnectionManager:
    """ConnectionManager is a helper class for sharing connections to the Modal server.

    It can create, cache, and close channels to the Modal server. This is useful since
    multiple ModalClientModal stubs may target the same server URL, in which case they
    should share the same connection.
    """

    def __init__(self, client: "modal.client._Client", metadata: dict[str, str] = {}):
        self._client = client
        # This metadata is injected into all requests on all channels created by this manager.
        self._metadata = metadata
        self._channels: dict[str, grpclib.client.Channel] = {}

    async def get_or_create_channel(self, server_url: str) -> grpclib.client.Channel:
        if server_url not in self._channels:
            self._channels[server_url] = create_channel(server_url, self._metadata)
            try:
                await connect_channel(self._channels[server_url])
            except OSError as exc:
                raise ConnectionError("Could not connect to the Modal server.") from exc
        return self._channels[server_url]

    def close(self):
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()


@cache
def _sym_db() -> SymbolDatabase:
    from google.protobuf.symbol_database import Default

    return Default()


class CustomProtoStatusDetailsCodec(StatusDetailsCodecBase):
    """grpclib compatible details codec.

    The server can encode the details using `google.rpc.Status` using grpclib's default codec and this custom codec
    can decode it into a `api_pb2.RPCStatus`.
    """

    def encode(
        self,
        status: Status,
        message: Optional[str],
        details: Optional[Sequence[Message]],
    ) -> bytes:
        details_proto = api_pb2.RPCStatus(code=status.value, message=message or "")
        if details is not None:
            for detail in details:
                detail_container = details_proto.details.add()
                detail_container.Pack(detail)
        return details_proto.SerializeToString()

    def decode(
        self,
        status: Status,
        message: Optional[str],
        data: bytes,
    ) -> Any:
        sym_db = _sym_db()
        details_proto = api_pb2.RPCStatus.FromString(data)

        details = []
        for detail_container in details_proto.details:
            # If we do not know how to decode an emssage, we'll ignore it.
            with contextlib.suppress(Exception):
                msg_type = sym_db.GetSymbol(detail_container.TypeName())
                detail = msg_type()
                detail_container.Unpack(detail)
                details.append(detail)
        return details


custom_detail_codec = CustomProtoStatusDetailsCodec()


def _should_bypass_proxy(host: str, port: Optional[int] = None) -> bool:
    """Check if the target host should bypass the proxy based on NO_PROXY/no_proxy."""
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    if not no_proxy:
        return False

    # Normalize bracketed IPv6 host (e.g., "[::1]" -> "::1")
    host_lower = host.strip("[]").lower()

    for entry in no_proxy.split(","):
        entry = entry.strip().lower()
        if not entry:
            continue
        if entry == "*":
            return True

        # Handle bracketed IPv6 entries (e.g., "[::1]" or "[::1]:8443")
        entry_port = None
        if entry.startswith("["):
            bracket_end = entry.find("]")
            if bracket_end != -1:
                port_suffix = entry[bracket_end + 1 :]
                entry = entry[1:bracket_end]
                if port_suffix.startswith(":") and port_suffix[1:].isdigit():
                    entry_port = int(port_suffix[1:])
        elif ":" in entry and entry.count(":") == 1:
            # Parse optional port from non-IPv6 entry (e.g., "example.com:8443")
            host_part, port_part = entry.rsplit(":", 1)
            if port_part.isdigit():
                entry_port = int(port_part)
                entry = host_part

        # If entry specifies a port, skip if it doesn't match
        if entry_port is not None and port is not None and entry_port != port:
            continue

        # Handle wildcard prefix (*.example.com)
        if entry.startswith("*."):
            suffix = entry[1:]  # ".example.com"
            if host_lower.endswith(suffix) or host_lower == entry[2:]:
                return True
        # Handle dot prefix (.example.com) — matches example.com and *.example.com
        elif entry.startswith("."):
            if host_lower == entry[1:] or host_lower.endswith(entry):
                return True
        # Exact match or suffix match
        elif host_lower == entry or host_lower.endswith("." + entry):
            return True

    return False


def create_channel(
    server_url: str,
    metadata: dict[str, str] = {},
) -> grpclib.client.Channel:
    """Creates a grpclib.Channel to be used by a GRPC stub.

    The given metadata dict is injected into all outgoing requests on this channel.
    """
    o = urllib.parse.urlparse(server_url)

    channel: grpclib.client.Channel
    grpc_config = grpclib.config.Configuration(
        http2_connection_window_size=64 * 1024 * 1024,  # 64 MiB
        http2_stream_window_size=64 * 1024 * 1024,  # 64 MiB
    )

    if o.scheme == "unix":
        channel = grpclib.client.Channel(path=o.path, config=grpc_config, status_details_codec=custom_detail_codec)
    elif o.scheme in ("http", "https"):
        target = o.netloc
        parts = target.split(":")
        assert 1 <= len(parts) <= 2, "Invalid target location: " + target
        ssl = o.scheme.endswith("s")
        host = parts[0]
        port = int(parts[1]) if len(parts) == 2 else 443 if ssl else 80

        proxy_url = config.get("grpc_proxy") if ssl else None
        if proxy_url and _should_bypass_proxy(host, port):
            proxy_url = None
        if proxy_url:
            channel = ProxyChannel(
                host,
                port,
                proxy_url=proxy_url,
                ssl=ssl,
                config=grpc_config,
                status_details_codec=custom_detail_codec,
            )
        else:
            channel = grpclib.client.Channel(
                host,
                port,
                ssl=ssl,
                config=grpc_config,
                status_details_codec=custom_detail_codec,
            )
    else:
        raise Exception(f"Unknown scheme: {o.scheme}")

    target = o.path if o.scheme == "unix" else o.netloc
    logger.debug(f"Connecting to {target} using scheme {o.scheme}")

    # Inject metadata for the client.
    async def send_request(event: grpclib.events.SendRequest) -> None:
        for k, v in metadata.items():
            event.metadata[k] = v

        logger.debug(f"Sending request to {event.method_name}")

    grpclib.events.listen(channel, grpclib.events.SendRequest, send_request)

    return channel


@retry(n_attempts=5, base_delay=0.1)
async def connect_channel(channel: grpclib.client.Channel):
    """Connect to socket and raise exceptions when there is a connection issue."""
    await channel.__connect__()


if typing.TYPE_CHECKING:
    import modal.client


async def unary_stream(
    method: "modal._grpc_client.UnaryStreamWrapper[RequestType, ResponseType]",
    request: RequestType,
    metadata: Optional[Any] = None,
) -> AsyncIterator[ResponseType]:
    # TODO: remove this, since we have a method now
    async for item in method.unary_stream(request, metadata):
        yield item


@dataclass(frozen=True)
class Retry:
    base_delay: float = 0.1
    max_delay: float = 1
    delay_factor: float = 2
    max_retries: Optional[int] = 3
    additional_status_codes: list = field(default_factory=list)
    attempt_timeout: Optional[float] = None  # timeout for each attempt
    total_timeout: Optional[float] = None  # timeout for the entire function call
    attempt_timeout_floor: float = 2.0  # always have at least this much timeout (only for total_timeout)
    warning_message: Optional[RetryWarningMessage] = None


async def retry_transient_errors(
    fn: "grpclib.client.UnaryUnaryMethod[RequestType, ResponseType]",
    req: RequestType,
    max_retries: Optional[int] = 3,
) -> ResponseType:
    """Minimum API version of _retry_transient_errors that works with grpclib.client.UnaryUnaryMethod.

    Used by modal server.
    """
    return await _retry_transient_errors(fn, req, retry=Retry(max_retries=max_retries))


def get_server_retry_policy(exc: Exception) -> Optional[api_pb2.RPCRetryPolicy]:
    """Get server retry policy."""
    if not isinstance(exc, GRPCError) or not exc.details:
        return None

    # Server should not set multiple retry instructions, but if there is more than one, pick the first one
    for entry in exc.details:
        if isinstance(entry, api_pb2.RPCRetryPolicy):
            return entry
    return None


def process_exception_before_retry(
    exc: Exception,
    final_attempt: bool,
    fn_name: str,
    n_retries: int,
    delay: float,
    idempotency_key: str,
):
    """Process exception before retry, used by `_retry_transient_errors`."""
    with suppress_tb_frame():
        if final_attempt:
            logger.debug(
                f"Final attempt failed with {repr(exc)} {n_retries=} {delay=} for {fn_name} ({idempotency_key[:8]})"
            )
            if isinstance(exc, OSError):
                raise ConnectionError(str(exc))
            elif isinstance(exc, asyncio.TimeoutError):
                raise ConnectionError(str(exc))
            else:
                raise exc

        if isinstance(exc, AttributeError) and "_write_appdata" not in str(exc):
            # StreamTerminatedError are not properly raised in grpclib<=0.4.7
            # fixed in https://github.com/vmagamedov/grpclib/issues/185
            # TODO: update to newer version (>=0.4.8) once stable
            # Also be sure to remove the AttributeError from the set of exceptions
            # we handle in the retry logic once we drop this check!
            raise exc

    logger.debug(f"Retryable failure {repr(exc)} {n_retries=} {delay=} for {fn_name} ({idempotency_key[:8]})")


async def _retry_transient_errors(
    fn: typing.Union[
        "modal._grpc_client.UnaryUnaryWrapper[RequestType, ResponseType]",
        "grpclib.client.UnaryUnaryMethod[RequestType, ResponseType]",
    ],
    req: RequestType,
    retry: Retry,
    metadata: Optional[list[tuple[str, str]]] = None,
) -> ResponseType:
    """Retry on transient gRPC failures with back-off until max_retries is reached.
    If max_retries is None, retry forever."""
    import modal._grpc_client

    if isinstance(fn, modal._grpc_client.UnaryUnaryWrapper):
        fn_callable = fn.direct
    elif isinstance(fn, grpclib.client.UnaryUnaryMethod):
        fn_callable = fn  # type: ignore
    else:
        raise ValueError("Only modal._grpc_client.UnaryUnaryWrapper and grpclib.client.UnaryUnaryMethod are supported")

    delay = retry.base_delay
    n_retries = 0
    n_throttled_retries = 0

    status_codes = [*RETRYABLE_GRPC_STATUS_CODES, *retry.additional_status_codes]

    idempotency_key = str(uuid.uuid4())

    t0 = time.time()
    last_server_retry_warning_time = None

    if retry.total_timeout is not None:
        total_deadline = t0 + retry.total_timeout
    else:
        total_deadline = None

    metadata = (metadata or []) + [("x-modal-timestamp", str(time.time()))]

    while True:
        attempt_metadata = [
            ("x-idempotency-key", idempotency_key),
            ("x-retry-attempt", str(n_retries)),
            ("x-throttle-retry-attempt", str(n_throttled_retries)),
            *metadata,
        ]
        if n_retries > 0:
            attempt_metadata.append(("x-retry-delay", str(time.time() - t0)))
        if n_throttled_retries > 0:
            attempt_metadata.append(("x-throttle-retry-delay", str(time.time() - t0)))

        timeouts = []
        if retry.attempt_timeout is not None:
            timeouts.append(retry.attempt_timeout)
        if total_deadline is not None:
            timeouts.append(max(total_deadline - time.time(), retry.attempt_timeout_floor))
        if timeouts:
            timeout = min(timeouts)  # In case the function provided both types of timeouts
        else:
            timeout = None

        try:
            with suppress_tb_frame():
                return await fn_callable(req, metadata=attempt_metadata, timeout=timeout)
        except (StreamTerminatedError, GRPCError, OSError, asyncio.TimeoutError, AttributeError) as exc:
            # Note that we only catch AttributeError to handle a specific case that works around a bug
            # in grpclib<=0.4.7. See above (search for `write_appdata`).

            # Server side instruction for retries
            max_throttle_wait: Optional[int] = config.get("max_throttle_wait")
            if (
                max_throttle_wait != 0
                and isinstance(exc, GRPCError)
                and (server_retry_policy := get_server_retry_policy(exc))
            ):
                server_delay = server_retry_policy.retry_after_secs

                now = time.time()

                # We check if the timeout will be reached **after** the sleep, so we can raise an error early
                # without needing to actually sleep.
                total_timeout_will_be_reached = (
                    retry.total_timeout is not None and (now + server_delay - t0) >= retry.total_timeout
                )
                max_throttle_will_be_reached = (
                    max_throttle_wait is not None and (now + server_delay - t0) >= max_throttle_wait
                )
                final_attempt = total_timeout_will_be_reached or max_throttle_will_be_reached

                with suppress_tb_frame():
                    process_exception_before_retry(
                        exc, final_attempt, fn.name, n_retries, server_delay, idempotency_key
                    )

                now = time.time()
                if last_server_retry_warning_time is None or (
                    now - last_server_retry_warning_time >= SERVER_RETRY_WARNING_TIME_INTERVAL
                ):
                    last_server_retry_warning_time = now
                    logger.warning(
                        f"Warning: Received {exc.status}{os.linesep}"
                        f"{exc.message}{os.linesep}"
                        f"Will retry in {server_delay:0.2f} seconds."
                    )

                n_throttled_retries += 1
                await asyncio.sleep(server_delay)
                continue

            # Client handles retry
            if isinstance(exc, GRPCError) and exc.status not in status_codes:
                raise exc
            if retry.max_retries is not None and n_retries >= retry.max_retries:
                final_attempt = True
            elif total_deadline is not None and time.time() + delay + retry.attempt_timeout_floor >= total_deadline:
                final_attempt = True
            else:
                final_attempt = False

            with suppress_tb_frame():
                process_exception_before_retry(exc, final_attempt, fn.name, n_retries, delay, idempotency_key)

            n_retries += 1

            if (
                retry.warning_message
                and n_retries % retry.warning_message.warning_interval == 0
                and isinstance(exc, GRPCError)
                and exc.status in retry.warning_message.errors_to_warn_for
            ):
                logger.warning(retry.warning_message.message)

            await asyncio.sleep(delay)
            delay = min(delay * retry.delay_factor, retry.max_delay)


def find_free_port() -> int:
    """
    Find a free TCP port, useful for testing.

    WARN: if a returned free port is not bound immediately by the caller, that same port
    may be returned in subsequent calls to this function, potentially creating port collisions.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_proto_oneof(message: Message, oneof_group: str) -> Optional[Message]:
    oneof_field = message.WhichOneof(oneof_group)
    if oneof_field is None:
        return None

    return getattr(message, oneof_field)
