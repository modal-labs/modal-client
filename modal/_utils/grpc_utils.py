# Copyright Modal Labs 2022
import asyncio
import contextlib
import platform
import socket
import time
import urllib.parse
import uuid
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Optional,
    TypeVar,
)

import grpclib.client
import grpclib.config
import grpclib.events
from google.protobuf.message import Message
from grpclib import GRPCError, Status
from grpclib.exceptions import StreamTerminatedError
from grpclib.protocol import H2Protocol

from modal_version import __version__

from .logger import logger

# Monkey patches grpclib to have a Modal User Agent header.
grpclib.client.USER_AGENT = "modal-client/{version} ({sys}; {py}/{py_ver})'".format(
    version=__version__,
    sys=platform.system(),
    py=platform.python_implementation(),
    py_ver=platform.python_version(),
).lower()


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


_SendType = TypeVar("_SendType")
_RecvType = TypeVar("_RecvType")

RETRYABLE_GRPC_STATUS_CODES = [
    Status.DEADLINE_EXCEEDED,
    Status.UNAVAILABLE,
    Status.CANCELLED,
    Status.INTERNAL,
]


def create_channel(
    server_url: str,
    metadata: Dict[str, str] = {},
) -> grpclib.client.Channel:
    """Creates a grpclib.Channel.

    Either to be used directly by a GRPC stub, or indirectly used through the channel pool.
    See `create_channel`.
    """
    o = urllib.parse.urlparse(server_url)

    channel: grpclib.client.Channel
    config = grpclib.config.Configuration(
        http2_connection_window_size=64 * 1024 * 1024,  # 64 MiB
        http2_stream_window_size=64 * 1024 * 1024,  # 64 MiB
    )

    if o.scheme == "unix":
        channel = grpclib.client.Channel(path=o.path, config=config)  # probably pointless to use a pool ever
    elif o.scheme in ("http", "https"):
        target = o.netloc
        parts = target.split(":")
        assert 1 <= len(parts) <= 2, "Invalid target location: " + target
        ssl = o.scheme.endswith("s")
        host = parts[0]
        port = int(parts[1]) if len(parts) == 2 else 443 if ssl else 80
        channel = grpclib.client.Channel(host, port, ssl=ssl, config=config)
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


async def unary_stream(
    method: grpclib.client.UnaryStreamMethod[_SendType, _RecvType],
    request: _SendType,
    metadata: Optional[Any] = None,
) -> AsyncIterator[_RecvType]:
    """Helper for making a unary-streaming gRPC request."""
    async with method.open(metadata=metadata) as stream:
        await stream.send_message(request, end=True)
        async for item in stream:
            yield item


async def retry_transient_errors(
    fn,
    *args,
    base_delay: float = 0.1,
    max_delay: float = 1,
    delay_factor: float = 2,
    max_retries: Optional[int] = 3,
    additional_status_codes: list = [],
    attempt_timeout: Optional[float] = None,  # timeout for each attempt
    total_timeout: Optional[float] = None,  # timeout for the entire function call
    attempt_timeout_floor=2.0,  # always have at least this much timeout (only for total_timeout)
):
    """Retry on transient gRPC failures with back-off until max_retries is reached.
    If max_retries is None, retry forever."""

    delay = base_delay
    n_retries = 0

    status_codes = [*RETRYABLE_GRPC_STATUS_CODES, *additional_status_codes]

    idempotency_key = str(uuid.uuid4())

    t0 = time.time()
    if total_timeout is not None:
        total_deadline = t0 + total_timeout
    else:
        total_deadline = None

    while True:
        metadata = [("x-idempotency-key", idempotency_key), ("x-retry-attempt", str(n_retries))]
        if n_retries > 0:
            metadata.append(("x-retry-delay", str(time.time() - t0)))
        timeouts = []
        if attempt_timeout is not None:
            timeouts.append(attempt_timeout)
        if total_timeout is not None:
            timeouts.append(max(total_deadline - time.time(), attempt_timeout_floor))
        if timeouts:
            timeout = min(timeouts)  # In case the function provided both types of timeouts
        else:
            timeout = None
        try:
            return await fn(*args, metadata=metadata, timeout=timeout)
        except (StreamTerminatedError, GRPCError, socket.gaierror, asyncio.TimeoutError) as exc:
            if isinstance(exc, GRPCError) and exc.status not in status_codes:
                raise exc

            if max_retries is not None and n_retries >= max_retries:
                raise exc

            if total_deadline and time.time() + delay + attempt_timeout_floor >= total_deadline:
                # no point sleeping if that's going to push us past the deadline
                raise exc

            n_retries += 1

            await asyncio.sleep(delay)
            delay = min(delay * delay_factor, max_delay)


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
