# Copyright Modal Labs 2022
import asyncio
import contextlib
import socket
import time
import platform
import urllib.parse
import uuid
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
)

import grpclib.events
import grpclib.client
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


class ChannelPool(grpclib.client.Channel):
    """Use multiple channels under the hood. A drop-in replacement for the grpclib Channel.

    The main reason is to get around limitations with TCP connections over the internet,
    in particular idle timeouts, but also the fact that ALBs in AWS limits the number of
    streams per connection to 128.

    The algorithm is very simple. It reuses the last subchannel as long as it has had less
    than 64 requests or if it was created less than 30s ago. It closes any subchannel that
    hits 90s age. This means requests using the ChannelPool can't be longer than 60s.
    """

    _max_requests: int
    _max_lifetime: float
    _max_active: float
    _subchannels: List[Subchannel]

    def __init__(
        self,
        *args,
        max_requests=64,  # Maximum number of total requests per subchannel
        max_active=30,  # Don't accept more connections on the subchannel after this many seconds
        max_lifetime=90,  # Close subchannel after this many seconds
        **kwargs,
    ):
        self._subchannels = []
        self._max_requests = max_requests
        self._max_active = max_active
        self._max_lifetime = max_lifetime
        super().__init__(*args, **kwargs)

    async def __connect__(self):
        now = time.time()
        # Remove any closed subchannels
        while len(self._subchannels) > 0 and not self._subchannels[-1].connected():
            self._subchannels.pop()

        # Close and delete any subchannels that are past their lifetime
        while len(self._subchannels) > 0 and now - self._subchannels[0].created_at > self._max_lifetime:
            self._subchannels.pop(0).protocol.processor.close()

        # See if we can reuse the last subchannel
        create_subchannel = None
        if len(self._subchannels) > 0:
            if self._subchannels[-1].created_at < now - self._max_active:
                # Don't reuse subchannel that's too old
                create_subchannel = True
            elif self._subchannels[-1].requests > self._max_requests:
                create_subchannel = True
            else:
                create_subchannel = False
        else:
            create_subchannel = True

        # Create new if needed
        # There's a theoretical race condition here.
        # This is harmless but may lead to superfluous protocols.
        if create_subchannel:
            protocol = await self._create_connection()
            self._subchannels.append(Subchannel(protocol))

        self._subchannels[-1].requests += 1
        return self._subchannels[-1].protocol

    def close(self) -> None:
        while len(self._subchannels) > 0:
            self._subchannels.pop(0).protocol.processor.close()

    def __del__(self) -> None:
        if len(self._subchannels) > 0:
            logger.warning("Channel pool not properly closed")


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
    *,
    inject_tracing_context: Optional[Callable[[Dict[str, str]], None]] = None,
    use_pool: Optional[bool] = None,  # If None, inferred from the scheme
) -> grpclib.client.Channel:
    """Creates a grpclib.Channel.

    Either to be used directly by a GRPC stub, or indirectly used through the channel pool.
    See `create_channel`.
    """
    o = urllib.parse.urlparse(server_url)

    if use_pool is None:
        use_pool = o.scheme in ("http", "https")

    channel_cls: Type[grpclib.client.Channel]
    if use_pool:
        channel_cls = ChannelPool
    else:
        channel_cls = grpclib.client.Channel

    channel: grpclib.client.Channel
    if o.scheme == "unix":
        channel = channel_cls(path=o.path)  # probably pointless to use a pool ever
    elif o.scheme in ("http", "https"):
        target = o.netloc
        parts = target.split(":")
        assert 1 <= len(parts) <= 2, "Invalid target location: " + target
        ssl = o.scheme.endswith("s")
        host = parts[0]
        port = int(parts[1]) if len(parts) == 2 else 443 if ssl else 80
        channel = channel_cls(host, port, ssl=ssl)
    else:
        raise Exception(f"Unknown scheme: {o.scheme}")

    logger.debug(f"Connecting to {o.netloc} using scheme {o.scheme}")

    # Inject metadata for the client.
    async def send_request(event: grpclib.events.SendRequest) -> None:
        for k, v in metadata.items():
            event.metadata[k] = v

        if inject_tracing_context is not None:
            inject_tracing_context(cast(Dict[str, str], event.metadata))

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
    ignore_errors: list = [],
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
    """Find a free TCP port, useful for testing."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def patch_mock_servicer(cls):
    """Patches all unimplemented abstract methods in a mock servicer."""

    async def fallback(self, stream) -> None:
        raise GRPCError(Status.UNIMPLEMENTED, "Not implemented in mock servicer " + repr(cls))

    # Fill in the remaining methods on the class
    for name in dir(cls):
        method = getattr(cls, name)
        if getattr(method, "__isabstractmethod__", False):
            setattr(cls, name, fallback)

    cls.__abstractmethods__ = frozenset()
    return cls


def get_proto_oneof(message: Message, oneof_group: str) -> Optional[Message]:
    oneof_field = message.WhichOneof(oneof_group)
    if oneof_field is None:
        return None

    return getattr(message, oneof_field)
