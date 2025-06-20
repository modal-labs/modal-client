# Copyright Modal Labs 2022
import asyncio
import contextlib
import platform
import socket
import time
import typing
import urllib.parse
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    TypeVar,
)

import grpclib.client
import grpclib.config
import grpclib.events
import grpclib.protocol
import grpclib.stream
from google.protobuf.message import Message
from grpclib import GRPCError, Status
from grpclib.exceptions import StreamTerminatedError
from grpclib.protocol import H2Protocol

from modal.exception import AuthError, ConnectionError
from modal_version import __version__

from .logger import logger

RequestType = TypeVar("RequestType", bound=Message)
ResponseType = TypeVar("ResponseType", bound=Message)

if typing.TYPE_CHECKING:
    import modal.client

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


RETRYABLE_GRPC_STATUS_CODES = [
    Status.DEADLINE_EXCEEDED,
    Status.UNAVAILABLE,
    Status.CANCELLED,
    Status.INTERNAL,
    Status.UNKNOWN,
]


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
        # Warning: This metadata is shared across all channels! If the metadata is mutated
        # in one `create_channel` call, the mutation will be reflected in all channels.
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


def create_channel(
    server_url: str,
    metadata: dict[str, str] = {},
) -> grpclib.client.Channel:
    """Creates a grpclib.Channel to be used by a GRPC stub.

    Note that this function mutates the given metadata argument by adding an x-modal-auth-token
    if one is present in the trailing metadata of any response.
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

    async def recv_initial_metadata(initial_metadata: grpclib.events.RecvInitialMetadata) -> None:
        # If we receive an auth token from the server, include it in all future requests.
        # TODO(nathan): This isn't perfect because the metadata isn't propagated when the
        # process is forked and a new channel is created. This is OK for now since this
        # token is only used by the experimental input plane
        if token := initial_metadata.metadata.get("x-modal-auth-token"):
            metadata["x-modal-auth-token"] = str(token)

    async def recv_trailing_metadata(trailing_metadata: grpclib.events.RecvTrailingMetadata) -> None:
        if token := trailing_metadata.metadata.get("x-modal-auth-token"):
            metadata["x-modal-auth-token"] = str(token)

    grpclib.events.listen(channel, grpclib.events.SendRequest, send_request)
    grpclib.events.listen(channel, grpclib.events.RecvInitialMetadata, recv_initial_metadata)
    grpclib.events.listen(channel, grpclib.events.RecvTrailingMetadata, recv_trailing_metadata)

    return channel


async def connect_channel(channel: grpclib.client.Channel):
    """Connects socket (potentially raising errors raising to connectivity."""
    await channel.__connect__()


if typing.TYPE_CHECKING:
    import modal.client


async def unary_stream(
    method: "modal.client.UnaryStreamWrapper[RequestType, ResponseType]",
    request: RequestType,
    metadata: Optional[Any] = None,
) -> AsyncIterator[ResponseType]:
    # TODO: remove this, since we have a method now
    async for item in method.unary_stream(request, metadata):
        yield item


async def retry_transient_errors(
    fn: "modal.client.UnaryUnaryWrapper[RequestType, ResponseType]",
    *args,
    base_delay: float = 0.1,
    max_delay: float = 1,
    delay_factor: float = 2,
    max_retries: Optional[int] = 3,
    additional_status_codes: list = [],
    attempt_timeout: Optional[float] = None,  # timeout for each attempt
    total_timeout: Optional[float] = None,  # timeout for the entire function call
    attempt_timeout_floor=2.0,  # always have at least this much timeout (only for total_timeout)
    retry_warning_message: Optional[RetryWarningMessage] = None,
    metadata: list[tuple[str, str]] = [],
) -> ResponseType:
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
        attempt_metadata = [
            ("x-idempotency-key", idempotency_key),
            ("x-retry-attempt", str(n_retries)),
            *metadata,
        ]
        if n_retries > 0:
            attempt_metadata.append(("x-retry-delay", str(time.time() - t0)))
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
            return await fn(*args, metadata=attempt_metadata, timeout=timeout)
        except (StreamTerminatedError, GRPCError, OSError, asyncio.TimeoutError, AttributeError) as exc:
            if isinstance(exc, GRPCError) and exc.status not in status_codes:
                if exc.status == Status.UNAUTHENTICATED:
                    raise AuthError(exc.message)
                else:
                    raise exc

            if max_retries is not None and n_retries >= max_retries:
                final_attempt = True
            elif total_deadline is not None and time.time() + delay + attempt_timeout_floor >= total_deadline:
                final_attempt = True
            else:
                final_attempt = False

            if final_attempt:
                logger.debug(
                    f"Final attempt failed with {repr(exc)} {n_retries=} {delay=} "
                    f"{total_deadline=} for {fn.name} ({idempotency_key[:8]})"
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
                raise exc

            logger.debug(f"Retryable failure {repr(exc)} {n_retries=} {delay=} for {fn.name} ({idempotency_key[:8]})")

            n_retries += 1

            if (
                retry_warning_message
                and n_retries % retry_warning_message.warning_interval == 0
                and isinstance(exc, GRPCError)
                and exc.status in retry_warning_message.errors_to_warn_for
            ):
                logger.warning(retry_warning_message.message)

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
