import asyncio
import enum
import re
import time

from grpc import StatusCode
from grpc.aio import AioRpcError, Channel
from sentry_sdk import capture_exception

from modal_utils.server_connection import GRPCConnectionFactory

from .async_utils import TaskContext
from .logger import logger

RETRYABLE_GRPC_STATUS_CODES = [
    StatusCode.DEADLINE_EXCEEDED,
    StatusCode.UNAVAILABLE,
    StatusCode.INTERNAL,
]


class RPCType(enum.Enum):
    UNARY_UNARY = 1
    UNARY_STREAM = 2
    STREAM_UNARY = 3
    STREAM_STREAM = 4


class ChannelStruct:
    def __init__(self, channel: Channel) -> None:
        self.channel = channel
        self.n_concurrent_requests = 0
        self.created_at = time.time()
        self.last_active = self.created_at
        self._callables = {}
        self._constructors = {
            RPCType.UNARY_UNARY: channel.unary_unary,
            RPCType.UNARY_STREAM: channel.unary_stream,
            RPCType.STREAM_UNARY: channel.stream_unary,
            RPCType.STREAM_STREAM: channel.stream_stream,
        }

    def closed(self) -> bool:
        return self.channel._channel.closed()

    def get_method(self, rpc_type, method, request_serializer, response_deserializer):
        if (rpc_type, method) not in self._callables:
            self._callables[(rpc_type, method)] = self._constructors[rpc_type](
                method, request_serializer, response_deserializer
            )
        return self._callables[(rpc_type, method)]


class ChannelPool:
    """Use multiple channels under the hood. A drop-in replacement for the GRPC channel.

    The ALB in AWS limits the number of streams per connection to 128.
    This is super annoying and means we can't put every request on the same channel.
    As a dumb workaround, we use a pool of channels.

    This object is not thread-safe.
    """

    # How long to keep alive unused channels in the pool, before closing them.
    CHANNEL_KEEP_ALIVE = 40

    # Maximum number of concurrent requests per channel.
    MAX_REQUESTS_PER_CHANNEL = 64

    # Don't accept more connections on this channel after this many seconds
    MAX_CHANNEL_LIFETIME = 30

    def __init__(self, task_context: TaskContext, conn_factory: GRPCConnectionFactory) -> None:
        # Only used by start()
        self._task_context = task_context

        # Threadsafe because it is read-only
        self._conn_factory = conn_factory

        # Protects the channels list below
        self._lock = asyncio.Lock()
        self._channels: list[ChannelStruct] = []

    async def _purge_channels(self):
        to_close: list[ChannelStruct] = []
        async with self._lock:
            for ch in self._channels:
                now = time.time()
                inactive_time = now - ch.last_active
                if ch.closed():
                    logger.debug("Purging channel that's already closed.")
                    self._channels.remove(ch)
                elif ch.n_concurrent_requests > 0:
                    ch.last_active = now
                elif inactive_time >= self.CHANNEL_KEEP_ALIVE:
                    logger.debug(f"Closing channel of age {now - ch.created_at}s, inactive for {inactive_time}s")
                    to_close.append(ch)
            for ch in to_close:
                self._channels.remove(ch)
        for ch in to_close:
            await ch.channel.close()

    async def start(self) -> None:
        self._task_context.infinite_loop(self._purge_channels, sleep=10.0)

    async def _get_channel(self) -> ChannelStruct:
        async with self._lock:
            eligible_channels = [
                ch
                for ch in self._channels
                if ch.n_concurrent_requests < self.MAX_REQUESTS_PER_CHANNEL
                and time.time() - ch.created_at < self.MAX_CHANNEL_LIFETIME
            ]
            if eligible_channels:
                ch = eligible_channels[0]
            else:
                channel = self._conn_factory.create()
                ch = ChannelStruct(channel)
                self._channels.append(ch)
                n_conc_reqs = [ch.n_concurrent_requests for ch in self._channels]
                n_conc_reqs_str = ", ".join(str(z) for z in n_conc_reqs)
                logger.debug(f"Pool: Added new channel (concurrent requests: {n_conc_reqs_str}")

        return ch

    async def close(self) -> None:
        logger.debug("Pool: Shutting down")
        for ch in self._channels:
            await ch.channel.close()
        self._channels = []

    def size(self) -> int:
        return len(self._channels)

    def _wrap_base(self, coro, method):
        # Put a name on the coroutine so that stack traces are a bit more readable
        coro.__name__ = "__wrapped_" + re.sub(r"\W", "", method)
        coro.__qualname__ = "ChannelPool." + coro.__name__
        return coro

    def _wrap_function(self, rpc_type, method, request_serializer, response_deserializer):
        async def coro(req, **kwargs):
            ch = await self._get_channel()
            ch.n_concurrent_requests += 1
            try:
                fn = ch.get_method(rpc_type, method, request_serializer, response_deserializer)
                ret = await fn(req, **kwargs)
            finally:
                ch.n_concurrent_requests -= 1
            return ret

        return self._wrap_base(coro, method)

    def _wrap_generator(self, rpc_type, method, request_serializer, response_deserializer):
        async def coro_gen(req, **kwargs):
            ch = await self._get_channel()
            ch.n_concurrent_requests += 1
            try:
                fn = ch.get_method(rpc_type, method, request_serializer, response_deserializer)
                gen = fn(req, **kwargs)
                async for ret in gen:
                    yield ret
            finally:
                ch.n_concurrent_requests -= 1

        return self._wrap_base(coro_gen, method)

    def unary_unary(self, method, request_serializer, response_deserializer):
        return self._wrap_function(RPCType.UNARY_UNARY, method, request_serializer, response_deserializer)

    def stream_unary(self, method, request_serializer, response_deserializer):
        return self._wrap_function(RPCType.STREAM_UNARY, method, request_serializer, response_deserializer)

    def unary_stream(self, method, request_serializer, response_deserializer):
        return self._wrap_generator(RPCType.UNARY_STREAM, method, request_serializer, response_deserializer)

    def stream_stream(self, method, request_serializer, response_deserializer):
        return self._wrap_generator(RPCType.STREAM_STREAM, method, request_serializer, response_deserializer)


async def retry_transient_errors(fn, *args, base_delay=0.1, max_delay=1, delay_factor=2, max_retries=3):
    """Retry on transient gRPC failures with back-off until max_retries is reached.
    If max_retries is None, retry forever."""

    delay = base_delay
    n_retries = 0

    while True:
        try:
            return await fn(*args)
        except AioRpcError as exc:
            if exc.code() in RETRYABLE_GRPC_STATUS_CODES:
                if max_retries is not None and n_retries >= max_retries:
                    raise
                n_retries += 1
                capture_exception(exc)
                await asyncio.sleep(delay)
                delay = min(delay * delay_factor, max_delay)
            else:
                raise
