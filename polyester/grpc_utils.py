import abc
import asyncio
import enum
import re
import time
import traceback

import grpc.aio

from .async_utils import add_traceback, chunk_generator
from .config import logger

HTTP_TIME_BUFFER = 5
GRPC_REQUEST_TIME_BUFFER = 5

HTTP_IDLE_TIMEOUT = 60  # EC2 ALBs don't seem trustworthy above this
GRPC_REQUEST_TIMEOUT = HTTP_IDLE_TIMEOUT - HTTP_TIME_BUFFER  # Timeout enforced on all requests
BLOCKING_REQUEST_TIMEOUT = (
    GRPC_REQUEST_TIMEOUT - GRPC_REQUEST_TIME_BUFFER
)  # Timeout used for blocking requests on the app layer

MAX_CHANNEL_LIFETIME = 180


def not_implemented_callable(*args, **kwargs):
    raise NotImplementedError


class RPCType(enum.Enum):
    UNARY_UNARY = 1
    UNARY_STREAM = 2
    STREAM_UNARY = 3
    STREAM_STREAM = 4


class ChannelStruct:
    def __init__(self, channel):
        self.channel = channel
        self.n_concurrent_requests = 0
        self.created_at = time.time()
        self._callables = {}
        self._constructors = {
            RPCType.UNARY_UNARY: channel.unary_unary,
            RPCType.UNARY_STREAM: channel.unary_stream,
            RPCType.STREAM_UNARY: channel.stream_unary,
            RPCType.STREAM_STREAM: channel.stream_stream,
        }

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

    This also disconnects channels without concurrent requests, which is good because
    we won't get idle timeouts.

    TODO: we should build in something that detects if a channel is closed and then
    purges it.
    """

    def __init__(self, conn_factory, max_channel_lifetime=MAX_CHANNEL_LIFETIME):
        self._conn_factory = conn_factory
        self._max_requests_per_channel = 64
        self._channels = []
        self._lock = asyncio.Lock()
        self._max_channel_lifetime = max_channel_lifetime  # Don't reuse channels after this

    async def _purge_channels(self):
        to_close = []
        async with self._lock:
            for ch in self._channels:
                age = time.time() - ch.created_at
                if ch.n_concurrent_requests <= 0 and age >= self._max_channel_lifetime:
                    logger.debug(f"Closing old channel of age {age}s")
                    to_close.append(ch)
                elif age >= 2 * self._max_channel_lifetime:
                    logger.warning(f"Channel is age {age}s but has {ch.n_concurrent_requests} concurrent requests")
            for ch in to_close:
                self._channels.remove(ch)
        for ch in to_close:
            await ch.channel.close()

    async def start(self):
        async def purge_channels_loop():
            while True:
                await self._purge_channels()
                await asyncio.sleep(10.0)

        loop = asyncio.get_event_loop()
        self.purge_task = loop.create_task(purge_channels_loop())

    async def _get_channel(self):
        async with self._lock:
            eligible_channels = [
                ch
                for ch in self._channels
                if ch.n_concurrent_requests < self._max_requests_per_channel
                and ch.created_at + self._max_channel_lifetime >= time.time()
            ]
            if eligible_channels:
                ch = eligible_channels[0]
            else:
                channel = await self._conn_factory.create()
                ch = ChannelStruct(channel)
                self._channels.append(ch)
                n_conc_reqs = [ch.n_concurrent_requests for ch in self._channels]
                n_conc_reqs_str = ", ".join(str(z) for z in n_conc_reqs)
                logger.debug(f"Pool: Added new channel (concurrent requests: {n_conc_reqs_str}")

        return ch

    async def close(self):
        logger.debug("Pool: Shutting down")
        for ch in self._channels:
            await ch.channel.close()
        self._channels = []
        if self.purge_task:
            self.purge_task.cancel()

    def _update_kwargs(self, kwargs):
        # Override timeout (or set it if it's not set) and cap it to the channel lifetime
        new_timeout = min(GRPC_REQUEST_TIMEOUT, kwargs.get("timeout", 1e24))
        return {**kwargs, **dict(timeout=new_timeout)}

    def _wrap_base(self, coro, method):
        # grpcio wants a sync function that returns an coroutine (or a async generator)
        def f(req, **kwargs):
            ret = coro(req, **self._update_kwargs(kwargs))
            return add_traceback(ret, method)  # gRPC seems to suppress tracebacks in many cases

        # Put a name on the coroutine so that stack traces are a bit more readable
        f.__name__ = "__wrapped_" + re.sub(r"\W", "", method)
        f.__qualname__ = "ChannelPool." + f.__name__

        return f

    async def _chunk_generator(self, req, rpc_type, method):
        """Chunks any stream _inputs_ into multiple requests.

        If the input to a gRPC call is a generator, that request can hog up
        channels. To avoid this, we break up the generators into multiple
        requests that are each below the idle timeout.

        Note that we cannot do this automatically for stream outputs since
        we don't have control of the stream (since it's generated on the server).
        """
        if rpc_type in [RPCType.UNARY_UNARY, RPCType.UNARY_STREAM]:
            # Scalar input, nothing to do
            yield req
        else:
            # TODO: there's a bit of a problem if we need to rely on say the first message to send some
            # valuable context to the server. For instance let's we want to stream logs from the server,
            # but we also want to be able to provide with some updates such as "client is closing now",
            # then we need a stream-stream method, but with a special "first" message to start off the
            # interchange. However if the stream gets chopped up, then we lose the ability to send those
            # first messages.
            async for sub_req in chunk_generator(req, BLOCKING_REQUEST_TIMEOUT):
                yield sub_req

    def _wrap_function(self, rpc_type, method, request_serializer, response_deserializer):
        async def coro(req, **kwargs):
            async for sub_req in self._chunk_generator(req, rpc_type, method):
                ch = await self._get_channel()
                ch.n_concurrent_requests += 1
                try:
                    fn = ch.get_method(rpc_type, method, request_serializer, response_deserializer)
                    ret = await fn(sub_req, **kwargs)
                finally:
                    ch.n_concurrent_requests -= 1
            return ret

        return self._wrap_base(coro, method)

    def _wrap_generator(self, rpc_type, method, request_serializer, response_deserializer):
        async def coro_gen(req, **kwargs):
            async for sub_req in self._chunk_generator(req, rpc_type, method):
                ch = await self._get_channel()
                ch.n_concurrent_requests += 1
                try:
                    fn = ch.get_method(rpc_type, method, request_serializer, response_deserializer)
                    gen = fn(sub_req, **kwargs)
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
