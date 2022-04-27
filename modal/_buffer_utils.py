import asyncio
import time

from grpc import StatusCode
from grpc.aio import AioRpcError

from .config import logger

INITIAL_STREAM_SIZE = 5


async def buffered_rpc_write(fn, request):
    """Writes requests to buffered RPC method."""

    fn_name = fn.__name__  # for logging

    # TODO: the idempotency tokens are not currently used by the server
    # request.buffer_req.idempotency_key = str(uuid.uuid4())

    while True:
        try:
            await fn(request)
            return
        except AioRpcError as exc:
            if exc.code() == StatusCode.UNAVAILABLE:
                logger.debug(f"{fn_name}: no space left in buffer. Sleeping.")
                # TODO: maybe have some kind of exponential back-off and timeout.
                await asyncio.sleep(1)
            else:
                raise


async def buffered_rpc_read(fn, request, timeout=None, warn_on_cancel=True):
    """Reads from buffered method."""

    fn_name = fn.__name__  # for logging
    t0 = time.time()

    while True:
        if timeout is not None:
            request.timeout = timeout - (time.time() - t0)
        else:
            request.timeout = 60

        try:
            return await fn(request)
        except AioRpcError as exc:
            if exc.code() not in (StatusCode.DEADLINE_EXCEEDED, StatusCode.RESOURCE_EXHAUSTED):
                raise
            if timeout is not None and (time.time() - t0) > timeout:
                raise

            logger.debug(f"{fn_name}: buffer read timed out. Retrying.")
            # TODO: maybe have some kind of exponential back-off.
