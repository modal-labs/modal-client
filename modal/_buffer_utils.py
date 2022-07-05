import time

from grpc import StatusCode
from grpc.aio import AioRpcError

from .config import logger

INITIAL_STREAM_SIZE = 5


async def buffered_rpc_read(fn, request, timeout=None):
    """Reads from buffered method."""

    fn_name = getattr(fn, "__name__", None)  # for logging
    t0 = time.time()

    while True:
        if timeout is not None:
            request.timeout = timeout - (time.time() - t0)
        else:
            request.timeout = 60

        try:
            return await fn(request)
        except AioRpcError as exc:
            if exc.code() not in (StatusCode.UNAVAILABLE, StatusCode.DEADLINE_EXCEEDED, StatusCode.RESOURCE_EXHAUSTED):
                raise
            if timeout is not None and (time.time() - t0) > timeout:
                raise

            logger.debug(f"{fn_name}: buffer read timed out. Retrying.")
            # TODO: maybe have some kind of exponential back-off.
