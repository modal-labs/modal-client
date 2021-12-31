import asyncio
import time
import uuid

from ._async_utils import retry
from ._grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIME_BUFFER
from .config import logger
from .proto import api_pb2

INITIAL_STREAM_SIZE = 5


async def buffered_rpc_write(fn, request):
    """Writes requests to buffered RPC method."""

    fn_name = fn.__name__  # for logging

    # TODO: the idempotency tokens are not currently used by the server
    # request.buffer_req.idempotency_key = str(uuid.uuid4())

    while True:
        response = await retry(fn)(request)

        if response.status == api_pb2.BufferWriteResponse.BufferWriteStatus.SUCCESS:
            return response

        logger.debug(f"{fn_name}: no space left in buffer. Sleeping.")
        # TODO: maybe have some kind of exponential back-off and timeout.
        await asyncio.sleep(1)


async def buffered_rpc_read(fn, request, buffer_id, timeout=None):
    """Reads from buffered method."""
    request.buffer_req.buffer_id = buffer_id

    fn_name = fn.__name__  # for logging
    t0 = time.time()

    while True:
        next_timeout = BLOCKING_REQUEST_TIMEOUT

        if timeout is not None:
            time_remaining = timeout - (time.time() - t0)
            next_timeout = min(next_timeout, time_remaining)

        request.buffer_req.timeout = next_timeout
        response = await retry(fn)(request, timeout=next_timeout + GRPC_REQUEST_TIME_BUFFER)

        if response.status == api_pb2.BufferReadResponse.BufferReadStatus.SUCCESS:
            return response

        if timeout is not None and (time.time() - t0) > timeout:
            return response

        logger.debug(f"{fn_name}: buffer read timed out. Retrying.")
        # TODO: maybe have some kind of exponential back-off.
