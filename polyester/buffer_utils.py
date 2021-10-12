import asyncio
import uuid

from .async_utils import retry, create_task
from .config import logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .proto import api_pb2

INITIAL_STREAM_SIZE = 5


async def buffered_write_all(fn, request_gen, /, send_EOF=True):
    """Writes all requests to buffered method in a TCP sliding window-ish fashion. Adds an EOF token at the end."""

    drain_event = asyncio.Event()
    requests = []
    # We want to asynchronously pull from the generator, while still allowing requests
    # to be streamed back.
    async def drain_generator():
        async for r in request_gen:
            requests.append(r)
            drain_event.set()

    drain_task = create_task(drain_generator())

    next_idx_to_send = 0
    # `max_idx_to_send` is updated based on how much space the server says is left,
    # but starts off at a default value of 100.
    max_idx_to_send = INITIAL_STREAM_SIZE
    idempotency_key = str(uuid.uuid4())
    # container_entrypoint map can use this in theory to write to multiple output buffers at once.
    buffer_ids = set()
    fn_name = fn.__name__  # for logging

    async def write_request_generator():
        for idx in range(next_idx_to_send, max_idx_to_send):
            request = requests[idx]
            buffer_ids.add(request.buffer_req.buffer_id)
            request.buffer_req.idempotency_key = idempotency_key
            request.buffer_req.idx = idx
            yield request

    async def eof_request_generator():
        for buffer_id in buffer_ids:
            # send EOF
            req_type = type(requests[0])
            eof_item = api_pb2.BufferItem(EOF=True)
            yield req_type(buffer_req=api_pb2.BufferWriteRequest(item=eof_item, buffer_id=buffer_id))

    while next_idx_to_send < len(requests) or not drain_task.done():
        if next_idx_to_send == len(requests):
            logger.debug(f"{fn_name}: no more requests to send, but generator is not done. Waiting.")
            drain_event.clear()
            # wait for more requests
            await drain_event.wait()
            continue

        max_idx_to_send = min(len(requests), max_idx_to_send)
        if next_idx_to_send == max_idx_to_send:
            # no space left.
            # TODO: maybe have some kind of exponential back-off.
            logger.debug(f"{fn_name}: no space left in buffer. Spinning.")
            await asyncio.sleep(1)
            max_idx_to_send = next_idx_to_send + INITIAL_STREAM_SIZE
            continue

        # TODO: the idempotency tokens are not currently used by the server
        response = await retry(fn)(write_request_generator())

        logger.debug(
            f"{fn_name}: sent range [{next_idx_to_send}, {max_idx_to_send}). Received {response.num_pushed=} {response.space_left=}."
        )

        next_idx_to_send += response.num_pushed
        max_idx_to_send = next_idx_to_send + response.space_left

    if buffer_ids and send_EOF:
        await retry(fn)(eof_request_generator())


async def buffered_read_all(fn, request, buffer_id, /, read_until_EOF=True):
    """Reads from buffered method until EOF has been reached or timeout."""

    request.buffer_req.buffer_id = buffer_id
    request.buffer_req.timeout = BLOCKING_REQUEST_TIMEOUT

    while True:
        async for buffer_response in fn(request, timeout=GRPC_REQUEST_TIMEOUT):
            item = buffer_response.item
            if item.EOF:
                return
            yield item

        if not read_until_EOF:
            break

        # TODO: maybe have some kind of exponential back-off.
        await asyncio.sleep(1)
