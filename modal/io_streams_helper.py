# Copyright Modal Labs 2024
import asyncio
from typing import AsyncIterator, Callable, TypeVar

from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal.exception import ClientClosed

from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES

T = TypeVar("T")


async def consume_stream_with_retries(
    stream: AsyncIterator[T],
    item_handler: Callable[[T], None],
    completion_check: Callable[[T], bool],
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> None:
    """mdmd:hidden
    Helper function to consume a stream with retry logic for transient errors.

    Args:
        stream_generator: Function that returns an AsyncIterator to consume
        item_handler: Callback function to handle each item from the stream
        completion_check: Callback function to check if the stream is complete
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
    """
    completed = False
    retries_remaining = max_retries

    while not completed:
        try:
            async for item in stream:
                item_handler(item)
                if completion_check(item):
                    completed = True
                    break

        except (GRPCError, StreamTerminatedError, ClientClosed) as exc:
            if retries_remaining > 0:
                retries_remaining -= 1
                if isinstance(exc, GRPCError):
                    if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                        await asyncio.sleep(retry_delay)
                        continue
                elif isinstance(exc, StreamTerminatedError):
                    continue
                elif isinstance(exc, ClientClosed):
                    break
            raise
