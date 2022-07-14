import queue  # The system library
import uuid
from typing import Any, List

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors

from ._serialization import deserialize, serialize
from .config import logger
from .object import Object


class _Queue(Object, type_prefix="qu"):
    """A distributed FIFO Queue.

    The contents of the Queue can be any serializable object.
    """

    def __init__(self):
        super().__init__()

    async def _load(self, client, app_id, existing_queue_id):
        request = api_pb2.QueueCreateRequest(app_id=app_id, existing_queue_id=existing_queue_id)
        response = await client.stub.QueueCreate(request)
        logger.debug("Created queue with id %s" % response.queue_id)
        return response.queue_id

    async def _get(self, block, timeout, n_values):
        while timeout is None or timeout > 0:
            request_timeout = 50.0  # We prevent longer ones in order to keep the connection alive
            if timeout is not None:
                request_timeout = min(request_timeout, timeout)
                timeout -= request_timeout
            request = api_pb2.QueueGetRequest(
                queue_id=self.object_id,
                block=block,
                timeout=request_timeout,
                n_values=n_values,
                idempotency_key=str(uuid.uuid4()),
            )
            response = await retry_transient_errors(self._client.stub.QueueGet, request)
            if response.values:
                return [deserialize(value, self._client) for value in response.values]
            logger.debug("Queue get for %s had empty results, trying again" % self.object_id)
        raise queue.Empty()

    async def get(self, block=True, timeout=None):
        """Get and pop the next object"""
        values = await self._get(block, timeout, 1)
        return values[0]

    async def get_many(self, n_values, block=True, timeout=None):
        """Get up to n_values multiple objects"""
        return await self._get(block, timeout, n_values)

    async def put_many(self, vs: List[Any]):
        """Put several objects"""
        vs_encoded = [serialize(v) for v in vs]
        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            values=vs_encoded,
            idempotency_key=str(uuid.uuid4()),
        )
        return await retry_transient_errors(self._client.stub.QueuePut, request)

    async def put(self, v):
        """Put an object"""
        return await self.put_many([v])


Queue, AioQueue = synchronize_apis(_Queue)
