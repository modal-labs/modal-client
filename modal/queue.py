import queue  # The system library
import uuid
from typing import Any, List

from ._async_utils import retry
from .config import logger
from .object import Object, requires_create
from .proto import api_pb2


class Queue(Object):
    """A distributed FIFO Queue.

    The contents of the Queue can be any serializable object.
    """

    def _init(self):
        pass

    async def _create_impl(self, session):
        """This creates a queue on the server and returns its id."""
        request = api_pb2.QueueCreateRequest(session_id=session.session_id)
        response = await session.client.stub.QueueCreate(request)
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
            response = await retry(self._session.client.stub.QueueGet)(request, timeout=60.0)
            if response.values:
                return [self._session.deserialize(value) for value in response.values]
            logger.debug("Queue get for %s had empty results, trying again" % self.object_id)
        raise queue.Empty()

    @requires_create
    async def get(self, block=True, timeout=None):
        """Get and pop the next object"""
        values = await self._get(block, timeout, 1)
        return values[0]

    @requires_create
    async def get_many(self, n_values, block=True, timeout=None):
        """Get up to n_values multiple objects"""
        return await self._get(block, timeout, n_values)

    @requires_create
    async def put_many(self, vs: List[Any]):
        """Put several objects"""
        vs_encoded = [self._session.serialize(v) for v in vs]
        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            values=vs_encoded,
            idempotency_key=str(uuid.uuid4()),
        )
        return await retry(self._session.client.stub.QueuePut)(request, timeout=5.0)

    @requires_create
    async def put(self, v):
        """Put an object"""
        return await self.put_many([v])
