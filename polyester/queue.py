import queue  # The system library
from typing import Any, List
import uuid

from .async_utils import synchronizer, retry
from .config import logger
from .proto import api_pb2
from .serialization import serializable


@serializable
@synchronizer
class Queue:
    async def create(self):
        ''' This creates a queue on the server and returns its id.'''
        response = await self.client.stub.QueueCreate(api_pb2.Empty())
        logger.debug('Created queue with id %s' % response.queue_id)
        self.remote_id = response.queue_id

    async def _get(self, block, timeout, n_values):
        while timeout is None or timeout > 0:
            request_timeout = 50.0  # We prevent longer ones in order to keep the connection alive
            if timeout is not None:
                request_timeout = min(request_timeout, timeout)
                timeout -= request_timeout
            request = api_pb2.QueueGetRequest(queue_id=self.remote_id, block=block, timeout=request_timeout, n_values=n_values, idempotency_key=str(uuid.uuid4()))
            response = await retry(self.client.stub.QueueGet)(request, timeout=60.0)
            if response.values:
                return [self.client.deserialize(value) for value in response.values]
            logger.debug('Queue get for %s had empty results, trying again' % self.remote_id)
        raise queue.Empty()

    async def get(self, block=True, timeout=None):
        values = await self._get(block, timeout, 1)
        return values[0]

    async def get_many(self, n_values, block=True, timeout=None):
        return await self._get(block, timeout, n_values)

    async def put_many(self, vs: List[Any]):
        vs_encoded = [self.client.serialize(v) for v in vs]
        request = api_pb2.QueuePutRequest(queue_id=self.remote_id, values=vs_encoded, idempotency_key=str(uuid.uuid4()))
        return await retry(self.client.stub.QueuePut)(request, timeout=5.0)

    async def put(self, v):
        return await self.put_many([v])
