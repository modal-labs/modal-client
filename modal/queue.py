# Copyright Modal Labs 2022
import queue  # The system library
import time
import warnings
from typing import Any, List, Optional

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors

from ._resolver import Resolver
from ._serialization import deserialize, serialize
from .object import Handle, Provider


class _QueueHandle(Handle, type_prefix="qu"):
    """Handle for interacting with the contents of a `Queue`

    ```python
    stub.some_dict = modal.Queue()

    if __name__ == "__main__":
        with stub.run() as app:
            app.some_dict.put({"some": "object"})
    ```
    """

    async def _get_nonblocking(self, n_values: int) -> List[Any]:
        request = api_pb2.QueueGetRequest(
            queue_id=self.object_id,
            timeout=0,
            n_values=n_values,
        )

        response = await retry_transient_errors(self._client.stub.QueueGet, request)
        if response.values:
            return [deserialize(value, self._client) for value in response.values]
        else:
            return []

    async def _get_blocking(self, timeout: Optional[float], n_values: int) -> List[Any]:
        if timeout is not None:
            deadline = time.time() + timeout
        else:
            deadline = None

        while True:
            request_timeout = 50.0  # We prevent longer ones in order to keep the connection alive

            if deadline is not None:
                request_timeout = min(request_timeout, deadline - time.time())

            request = api_pb2.QueueGetRequest(
                queue_id=self.object_id,
                timeout=request_timeout,
                n_values=n_values,
            )

            response = await retry_transient_errors(self._client.stub.QueueGet, request)

            if response.values:
                return [deserialize(value, self._client) for value in response.values]

            if deadline is not None and time.time() > deadline:
                break

        raise queue.Empty()

    async def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """Remove and return the next object in the queue.

        If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
        an object, or until `timeout` if a `timeout` is specified. Raises the native Python `queue.Empty`
        exception if the `timeout` is reached.

        If `block` is `False`, `get` returns `None` immediately if the queue is empty. The `timeout` is
        ignored in this case.
        """

        if block:
            values = await self._get_blocking(timeout, 1)
        else:
            if timeout is not None:
                warnings.warn("Timeout is ignored for non-blocking get.")
            values = await self._get_nonblocking(1)

        if values:
            return values[0]
        else:
            return None

    async def get_many(self, n_values: int, block: bool = True, timeout: Optional[float] = None) -> List[Any]:
        """Remove and return up to `n_values` objects from the queue.

        If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
        the next object, or until `timeout` if a `timeout` is specified. Raises the native Python `queue.Empty`
        exception if the `timeout` is reached. Returns as many objects as are available (less then `n_values`)
        as soon as the queue becomes non-empty.

        If `block` is `False`, `get` returns `None` immediately if the queue is empty. The `timeout` is
        ignored in this case.
        """

        if block:
            return await self._get_blocking(timeout, n_values)
        else:
            if timeout is not None:
                warnings.warn("Timeout is ignored for non-blocking get.")
            return await self._get_nonblocking(n_values)

    async def put(self, v: Any) -> None:
        """Add an object to the end of the queue."""

        await self.put_many([v])

    async def put_many(self, vs: List[Any]) -> None:
        """Add several objects to the end of the queue."""

        vs_encoded = [serialize(v) for v in vs]

        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            values=vs_encoded,
        )

        await retry_transient_errors(self._client.stub.QueuePut, request)


QueueHandle, AioQueueHandle = synchronize_apis(_QueueHandle)


class _Queue(Provider[_QueueHandle]):
    """A distributed, FIFO Queue available to Modal apps.

    The queue can contain any object serializable by `cloudpickle`.
    """

    def __init__(self):
        async def _load(resolver: Resolver) -> _QueueHandle:
            request = api_pb2.QueueCreateRequest(app_id=resolver.app_id, existing_queue_id=resolver.existing_object_id)
            response = await resolver.client.stub.QueueCreate(request)
            return _QueueHandle._from_id(response.queue_id, resolver.client, None)

        super().__init__(_load, "Queue()")


Queue, AioQueue = synchronize_apis(_Queue)
