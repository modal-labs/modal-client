# Copyright Modal Labs 2022
import queue  # The system library
import time
import warnings
from datetime import date
from typing import Any, List, Optional

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import retry_transient_errors

from ._resolver import Resolver
from ._serialization import deserialize, serialize
from .exception import deprecation_error
from .object import _Object, live_method


class _Queue(_Object, type_prefix="qu"):
    """Distributed, FIFO queue for data flow in Modal apps.

    The queue can contain any object serializable by `cloudpickle`, including Modal objects.

    **Usage**

    Create a new `Queue` with `Queue.new()`, then assign it to a stub or function.

    ```python
    from modal import Queue, Stub

    stub = Stub()
    stub.my_queue = Queue.new()

    @stub.local_entrypoint()
    def main():
        stub.my_queue.put("some value")
        stub.my_queue.put(123)

        assert stub.my_queue.get() == "some value"
        assert stub.my_queue.get() == 123
    ```

    For more examples, see the [guide](/docs/guide/dicts-and-queues#modal-queues).
    """

    @staticmethod
    def new():
        """Create an empty Queue."""

        async def _load(provider: _Queue, resolver: Resolver, existing_object_id: Optional[str]):
            request = api_pb2.QueueCreateRequest(app_id=resolver.app_id, existing_queue_id=existing_object_id)
            response = await resolver.client.stub.QueueCreate(request)
            provider._hydrate(response.queue_id, resolver.client, None)

        return _Queue._from_loader(_load, "Queue()")

    def __init__(self):
        """mdmd:hidden"""
        deprecation_error(date(2023, 6, 27), "`Queue()` is deprecated. Please use `Queue.new()` instead.")
        obj = _Queue.new()
        self._init_from_other(obj)

    @staticmethod
    def persisted(
        label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None
    ) -> "_Queue":
        """Deploy a Modal app containing this object.

        The deployed object can then be imported from other apps, or by calling
        `Queue.from_name(label)` from that same app.

        **Examples**

        ```python notest
        # In one app:
        stub.queue = Queue.persisted("my-queue")

        # Later, in another app or Python file:
        stub.queue = Queue.from_name("my-queue")
        ```
        """
        return _Queue.new()._persist(label, namespace, environment_name)

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

    @live_method
    async def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """Remove and return the next object in the queue.

        If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
        an object, or until `timeout` if specified. Raises a native `queue.Empty` exception
        if the `timeout` is reached.

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

    @live_method
    async def get_many(self, n_values: int, block: bool = True, timeout: Optional[float] = None) -> List[Any]:
        """Remove and return up to `n_values` objects from the queue.

        If there are fewer than `n_values` items in the queue, return all of them.

        If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
        at least 1 object to be present, or until `timeout` if specified. Raises a native `queue.Empty`
        exception if the `timeout` is reached.

        If `block` is `False`, `get` returns `None` immediately if the queue is empty. The `timeout` is
        ignored in this case.
        """

        if block:
            return await self._get_blocking(timeout, n_values)
        else:
            if timeout is not None:
                warnings.warn("Timeout is ignored for non-blocking get.")
            return await self._get_nonblocking(n_values)

    @live_method
    async def put(self, v: Any) -> None:
        """Add an object to the end of the queue."""
        await self.put_many([v])

    @live_method
    async def put_many(self, vs: List[Any]) -> None:
        """Add several objects to the end of the queue."""
        vs_encoded = [serialize(v) for v in vs]
        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            values=vs_encoded,
        )
        await retry_transient_errors(self._client.stub.QueuePut, request)

    @live_method
    async def len(self) -> int:
        """Return the number of objects in the queue."""
        request = api_pb2.QueueLenRequest(queue_id=self.object_id)
        response = await retry_transient_errors(self._client.stub.QueueLen, request)
        return response.len


Queue = synchronize_api(_Queue)
