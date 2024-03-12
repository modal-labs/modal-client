# Copyright Modal Labs 2022
import queue  # The system library
import time
import warnings
from typing import Any, List, Optional

from grpclib import GRPCError, Status

from modal_proto import api_pb2

from ._resolver import Resolver
from ._serialization import deserialize, serialize
from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import retry_transient_errors
from .client import _Client
from .exception import deprecation_error, deprecation_warning
from .object import _get_environment_name, _Object, live_method


class _Queue(_Object, type_prefix="qu"):
    """Distributed, FIFO queue for data flow in Modal apps.

    The queue can contain any object serializable by `cloudpickle`, including Modal objects.

    **Lifetime of a queue and its contents**

    A `Queue`'s lifetime matches the lifetime of the app it's attached to, but the contents expire after 30 days.
    Because of this, `Queues`s are best used for communication between active functions and not relied on for
    persistent storage. On app completion or after stopping an app any associated `Queue` objects are cleaned up.

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

        async def _load(self: _Queue, resolver: Resolver, existing_object_id: Optional[str]):
            request = api_pb2.QueueCreateRequest(app_id=resolver.app_id, existing_queue_id=existing_object_id)
            response = await resolver.client.stub.QueueCreate(request)
            self._hydrate(response.queue_id, resolver.client, None)

        return _Queue._from_loader(_load, "Queue()")

    def __init__(self):
        """mdmd:hidden"""
        deprecation_error((2023, 6, 27), "`Queue()` is deprecated. Please use `Queue.new()` instead.")
        obj = _Queue.new()
        self._init_from_other(obj)

    @staticmethod
    def from_name(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_Queue":
        """Create a reference to a persisted Queue

        **Examples**

        ```python notest
        # In one app:
        stub.queue = Queue.persisted("my-queue")

        # Later, in another app or Python file:
        stub.queue = Queue.from_name("my-queue")
        ```
        """

        async def _load(self: _Queue, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.QueueGetOrCreateRequest(
                deployment_name=label,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
                object_creation_type=(api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING if create_if_missing else None),
            )
            response = await resolver.client.stub.QueueGetOrCreate(req)
            self._hydrate(response.queue_id, resolver.client, None)

        return _Queue._from_loader(_load, "Queue()", is_another_app=True)

    @staticmethod
    def persisted(
        label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None
    ) -> "_Queue":
        """Deprecated! Use `Queue.from_name(name, create_if_missing=True)`."""
        deprecation_warning((2024, 3, 1), _Queue.persisted.__doc__)
        return _Queue.from_name(label, namespace, environment_name, create_if_missing=True)

    @staticmethod
    async def lookup(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_Queue":
        """Lookup a queue with a given name and tag.

        ```python
        q = modal.Queue.lookup("my-queue")
        q.put(123)
        ```
        """
        obj = _Queue.from_name(
            label, namespace=namespace, environment_name=environment_name, create_if_missing=create_if_missing
        )
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

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
        at least 1 object to be present, or until `timeout` if specified. Raises the stdlib's `queue.Empty`
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
    async def put(self, v: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        """Add an object to the end of the queue.

        If `block` is `True` and the queue is full, this method will retry indefinitely or
        until `timeout` if specified. Raises the stdlib's `queue.Full` exception if the `timeout` is reached.
        If blocking it is not recommended to omit the `timeout`, as the operation could wait indefinitely.

        If `block` is `False`, this method raises `queue.Full` immediately if the queue is full. The `timeout` is
        ignored in this case."""
        await self.put_many([v], block, timeout)

    @live_method
    async def put_many(self, vs: List[Any], block: bool = True, timeout: Optional[float] = None) -> None:
        """Add several objects to the end of the queue.

        If `block` is `True` and the queue is full, this method will retry indefinitely or
        until `timeout` if specified. Raises the stdlib's `queue.Full` exception if the `timeout` is reached.
        If blocking it is not recommended to omit the `timeout`, as the operation could wait indefinitely.

        If `block` is `False`, this method raises `queue.Full` immediately if the queue is full. The `timeout` is
        ignored in this case."""
        if block:
            await self._put_many_blocking(vs, timeout)
        else:
            if timeout is not None:
                warnings.warn("`timeout` argument is ignored for non-blocking put.")
            await self._put_many_nonblocking(vs)

    async def _put_many_blocking(self, vs: List[Any], timeout: Optional[float] = None):
        vs_encoded = [serialize(v) for v in vs]

        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            values=vs_encoded,
        )
        try:
            await retry_transient_errors(
                self._client.stub.QueuePut,
                request,
                # A full queue will return this status.
                additional_status_codes=[Status.RESOURCE_EXHAUSTED],
                max_delay=30.0,
                total_timeout=timeout,
            )
        except GRPCError as exc:
            raise queue.Full(str(exc)) if exc.status == Status.RESOURCE_EXHAUSTED else exc

    async def _put_many_nonblocking(self, vs: List[Any]):
        vs_encoded = [serialize(v) for v in vs]
        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            values=vs_encoded,
        )
        try:
            await retry_transient_errors(self._client.stub.QueuePut, request)
        except GRPCError as exc:
            raise queue.Full(exc.message) if exc.status == Status.RESOURCE_EXHAUSTED else exc

    @live_method
    async def len(self) -> int:
        """Return the number of objects in the queue."""
        request = api_pb2.QueueLenRequest(queue_id=self.object_id)
        response = await retry_transient_errors(self._client.stub.QueueLen, request)
        return response.len


Queue = synchronize_api(_Queue)
