# Copyright Modal Labs 2022
import queue  # The system library
import time
import warnings
from typing import Any, AsyncGenerator, AsyncIterator, List, Optional, Type

from grpclib import GRPCError, Status
from synchronicity.async_wrap import asynccontextmanager

from modal_proto import api_pb2

from ._resolver import Resolver
from ._serialization import deserialize, serialize
from ._utils.async_utils import TaskContext, synchronize_api, warn_if_generator_is_not_consumed
from ._utils.grpc_utils import retry_transient_errors
from ._utils.name_utils import check_object_name
from .client import _Client
from .exception import InvalidError, RequestSizeError, deprecation_error
from .object import EPHEMERAL_OBJECT_HEARTBEAT_SLEEP, _get_environment_name, _Object, live_method, live_method_gen


class _Queue(_Object, type_prefix="qu"):
    """Distributed, FIFO queue for data flow in Modal apps.

    The queue can contain any object serializable by `cloudpickle`, including Modal objects.

    By default, the `Queue` object acts as a single FIFO queue which supports puts and gets (blocking and non-blocking).

    **Queue partitions (beta)**

    Specifying partition keys gives access to other independent FIFO partitions within the same `Queue` object.
    Across any two partitions, puts and gets are completely independent.
    For example, a put in one partition does not affect a get in any other partition.

    When no partition key is specified (by default), puts and gets will operate on a default partition.
    This default partition is also isolated from all other partitions.
    Please see the Usage section below for an example using partitions.

    **Lifetime of a queue and its partitions**

    By default, each partition is cleared 24 hours after the last `put` operation.
    A lower TTL can be specified by the `partition_ttl` argument in the `put` or `put_many` methods.
    Each partition's expiry is handled independently.

    As such, `Queue`s are best used for communication between active functions and not relied on for persistent storage.

    On app completion or after stopping an app any associated `Queue` objects are cleaned up.
    All its partitions will be cleared.

    **Limits**

    A single `Queue` can contain up to 100,000 partitions, each with up to 5,000 items. Each item can be up to 256 KiB.

    Partition keys must be non-empty and must not exceed 64 bytes.

    **Usage**

    ```python
    from modal import Queue

    with Queue.ephemeral() as my_queue:
        # Putting values
        my_queue.put("some value")
        my_queue.put(123)

        # Getting values
        assert my_queue.get() == "some value"
        assert my_queue.get() == 123

        # Using partitions
        my_queue.put(0)
        my_queue.put(1, partition="foo")
        my_queue.put(2, partition="bar")

        # Default and "foo" partition are ignored by the get operation.
        assert my_queue.get(partition="bar") == 2

        # Set custom 10s expiration time on "foo" partition.
        my_queue.put(3, partition="foo", partition_ttl=10)

        # (beta feature) Iterate through items in place (read immutably)
        my_queue.put(1)
        assert [v for v in my_queue.iterate()] == [0, 1]
    ```

    For more examples, see the [guide](/docs/guide/dicts-and-queues#modal-queues).
    """

    @staticmethod
    def new():
        """`Queue.new` is deprecated.

        Please use `Queue.from_name` (for persisted) or `Queue.ephemeral` (for ephemeral) queues.
        """
        deprecation_error((2024, 3, 19), Queue.new.__doc__)

    def __init__(self):
        """mdmd:hidden"""
        raise RuntimeError("Queue() is not allowed. Please use `Queue.from_name(...)` or `Queue.ephemeral()` instead.")

    @staticmethod
    def validate_partition_key(partition: Optional[str]) -> bytes:
        if partition is not None:
            partition_key = partition.encode("utf-8")
            if len(partition_key) == 0 or len(partition_key) > 64:
                raise InvalidError("Queue partition key must be between 1 and 64 characters.")
        else:
            partition_key = b""

        return partition_key

    @classmethod
    @asynccontextmanager
    async def ephemeral(
        cls: Type["_Queue"],
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        _heartbeat_sleep: float = EPHEMERAL_OBJECT_HEARTBEAT_SLEEP,
    ) -> AsyncIterator["_Queue"]:
        """Creates a new ephemeral queue within a context manager:

        Usage:
        ```python
        from modal import Queue

        with Queue.ephemeral() as q:
            q.put(123)

        async with Queue.ephemeral() as q:
            await q.put.aio(123)
        ```
        """
        if client is None:
            client = await _Client.from_env()
        request = api_pb2.QueueGetOrCreateRequest(
            object_creation_type=api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL,
            environment_name=_get_environment_name(environment_name),
        )
        response = await client.stub.QueueGetOrCreate(request)
        async with TaskContext() as tc:
            request = api_pb2.QueueHeartbeatRequest(queue_id=response.queue_id)
            tc.infinite_loop(lambda: client.stub.QueueHeartbeat(request), sleep=_heartbeat_sleep)
            yield cls._new_hydrated(response.queue_id, client, None, is_another_app=True)

    @staticmethod
    def from_name(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_Queue":
        """Create a reference to a persisted Queue

        **Examples**

        ```python
        from modal import Queue

        queue = Queue.from_name("my-queue", create_if_missing=True)
        queue.put(123)
        ```
        """
        check_object_name(label, "Queue")

        async def _load(self: _Queue, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.QueueGetOrCreateRequest(
                deployment_name=label,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
                object_creation_type=(api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING if create_if_missing else None),
            )
            response = await resolver.client.stub.QueueGetOrCreate(req)
            self._hydrate(response.queue_id, resolver.client, None)

        return _Queue._from_loader(_load, "Queue()", is_another_app=True, hydrate_lazily=True)

    @staticmethod
    def persisted(label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None):
        """Deprecated! Use `Queue.from_name(name, create_if_missing=True)`."""
        deprecation_error((2024, 3, 1), _Queue.persisted.__doc__)

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
        from modal import Queue

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

    @staticmethod
    async def delete(label: str, *, client: Optional[_Client] = None, environment_name: Optional[str] = None):
        obj = await _Queue.lookup(label, client=client, environment_name=environment_name)
        req = api_pb2.QueueDeleteRequest(queue_id=obj.object_id)
        await retry_transient_errors(obj._client.stub.QueueDelete, req)

    async def _get_nonblocking(self, partition: Optional[str], n_values: int) -> List[Any]:
        request = api_pb2.QueueGetRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            timeout=0,
            n_values=n_values,
        )

        response = await retry_transient_errors(self._client.stub.QueueGet, request)
        if response.values:
            return [deserialize(value, self._client) for value in response.values]
        else:
            return []

    async def _get_blocking(self, partition: Optional[str], timeout: Optional[float], n_values: int) -> List[Any]:
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
                partition_key=self.validate_partition_key(partition),
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
    async def clear(self, *, partition: Optional[str] = None, all: bool = False) -> None:
        """Clear the contents of a single partition or all partitions."""
        if partition and all:
            raise InvalidError("Partition must be null when requesting to clear all.")
        request = api_pb2.QueueClearRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            all_partitions=all,
        )
        await retry_transient_errors(self._client.stub.QueueClear, request)

    @live_method
    async def get(
        self, block: bool = True, timeout: Optional[float] = None, *, partition: Optional[str] = None
    ) -> Optional[Any]:
        """Remove and return the next object in the queue.

        If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
        an object, or until `timeout` if specified. Raises a native `queue.Empty` exception
        if the `timeout` is reached.

        If `block` is `False`, `get` returns `None` immediately if the queue is empty. The `timeout` is
        ignored in this case.
        """

        if block:
            values = await self._get_blocking(partition, timeout, 1)
        else:
            if timeout is not None:
                warnings.warn("Timeout is ignored for non-blocking get.")
            values = await self._get_nonblocking(partition, 1)

        if values:
            return values[0]
        else:
            return None

    @live_method
    async def get_many(
        self, n_values: int, block: bool = True, timeout: Optional[float] = None, *, partition: Optional[str] = None
    ) -> List[Any]:
        """Remove and return up to `n_values` objects from the queue.

        If there are fewer than `n_values` items in the queue, return all of them.

        If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
        at least 1 object to be present, or until `timeout` if specified. Raises the stdlib's `queue.Empty`
        exception if the `timeout` is reached.

        If `block` is `False`, `get` returns `None` immediately if the queue is empty. The `timeout` is
        ignored in this case.
        """

        if block:
            return await self._get_blocking(partition, timeout, n_values)
        else:
            if timeout is not None:
                warnings.warn("Timeout is ignored for non-blocking get.")
            return await self._get_nonblocking(partition, n_values)

    @live_method
    async def put(
        self,
        v: Any,
        block: bool = True,
        timeout: Optional[float] = None,
        *,
        partition: Optional[str] = None,
        partition_ttl: int = 24 * 3600,  # After 24 hours of no activity, this partition will be deletd.
    ) -> None:
        """Add an object to the end of the queue.

        If `block` is `True` and the queue is full, this method will retry indefinitely or
        until `timeout` if specified. Raises the stdlib's `queue.Full` exception if the `timeout` is reached.
        If blocking it is not recommended to omit the `timeout`, as the operation could wait indefinitely.

        If `block` is `False`, this method raises `queue.Full` immediately if the queue is full. The `timeout` is
        ignored in this case."""
        await self.put_many([v], block, timeout, partition=partition, partition_ttl=partition_ttl)

    @live_method
    async def put_many(
        self,
        vs: List[Any],
        block: bool = True,
        timeout: Optional[float] = None,
        *,
        partition: Optional[str] = None,
        partition_ttl: int = 24 * 3600,  # After 24 hours of no activity, this partition will be deletd.
    ) -> None:
        """Add several objects to the end of the queue.

        If `block` is `True` and the queue is full, this method will retry indefinitely or
        until `timeout` if specified. Raises the stdlib's `queue.Full` exception if the `timeout` is reached.
        If blocking it is not recommended to omit the `timeout`, as the operation could wait indefinitely.

        If `block` is `False`, this method raises `queue.Full` immediately if the queue is full. The `timeout` is
        ignored in this case.
        """
        if block:
            await self._put_many_blocking(partition, partition_ttl, vs, timeout)
        else:
            if timeout is not None:
                warnings.warn("`timeout` argument is ignored for non-blocking put.")
            await self._put_many_nonblocking(partition, partition_ttl, vs)

    async def _put_many_blocking(
        self, partition: Optional[str], partition_ttl: int, vs: List[Any], timeout: Optional[float] = None
    ):
        vs_encoded = [serialize(v) for v in vs]

        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            values=vs_encoded,
            partition_ttl_seconds=partition_ttl,
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
            if exc.status == Status.RESOURCE_EXHAUSTED:
                raise queue.Full(str(exc))
            elif "status = '413'" in exc.message:
                method = "put_many" if len(vs) > 1 else "put"
                raise RequestSizeError(f"Queue.{method} request is too large") from exc
            else:
                raise exc

    async def _put_many_nonblocking(self, partition: Optional[str], partition_ttl: int, vs: List[Any]):
        vs_encoded = [serialize(v) for v in vs]
        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            values=vs_encoded,
            partition_ttl_seconds=partition_ttl,
        )
        try:
            await retry_transient_errors(self._client.stub.QueuePut, request)
        except GRPCError as exc:
            if exc.status == Status.RESOURCE_EXHAUSTED:
                raise queue.Full(exc.message)
            elif "status = '413'" in exc.message:
                method = "put_many" if len(vs) > 1 else "put"
                raise RequestSizeError(f"Queue.{method} request is too large") from exc
            else:
                raise exc

    @live_method
    async def len(self, *, partition: Optional[str] = None, total: bool = False) -> int:
        """Return the number of objects in the queue partition."""
        if partition and total:
            raise InvalidError("Partition must be null when requesting total length.")
        request = api_pb2.QueueLenRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            total=total,
        )
        response = await retry_transient_errors(self._client.stub.QueueLen, request)
        return response.len

    @warn_if_generator_is_not_consumed()
    @live_method_gen
    async def iterate(
        self, *, partition: Optional[str] = None, item_poll_timeout: float = 0.0
    ) -> AsyncGenerator[Any, None]:
        """(Beta feature) Iterate through items in the queue without mutation.

        Specify `item_poll_timeout` to control how long the iterator should wait for the next time before giving up.
        """
        last_entry_id: Optional[str] = None
        validated_partition_key = self.validate_partition_key(partition)
        fetch_deadline = time.time() + item_poll_timeout

        MAX_POLL_DURATION = 30.0
        while True:
            poll_duration = max(0.0, min(MAX_POLL_DURATION, fetch_deadline - time.time()))
            request = api_pb2.QueueNextItemsRequest(
                queue_id=self.object_id,
                partition_key=validated_partition_key,
                last_entry_id=last_entry_id,
                item_poll_timeout=poll_duration,
            )

            response: api_pb2.QueueNextItemsResponse = await retry_transient_errors(
                self._client.stub.QueueNextItems, request
            )
            if response.items:
                for item in response.items:
                    yield deserialize(item.value, self._client)
                    last_entry_id = item.entry_id
                fetch_deadline = time.time() + item_poll_timeout
            elif time.time() > fetch_deadline:
                break


Queue = synchronize_api(_Queue)
