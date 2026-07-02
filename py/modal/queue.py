# Copyright Modal Labs 2022
import builtins
import queue  # The system library
import time
import warnings
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime
from typing import Any

from google.protobuf.message import Message
from grpclib import Status
from synchronicity import classproperty
from synchronicity.async_wrap import asynccontextmanager

from modal_proto import api_pb2

from ._load_context import LoadContext
from ._object import (
    EPHEMERAL_OBJECT_HEARTBEAT_SLEEP,
    _get_environment_name,
    _Object,
    live_method,
    live_method_gen,
)
from ._resolver import Resolver
from ._serialization import deserialize, serialize
from ._utils.async_utils import TaskContext, synchronize_api, warn_if_generator_is_not_consumed
from ._utils.grpc_utils import Retry
from ._utils.name_utils import check_object_name
from ._utils.time_utils import as_timestamp, timestamp_to_localized_dt
from .client import _Client
from .exception import AlreadyExistsError, Error, InvalidError, NotFoundError, RequestSizeError, ResourceExhaustedError
from .types import QueueInfo


class _QueueManager:
    """Namespace with methods for managing named Queue objects."""

    async def create(
        self,
        name: str,
        *,
        allow_existing: bool = False,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> None:
        """Create a new named Queue in the workspace environment.

        This does not return a local handle; use `modal.Queue.from_name` to look up the Queue after creation.

        Added in v1.1.2.

        Args:
            name: Name for the new Queue.
            allow_existing: If True, do nothing when a Queue with this name already exists.
            environment_name: Environment to create in; defaults to the active environment.
            client: Modal client to use; defaults to `Client.from_env()` when omitted.

        Examples:
            ```python notest
            modal.Queue.objects.create("my-queue")
            ```

            Queues will be created in the active environment, or another one can be specified:

            ```python notest
            modal.Queue.objects.create("my-queue", environment_name="dev")
            ```

            By default, an error is raised if the Queue already exists; `allow_existing=True` makes that case a no-op:

            ```python notest
            modal.Queue.objects.create("my-queue", allow_existing=True)
            ```

            Note that this method does not return a local instance of the Queue. You can use
            `modal.Queue.from_name` to perform a lookup after creation.
        """
        check_object_name(name, "Queue")
        client = await _Client.from_env() if client is None else client
        object_creation_type = (
            api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING
            if allow_existing
            else api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS
        )
        req = api_pb2.QueueGetOrCreateRequest(
            deployment_name=name,
            environment_name=_get_environment_name(environment_name),
            object_creation_type=object_creation_type,
        )
        try:
            await client.stub.QueueGetOrCreate(req)
        except AlreadyExistsError:
            if not allow_existing:
                raise

    async def list(
        self,
        *,
        max_objects: int | None = None,
        created_before: datetime | str | None = None,
        environment_name: str = "",
        client: _Client | None = None,
    ) -> builtins.list["_Queue"]:
        """List named Queues in the workspace environment as hydrated handles.

        Results are ordered newest to oldest. By default, all matching Queues are returned.

        Added in v1.1.2.

        Args:
            max_objects: Maximum number of Queues to return.
            created_before: Only include Queues created before this time (datetime or ISO date string).
            environment_name: Environment to list from; defaults to the active environment.
            client: Modal client to use; defaults to `Client.from_env()` when omitted.

        Returns:
            Hydrated `Queue` objects for each named Queue in the listing.

        Examples:
            ```python
            queues = modal.Queue.objects.list()
            print([q.name for q in queues])
            ```

            Queues will be retrieved from the active environment, or another one can be specified:

            ```python notest
            dev_queues = modal.Queue.objects.list(environment_name="dev")
            ```

            By default, all named Queues are returned, newest to oldest. It's also possible to limit the
            number of results and to filter by creation date:

            ```python
            queues = modal.Queue.objects.list(max_objects=10, created_before="2025-01-01")
            ```
        """
        client = await _Client.from_env() if client is None else client
        if max_objects is not None and max_objects < 0:
            raise InvalidError("max_objects cannot be negative")

        items: list[api_pb2.QueueListResponse.QueueInfo] = []

        async def retrieve_page(created_before: float) -> bool:
            max_page_size = 100 if max_objects is None else min(100, max_objects - len(items))
            pagination = api_pb2.ListPagination(max_objects=max_page_size, created_before=created_before)
            req = api_pb2.QueueListRequest(
                environment_name=_get_environment_name(environment_name), pagination=pagination
            )
            resp = await client.stub.QueueList(req)
            items.extend(resp.queues)
            finished = (len(resp.queues) < max_page_size) or (max_objects is not None and len(items) >= max_objects)
            return finished

        finished = await retrieve_page(as_timestamp(created_before))
        while True:
            if finished:
                break
            finished = await retrieve_page(items[-1].metadata.creation_info.created_at)

        queues = [
            _Queue._new_hydrated(
                item.queue_id,
                client,
                item.metadata,
                skip_reload=True,
                rep=_Queue._repr(item.name, environment_name),
            )
            for item in items
        ]
        return queues[:max_objects] if max_objects is not None else queues

    async def delete(
        self,
        name: str,
        *,
        allow_missing: bool = False,
        environment_name: str | None = None,
        client: _Client | None = None,
    ):
        """Delete a named Queue entirely (not a single message or partition).

        Deletion is irreversible and affects any Apps using this Queue.

        Added in v1.1.2.

        Args:
            name: Name of the Queue to delete.
            allow_missing: If True, do nothing when the Queue does not exist.
            environment_name: Environment to delete from; defaults to the active environment.
            client: Modal client to use; defaults to `Client.from_env()` when omitted.

        Examples:
            ```python notest
            await modal.Queue.objects.delete("my-queue")
            ```

            Queues will be deleted from the active environment, or another one can be specified:

            ```python notest
            await modal.Queue.objects.delete("my-queue", environment_name="dev")
            ```
        """
        try:
            obj = await _Queue.from_name(name, environment_name=environment_name).hydrate(client)
        except NotFoundError:
            if not allow_missing:
                raise
        else:
            req = api_pb2.QueueDeleteRequest(queue_id=obj.object_id)
            await obj._client.stub.QueueDelete(req)


QueueManager = synchronize_api(_QueueManager)


class _Queue(_Object, type_prefix="qu"):
    """Distributed, FIFO queue for data flow in Modal apps.

    The queue can contain any object serializable by `cloudpickle`, including Modal objects.

    By default, the `Queue` object acts as a single FIFO queue which supports puts and gets (blocking and non-blocking).

    Examples:
        ```python
        from modal import Queue

        # Create an ephemeral queue which is anonymous and garbage collected
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

            # Iterate through items in place (read immutably)
            my_queue.put(1)
            assert [v for v in my_queue.iterate()] == [0, 1]

        # You can also create persistent queues that can be used across apps
        queue = Queue.from_name("my-persisted-queue", create_if_missing=True)
        queue.put(42)
        assert queue.get() == 42
        ```

        For more examples, see the [guide](https://modal.com/docs/guide/dicts-and-queues#modal-queues).

        **Queue partitions**

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

        As such, `Queue`s are best used for communication between active functions and not relied on for persistent
        storage.

        On app completion or after stopping an app any associated `Queue` objects are cleaned up.
        All its partitions will be cleared.

        **Limits**

        A single `Queue` can contain up to 100,000 partitions, each with up to 5,000 items. Each item can be up to
        1 MiB.

        Partition keys must be non-empty and must not exceed 64 bytes.
    """

    _metadata: api_pb2.QueueMetadata | None = None

    def __init__(self):
        """mdmd:hidden"""
        raise RuntimeError("Queue() is not allowed. Please use `Queue.from_name(...)` or `Queue.ephemeral()` instead.")

    @classproperty
    @classmethod
    def objects(cls) -> _QueueManager:
        return _QueueManager()

    @property
    def name(self) -> str | None:
        return self._name

    def _hydrate_metadata(self, metadata: Message | None):
        if metadata:
            assert isinstance(metadata, api_pb2.QueueMetadata)
            self._metadata = metadata
            self._name = metadata.name

    def _get_metadata(self) -> api_pb2.QueueMetadata:
        assert self._metadata
        return self._metadata

    @staticmethod
    def validate_partition_key(partition: str | None) -> bytes:
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
        cls: type["_Queue"],
        client: _Client | None = None,
        environment_name: str | None = None,
        _heartbeat_sleep: float = EPHEMERAL_OBJECT_HEARTBEAT_SLEEP,  # mdmd:line-hidden
    ) -> AsyncIterator["_Queue"]:
        """Create an anonymous Queue that exists for the duration of the context manager.

        Args:
            client: Modal client to use; defaults to `Client.from_env()` when omitted.
            environment_name: Environment for the ephemeral Queue; defaults to the active environment.

        Examples:
            ```python
            from modal import Queue

            with Queue.ephemeral() as q:
                q.put(123)
            ```

            ```python notest
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
            yield cls._new_hydrated(response.queue_id, client, response.metadata, skip_reload=True)

    @staticmethod
    def from_name(
        name: str,
        *,
        environment_name: str | None = None,
        create_if_missing: bool = False,
        client: _Client | None = None,
    ) -> "_Queue":
        """Reference a named Queue, optionally creating it on the server first.

        Hydration is lazy: metadata is fetched from Modal the first time the handle is used.

        Args:
            name: Deployment name of the Queue.
            environment_name: Environment to resolve the name in; defaults to the active environment.
            create_if_missing: If True, create the Queue when it does not already exist.
            client: Modal client to use for loading; defaults to `Client.from_env()` when omitted.

        Returns:
            A `Queue` handle (possibly not yet hydrated).

        Examples:
            ```python
            q = modal.Queue.from_name("my-queue", create_if_missing=True)
            q.put(123)
            ```
        """
        check_object_name(name, "Queue")

        async def _load(self: _Queue, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            req = api_pb2.QueueGetOrCreateRequest(
                deployment_name=name,
                environment_name=load_context.environment_name,
                object_creation_type=(api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING if create_if_missing else None),
            )
            response = await load_context.client.stub.QueueGetOrCreate(req)
            self._hydrate(response.queue_id, load_context.client, response.metadata)

        rep = _Queue._repr(name, environment_name)
        return _Queue._from_loader(
            _load,
            rep,
            skip_reload=True,
            hydrate_lazily=True,
            name=name,
            load_context_overrides=LoadContext(environment_name=environment_name, client=client),
        )

    @staticmethod
    def from_id(
        queue_id: str,
        client: _Client | None = None,
    ) -> "_Queue":
        """Construct a Queue from an id and look up the Queue metadata.

        This is a lazy method that defers hydrating the local
        object with metadata from Modal servers until the first
        time it is actually used.

        The ID of a Queue object can be accessed using `.object_id`.

        Args:
            queue_id: Queue object ID to attach to.
            client: Modal client to use for loading; defaults to `Client.from_env()` when omitted.

        Returns:
            A `Queue` handle (possibly not yet hydrated).

        Examples:
            ```python notest
            @app.function()
            def my_consumer(queue_id: str):
                queue = modal.Queue.from_id(queue_id)
                queue.put("Hello from remote function!")

            with modal.Queue.ephemeral() as q:
                my_consumer.remote(q.object_id)
                print(q.get())  # "Hello from remote function!"
            ```
        """

        async def _load(self: _Queue, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            req = api_pb2.QueueGetByIdRequest(queue_id=queue_id)
            response = await load_context.client.stub.QueueGetById(req)
            self._hydrate(response.queue_id, load_context.client, response.metadata)

        rep = f"Queue.from_id({queue_id!r})"
        return _Queue._from_loader(
            _load,
            rep,
            skip_reload=True,
            hydrate_lazily=True,
            load_context_overrides=LoadContext(client=client),
        )

    @live_method
    async def info(self) -> QueueInfo:
        """Return information about the Queue object."""
        metadata = self._get_metadata()
        creation_info = metadata.creation_info
        return QueueInfo(
            name=metadata.name or None,
            created_at=timestamp_to_localized_dt(creation_info.created_at),
            created_by=creation_info.created_by or None,
        )

    async def _get_nonblocking(self, partition: str | None, n_values: int) -> list[Any]:
        request = api_pb2.QueueGetRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            timeout=0,
            n_values=n_values,
        )

        response = await self._client.stub.QueueGet(request)
        if response.values:
            return [deserialize(value, self._client) for value in response.values]
        else:
            return []

    async def _get_blocking(self, partition: str | None, timeout: float | None, n_values: int) -> list[Any]:
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

            response = await self._client.stub.QueueGet(request)

            if response.values:
                return [deserialize(value, self._client) for value in response.values]

            if deadline is not None and time.time() > deadline:
                break

        raise queue.Empty()

    @live_method
    async def clear(self, *, partition: str | None = None, all: bool = False) -> None:
        """Clear the contents of a single partition or all partitions.

        Warning: this is a destructive operation and will irrevocably delete data.

        Args:
            partition: Partition to clear; omit with `all=True` to clear every partition.
            all: If True, clear all partitions (`partition` must not be set).

        Examples:
            ```python
            q = modal.Queue.from_name("my-queue", create_if_missing=True)
            q.clear()
            ```
        """
        if partition and all:
            raise InvalidError("Partition must be null when requesting to clear all.")
        request = api_pb2.QueueClearRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            all_partitions=all,
        )
        await self._client.stub.QueueClear(request)

    @live_method
    async def get(
        self, block: bool = True, timeout: float | None = None, *, partition: str | None = None
    ) -> Any | None:
        """Remove and return the next object in the queue.

        If `block` is `True` (the default) and the queue is empty, `get` will wait indefinitely for
        an object, or until `timeout` if specified. Raises a native `queue.Empty` exception
        if the `timeout` is reached.

        If `block` is `False`, `get` returns `None` immediately if the queue is empty. The `timeout` is
        ignored in this case.

        Args:
            block: If True, wait for an item; if False, return ``None`` immediately when empty.
            timeout: Seconds to wait when blocking; ignored when ``block`` is False.
            partition: FIFO partition to read from; uses the default partition when omitted.
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
        self, n_values: int, block: bool = True, timeout: float | None = None, *, partition: str | None = None
    ) -> list[Any]:
        """Remove and return up to `n_values` objects from the queue.

        If there are fewer than `n_values` items in the queue, return all of them.

        If `block` is `True` (the default) and the queue is empty, `get_many` waits until at least one
        object is present, or until `timeout` if specified. Raises the stdlib's `queue.Empty` if the
        timeout is reached before any item arrives.

        If `block` is `False`, this returns an empty list immediately when the queue is empty. The `timeout`
        is ignored in that case.

        Args:
            n_values: Maximum number of items to remove and return.
            block: If True, wait until at least one item is available (or until `timeout`); if False, return
                immediately when empty.
            timeout: Seconds to wait when blocking; ignored when ``block`` is False.
            partition: FIFO partition to read from; uses the default partition when omitted.
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
        timeout: float | None = None,
        *,
        partition: str | None = None,
        partition_ttl: int = 24 * 3600,
    ) -> None:
        """Add an object to the end of the queue.

        If `block` is `True` and the queue is full, this method will retry indefinitely or
        until `timeout` if specified. Raises the stdlib's `queue.Full` exception if the `timeout` is reached.
        If blocking it is not recommended to omit the `timeout`, as the operation could wait indefinitely.

        If `block` is `False`, this method raises `queue.Full` immediately if the queue is full. The `timeout` is
        ignored in this case.

        Args:
            v: Value to enqueue (must be serializable).
            block: If True, wait for capacity; if False, fail immediately when full.
            timeout: Max seconds to wait when blocking.
            partition: FIFO partition to write to; uses the default partition when omitted.
            partition_ttl: Seconds after the last activity before this partition may be cleared (default 24 hours).
        """
        await self.put_many([v], block, timeout, partition=partition, partition_ttl=partition_ttl)

    @live_method
    async def put_many(
        self,
        vs: list[Any],
        block: bool = True,
        timeout: float | None = None,
        *,
        partition: str | None = None,
        partition_ttl: int = 24 * 3600,
    ) -> None:
        """Add several objects to the end of the queue.

        If `block` is `True` and the queue is full, this method will retry indefinitely or
        until `timeout` if specified. Raises the stdlib's `queue.Full` exception if the `timeout` is reached.
        If blocking it is not recommended to omit the `timeout`, as the operation could wait indefinitely.

        If `block` is `False`, this method raises `queue.Full` immediately if the queue is full. The `timeout` is
        ignored in this case.

        Args:
            vs: Values to enqueue (each must be serializable).
            block: If True, wait for capacity; if False, fail immediately when full.
            timeout: Max seconds to wait when blocking.
            partition: FIFO partition to write to; uses the default partition when omitted.
            partition_ttl: Seconds after the last activity before this partition may be cleared (default 24 hours).
        """
        if block:
            await self._put_many_blocking(partition, partition_ttl, vs, timeout)
        else:
            if timeout is not None:
                warnings.warn("`timeout` argument is ignored for non-blocking put.")
            await self._put_many_nonblocking(partition, partition_ttl, vs)

    async def _put_many_blocking(
        self, partition: str | None, partition_ttl: int, vs: list[Any], timeout: float | None = None
    ):
        vs_encoded = [serialize(v) for v in vs]

        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            values=vs_encoded,
            partition_ttl_seconds=partition_ttl,
        )
        try:
            await self._client.stub.QueuePut(
                request,
                # A full queue will return this status.
                retry=Retry(
                    additional_status_codes=[Status.RESOURCE_EXHAUSTED],
                    max_delay=30.0,
                    max_retries=None,
                    total_timeout=timeout,
                ),
            )
        except Error as exc:
            if "status = '413'" in str(exc):
                method = "put_many" if len(vs) > 1 else "put"
                raise RequestSizeError(f"Queue.{method} request is too large") from exc
            elif isinstance(exc, ResourceExhaustedError):
                raise queue.Full(str(exc))
            else:
                raise exc

    async def _put_many_nonblocking(self, partition: str | None, partition_ttl: int, vs: list[Any]):
        vs_encoded = [serialize(v) for v in vs]
        request = api_pb2.QueuePutRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            values=vs_encoded,
            partition_ttl_seconds=partition_ttl,
        )
        try:
            await self._client.stub.QueuePut(request)
        except Error as exc:
            if "status = '413'" in str(exc):
                method = "put_many" if len(vs) > 1 else "put"
                raise RequestSizeError(f"Queue.{method} request is too large") from exc
            elif isinstance(exc, ResourceExhaustedError):
                raise queue.Full(str(exc))
            else:
                raise exc

    @live_method
    async def len(self, *, partition: str | None = None, total: bool = False) -> int:
        """Return the number of objects in the queue partition.

        Args:
            partition: Partition to measure; omit for the default partition.
            total: If True, return the combined length of all partitions (do not pass `partition`).

        Returns:
            Item count (capped by the server when very large).
        """
        if partition and total:
            raise InvalidError("Partition must be null when requesting total length.")
        request = api_pb2.QueueLenRequest(
            queue_id=self.object_id,
            partition_key=self.validate_partition_key(partition),
            total=total,
        )
        response = await self._client.stub.QueueLen(request)
        return response.len

    @warn_if_generator_is_not_consumed()
    @live_method_gen
    async def iterate(
        self, *, partition: str | None = None, item_poll_timeout: float = 0.0
    ) -> AsyncGenerator[Any, None]:
        """Iterate through items in the queue without mutation.

        Specify `item_poll_timeout` to control how long the iterator should wait for the next time before giving up.

        Args:
            partition: Partition to scan; uses the default partition when omitted.
            item_poll_timeout: How long to wait for another item before stopping the iterator.
        """
        last_entry_id: str | None = None
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

            response: api_pb2.QueueNextItemsResponse = await self._client.stub.QueueNextItems(request)
            if response.items:
                for item in response.items:
                    yield deserialize(item.value, self._client)
                    last_entry_id = item.entry_id
                fetch_deadline = time.time() + item_poll_timeout
            elif time.time() > fetch_deadline:
                break


Queue = synchronize_api(_Queue)
