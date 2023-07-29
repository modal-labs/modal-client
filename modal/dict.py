# Copyright Modal Labs 2022
from datetime import date
from typing import Any, Optional

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import retry_transient_errors

from ._resolver import Resolver
from ._serialization import deserialize, serialize
from ._types import typechecked
from .config import logger
from .exception import deprecation_warning
from .object import _Handle, _Provider


def _serialize_dict(data):
    return [api_pb2.DictEntry(key=serialize(k), value=serialize(v)) for k, v in data.items()]


class _DictHandle(_Handle, type_prefix="di"):
    """Handle for interacting with the contents of a `Dict`

    ```python
    stub.some_dict = modal.Dict.new()

    if __name__ == "__main__":
        with stub.run() as app:
            app.some_dict["message"] = "hello world"
    ```
    """

    async def get(self, key: Any) -> Any:
        """Get the value associated with the key.

        Raises `KeyError` if the key does not exist.
        """
        req = api_pb2.DictGetRequest(dict_id=self.object_id, key=serialize(key))
        resp = await retry_transient_errors(self._client.stub.DictGet, req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return deserialize(resp.value, self._client)

    async def contains(self, key: Any) -> bool:
        """Check if the key exists."""
        req = api_pb2.DictContainsRequest(dict_id=self.object_id, key=serialize(key))
        resp = await retry_transient_errors(self._client.stub.DictContains, req)
        return resp.found

    async def len(self) -> int:
        """
        Returns the length of the dictionary, _including any expired keys_.
        """
        req = api_pb2.DictLenRequest(dict_id=self.object_id)
        resp = await retry_transient_errors(self._client.stub.DictLen, req)
        return resp.len

    async def __getitem__(self, key: Any) -> Any:
        """Get an item from the dictionary."""
        return await self.get(key)

    async def update(self, **kwargs) -> None:
        """Update the dictionary with additional items."""
        serialized = _serialize_dict(kwargs)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await retry_transient_errors(self._client.stub.DictUpdate, req)

    async def put(self, key: Any, value: Any) -> None:
        """Add a specific key-value pair in the dictionary."""
        updates = {key: value}
        serialized = _serialize_dict(updates)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await retry_transient_errors(self._client.stub.DictUpdate, req)

    async def __setitem__(self, key: Any, value: Any) -> None:
        """Set a specific key-value pair in the dictionary.

        This function only works in a synchronous context.
        """
        return await self.put(key, value)

    async def pop(self, key: Any) -> Any:
        """Remove a key from the dictionary, returning the value if it exists."""
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=serialize(key))
        resp = await retry_transient_errors(self._client.stub.DictPop, req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return deserialize(resp.value, self._client)

    async def __delitem__(self, key: Any) -> Any:
        """Delete a key from the dictionary.

        This function only works in a synchronous context.
        """
        return await self.pop(key)

    async def __contains__(self, key: Any) -> bool:
        """Check if key in the dictionary exists.

        This function only works in a synchronous context.
        """
        return await self.contains(key)


DictHandle = synchronize_api(_DictHandle)


class _Dict(_Provider[_DictHandle]):
    """A distributed dictionary available to Modal apps.

    Keys and values can be essentially any object, so long as it can be
    serialized by `cloudpickle`, including Modal objects.

    **Lifetime of dictionary and its items**

    A `Dict`'s lifetime matches the lifetime of the app it's attached to, but invididual keys expire after 30 days.
    Because of this, `Dict`s are best used as a cache and not relied on for persistent storage.
    On app completion or after stopping an app any associated `Dict` objects are cleaned up.

    **Usage**

    This is the constructor object, used only to attach a `DictHandle` to an app.
    To interact with `Dict` contents, use `DictHandle` objects that are attached
    to the live app once an app is running.

    ```python
    import modal

    stub = modal.Stub()
    stub.some_dict = modal.Dict.new()
    # stub.some_dict["message"] = "hello world" # TypeError!

    if __name__ == "__main__":
        with stub.run() as app:
            handle = app.some_dict
            handle["message"] = "hello world"  # OK ✔️
    ```
    """

    @typechecked
    @staticmethod
    def new(data={}) -> "_Dict":
        """Create a new dictionary, optionally filled with initial data."""

        async def _load(resolver: Resolver, existing_object_id: Optional[str], handle: _DictHandle):
            serialized = _serialize_dict(data)
            req = api_pb2.DictCreateRequest(
                app_id=resolver.app_id, data=serialized, existing_dict_id=existing_object_id
            )
            response = await resolver.client.stub.DictCreate(req)
            logger.debug("Created dict with id %s" % response.dict_id)
            handle._hydrate(response.dict_id, resolver.client, None)

        return _Dict._from_loader(_load, "Dict()")

    def __init__(self, data={}):
        """`Dict({...})` is deprecated. Please use `Dict.new({...})` instead."""
        deprecation_warning(date(2023, 6, 27), self.__init__.__doc__)
        obj = _Dict.new(data)
        self._init_from_other(obj)

    @staticmethod
    def persisted(
        label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None
    ) -> "_Dict":
        """See `SharedVolume.persisted`."""
        return _Dict.new()._persist(label, namespace, environment_name)

    def persist(
        self, label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None
    ) -> "_Dict":
        """`Dict().persist("my-dict")` is deprecated. Use `Dict.persisted("my-dict")` instead."""
        deprecation_warning(date(2023, 6, 30), self.persist.__doc__)
        return self.persisted(label, namespace, environment_name)

    # Handle methods - temporary until we get rid of all user-facing handles
    async def get(self, key: Any) -> Any:
        return await self._handle.get(key)

    async def contains(self, key: Any) -> bool:
        return await self._handle.contains(key)

    async def len(self) -> int:
        return await self._handle.len()

    async def __getitem__(self, key: Any) -> Any:
        return await self._handle.__getitem__(key)

    async def update(self, **kwargs) -> None:
        return await self._handle.update(**kwargs)

    async def put(self, key: Any, value: Any) -> None:
        return await self._handle.put(key, value)

    async def __setitem__(self, key: Any, value: Any) -> None:
        return await self._handle.__setitem__(key, value)

    async def pop(self, key: Any) -> Any:
        return await self._handle.pop(key)

    async def __delitem__(self, key: Any) -> Any:
        return await self._handle.__delitem__(key)

    async def __contains__(self, key: Any) -> bool:
        return await self._handle.__contains(key)


Dict = synchronize_api(_Dict)
