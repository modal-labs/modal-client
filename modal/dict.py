from typing import Any

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._serialization import deserialize, serialize
from .config import logger
from .object import Handle, Provider


def _serialize_dict(data):
    return [api_pb2.DictEntry(key=serialize(k), value=serialize(v)) for k, v in data.items()]


class _DictHandle(Handle, type_prefix="di"):
    """Handle for interacting with the contents of a `Dict`

    ```python
    stub.some_dict = modal.Dict()

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
        resp = await self._client.stub.DictGet(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return deserialize(resp.value, self._client)

    async def contains(self, key: Any) -> bool:
        """Check if the key exists."""
        req = api_pb2.DictContainsRequest(dict_id=self.object_id, key=serialize(key))
        resp = await self._client.stub.DictContains(req)
        return resp.found

    async def len(self) -> int:
        """Returns the length of the dictionary."""
        req = api_pb2.DictLenRequest(dict_id=self.object_id)
        resp = await self._client.stub.DictLen(req)
        return resp.len

    async def __getitem__(self, key: Any) -> Any:
        """Get an item from the dictionary."""
        return await self.get(key)

    async def update(self, **kwargs) -> None:
        """Update the dictionary with additional items."""
        serialized = _serialize_dict(kwargs)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._client.stub.DictUpdate(req)

    async def put(self, key: Any, value: Any) -> None:
        """Add a specific key-value pair in the dictionary."""
        updates = {key: value}
        serialized = _serialize_dict(updates)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._client.stub.DictUpdate(req)

    async def __setitem__(self, key: Any, value: Any) -> None:
        """Set a specific key-value pair in the dictionary.

        This function only works in a synchronous context.
        """
        return await self.put(key, value)

    async def pop(self, key: Any) -> Any:
        """Remove a key from the dictionary, returning the value if it exists."""
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=serialize(key))
        resp = await self._client.stub.DictPop(req)
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


DictHandle, AioDictHandle = synchronize_apis(_DictHandle)


class _Dict(Provider[_DictHandle]):
    """A distributed dictionary available to Modal apps.

    Keys and values can be essentially any object, so long as it can be
    serialized by `cloudpickle`, including Modal objects.

    This is the constructor object, which can not be interacted with. Use the `DictHandle` that
    gets attached to the live app once an app is running to interact with the Dict contents
    """

    def __init__(self, data={}):
        """Create a new dictionary, optionally filled with initial data."""
        self._data = data
        super().__init__()

    async def _load(self, client, app_id, loader, message_callback, existing_dict_id):
        serialized = _serialize_dict(self._data)
        req = api_pb2.DictCreateRequest(app_id=app_id, data=serialized, existing_dict_id=existing_dict_id)
        response = await client.stub.DictCreate(req)
        logger.debug("Created dict with id %s" % response.dict_id)
        return _DictHandle(client, response.dict_id)


Dict, AioDict = synchronize_apis(_Dict)
