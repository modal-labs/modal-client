from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._serialization import deserialize, serialize
from .config import logger
from .object import Object


class _Dict(Object, type_prefix="di"):
    """A distributed dictionary.

    Keys and values can be essentially any object, so long as it can be
    serialized by cloudpickle, including Modal objects.
    """

    @classmethod
    def _serialize_dict(self, data):
        return [api_pb2.DictEntry(key=serialize(k), value=serialize(v)) for k, v in data.items()]

    def __init__(self, data={}):
        self._data = data
        super().__init__()

    async def _load(self, client, app_id, existing_dict_id):
        serialized = self._serialize_dict(self._data)
        req = api_pb2.DictCreateRequest(app_id=app_id, data=serialized, existing_dict_id=existing_dict_id)
        response = await client.stub.DictCreate(req)
        logger.debug("Created dict with id %s" % response.dict_id)
        return response.dict_id

    async def get(self, key):
        """Get the value associated with the key

        Raises KeyError if the key does not exist.
        """
        req = api_pb2.DictGetRequest(dict_id=self.object_id, key=serialize(key))
        resp = await self._client.stub.DictGet(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return deserialize(resp.value, self._client)

    async def contains(self, key):
        """Check if the key exists"""
        req = api_pb2.DictContainsRequest(dict_id=self.object_id, key=serialize(key))
        resp = await self._client.stub.DictContains(req)
        return resp.found

    async def len(self):
        """The length of the dictionary"""
        req = api_pb2.DictLenRequest(dict_id=self.object_id)
        resp = await self._client.stub.DictLen(req)
        return resp.len

    async def __getitem__(self, key):
        """Get an item from the dictionary"""
        return await self.get(key)

    async def update(self, **kwargs):
        """Update the dictionary with items

        Key-value pairs to update should be specified as keyword-arguments
        """
        serialized = self._serialize_dict(kwargs)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._client.stub.DictUpdate(req)

    async def put(self, key, value):
        """Set the specific key/value pair in the dictionary"""
        updates = {key: value}
        serialized = self._serialize_dict(updates)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._client.stub.DictUpdate(req)

    # NOTE: setitem only works in a synchronous context.
    async def __setitem__(self, key, value):
        """Set the specific key/value pair in the dictionary

        Only works in a synchronous context
        """
        return await self.put(key, value)

    async def pop(self, key):
        """Remove the specific key from the dictionary"""
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=serialize(key))
        resp = await self._client.stub.DictPop(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return deserialize(resp.value, self._client)

    async def __delitem__(self, key):
        """Delete the specific key from the dictionary

        Only works in a synchronous context
        """
        return await self.pop(key)

    async def __contains__(self, key):
        """Check if the key exists

        Only works in a synchronous context
        """
        return await self.contains(key)


Dict, AioDict = synchronize_apis(_Dict)
