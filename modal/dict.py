import queue  # The system library
import uuid
from typing import Any, List

from ._async_utils import retry
from ._client import Client
from .config import logger
from .object import Object, requires_create
from .proto import api_pb2


class Dict(Object):
    """A distributed dictionary.

    Keys and values can be essentially any object, so long as it can be
    serialized by cloudpickle, including Modal objects.
    """

    def _init(self, data={}):
        self.data = data

    def _serialize_dict(self, session, data):
        return [api_pb2.DictEntry(key=session.serialize(k), value=session.serialize(v)) for k, v in data.items()]

    async def _create_impl(self, session):
        serialized = self._serialize_dict(session, self.data)
        req = api_pb2.DictCreateRequest(session_id=session.session_id, data=serialized)
        response = await session.client.stub.DictCreate(req)
        logger.debug("Created dict with id %s" % response.dict_id)
        return response.dict_id

    @requires_create
    async def get(self, key):
        """Get the value associated with the key

        Raises KeyError if the key does not exist.
        """

        req = api_pb2.DictGetRequest(dict_id=self.object_id, key=self._session.serialize(key))
        resp = await self._session.client.stub.DictGet(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return self._session.deserialize(resp.value)

    @requires_create
    async def contains(self, key):
        """Check if the key exists"""
        req = api_pb2.DictContainsRequest(dict_id=self.object_id, key=self._session.serialize(key))
        resp = await self._session.client.stub.DictContains(req)
        return resp.found

    @requires_create
    async def len(self):
        """The length of the dictionary"""
        req = api_pb2.DictLenRequest(dict_id=self.object_id)
        resp = await self._session.client.stub.DictLen(req)
        return resp.len

    @requires_create
    async def __getitem__(self, key):
        """Get an item from the dictionary"""
        return await self.get(key)

    @requires_create
    async def update(self, **kwargs):
        """Update the dictionary with items

        Key-value pairs to update should be specified as keyword-arguments
        """
        serialized = self._serialize_dict(self._session, kwargs)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._session.client.stub.DictUpdate(req)

    @requires_create
    async def put(self, key, value):
        """Set the specific key/value pair in the dictionary"""
        updates = {key: value}
        serialized = self._serialize_dict(self._session, updates)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._session.client.stub.DictUpdate(req)

    # NOTE: setitem only works in a synchronous context.
    @requires_create
    async def __setitem__(self, key, value):
        """Set the specific key/value pair in the dictionary

        Only works in a synchronous context
        """
        return await self.put(key, value)

    @requires_create
    async def pop(self, key):
        """Remove the specific key from the dictionary"""
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=self._session.serialize(key))
        resp = await self._session.client.stub.DictPop(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return self._session.deserialize(resp.value)

    @requires_create
    async def __delitem__(self, key):
        """Delete the specific key from the dictionary

        Only works in a synchronous context
        """
        return await self.pop(key)
