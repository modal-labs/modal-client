import queue  # The system library
import uuid
from typing import Any, List

from ._async_utils import retry
from ._client import Client
from .config import logger
from .object import Object, requires_create
from .proto import api_pb2


class Dict(Object):
    def __init__(self, session, data={}):
        super().__init__(session=session)
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
        req = api_pb2.DictGetRequest(dict_id=self.object_id, key=self._session.serialize(key))
        resp = await self._session.client.stub.DictGet(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return self._session.deserialize(resp.value)

    @requires_create
    async def contains(self, key):
        req = api_pb2.DictContainsRequest(dict_id=self.object_id, key=self._session.serialize(key))
        resp = await self._session.client.stub.DictContains(req)
        return resp.found

    @requires_create
    async def len(self):
        req = api_pb2.DictLenRequest(dict_id=self.object_id)
        resp = await self._session.client.stub.DictLen(req)
        return resp.len

    @requires_create
    async def __getitem__(self, key):
        return await self.get(key)

    @requires_create
    async def update(self, **kwargs):
        serialized = self._serialize_dict(self._session, kwargs)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._session.client.stub.DictUpdate(req)

    @requires_create
    async def put(self, key, value):
        updates = {key: value}
        serialized = self._serialize_dict(self._session, updates)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._session.client.stub.DictUpdate(req)

    # NOTE: setitem only works in a synchronous context.
    @requires_create
    async def __setitem__(self, key, value):
        return await self.put(key, value)

    @requires_create
    async def pop(self, key):
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=self._session.serialize(key))
        resp = await self._session.client.stub.DictPop(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return self._session.deserialize(resp.value)
