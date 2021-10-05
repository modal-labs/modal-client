import queue  # The system library
import uuid
from typing import Any, List

from .async_utils import retry
from .client import Client
from .config import logger
from .object import Object, requires_join
from .proto import api_pb2


class Dict(Object):
    def __init__(self, init_data={}, DEPRECATED_session_tag=None):
        if DEPRECATED_session_tag is None:
            DEPRECATED_session_tag = str(uuid.uuid4())
        super().__init__(DEPRECATED_session_tag=DEPRECATED_session_tag, args={"init_data": init_data})

    def _serialize_values(self, data):
        return {k: self.client.serialize(v) for k, v in data.items()}

    async def _join(self):
        serialized = self._serialize_values(self.args.init_data)
        req = api_pb2.DictCreateRequest(session_id=self.DEPRECATED_session.session_id, data=serialized)
        response = await self.client.stub.DictCreate(req)
        logger.debug("Created dict with id %s" % response.dict_id)
        return response.dict_id

    @requires_join
    async def get(self, key):
        req = api_pb2.DictGetRequest(dict_id=self.object_id, key=key)
        resp = await self.client.stub.DictGet(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return self.client.deserialize(resp.value)

    @requires_join
    async def __getitem__(self, key):
        return await self.get(key)

    @requires_join
    async def update(self, **kwargs):
        serialized = self._serialize_values(kwargs)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self.client.stub.DictUpdate(req)

    @requires_join
    async def put(self, key, value):
        updates = {key: value}
        serialized = self._serialize_values(updates)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self.client.stub.DictUpdate(req)

    # NOTE: setitem only works in a synchronous context.
    @requires_join
    async def __setitem__(self, key, value):
        return await self.put(key, value)

    @requires_join
    async def pop(self, key):
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=key)
        resp = await self.client.stub.DictPop(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return self.client.deserialize(resp.value)
