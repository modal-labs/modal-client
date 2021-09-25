import queue  # The system library
import uuid
from typing import Any, List

from .async_utils import retry
from .client import Client
from .config import logger
from .object import Object, requires_join
from .proto import api_pb2
from .session import Session


class Dict(Object):
    def __init__(self, init_data={}, session_tag=None):
        if session_tag is None:
            session_tag = str(uuid.uuid4())
        super().__init__(session_tag=session_tag, args={"init_data": init_data})

    async def _join(self):
        req = api_pb2.DictCreateRequest(session_id=self.session.session_id, data=self.args.init_data)
        response = await self.client.stub.DictCreate(req)
        logger.debug("Created dict with id %s" % response.dict_id)
        return response.dict_id

    @requires_join
    async def get(self, key):
        req = api_pb2.DictGetRequest(dict_id=self.object_id, key=key)
        resp = await self.client.stub.DictGet(req)
        return resp.value

    @requires_join
    async def __getitem__(self, key):
        return await self.get(key)

    @requires_join
    async def update(self, **kwargs):
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=kwargs)
        await self.client.stub.DictUpdate(req)

    @requires_join
    async def put(self, key, value):
        updates = {key: value}
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=updates)
        await self.client.stub.DictUpdate(req)

    @requires_join
    async def pop(self, key):
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=key)
        resp = await self.client.stub.DictPop(req)
        return resp.value
