from .config import logger
from .object import Object
from .proto import api_pb2


class Dict(Object, type_prefix="di"):
    """A distributed dictionary.

    Keys and values can be essentially any object, so long as it can be
    serialized by cloudpickle, including Modal objects.
    """

    @classmethod
    def _serialize_dict(self, app, data):
        return [api_pb2.DictEntry(key=app.serialize(k), value=app.serialize(v)) for k, v in data.items()]

    @classmethod
    async def create(cls, data={}, app=None):
        app = cls._get_app(app)
        serialized = cls._serialize_dict(app, data)
        req = api_pb2.DictCreateRequest(app_id=app.app_id, data=serialized)
        response = await app.client.stub.DictCreate(req)
        logger.debug("Created dict with id %s" % response.dict_id)
        return cls._create_object_instance(response.dict_id, app)

    async def get(self, key):
        """Get the value associated with the key

        Raises KeyError if the key does not exist.
        """

        req = api_pb2.DictGetRequest(dict_id=self.object_id, key=self._app.serialize(key))
        resp = await self._app.client.stub.DictGet(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return self._app.deserialize(resp.value)

    async def contains(self, key):
        """Check if the key exists"""
        req = api_pb2.DictContainsRequest(dict_id=self.object_id, key=self._app.serialize(key))
        resp = await self._app.client.stub.DictContains(req)
        return resp.found

    async def len(self):
        """The length of the dictionary"""
        req = api_pb2.DictLenRequest(dict_id=self.object_id)
        resp = await self._app.client.stub.DictLen(req)
        return resp.len

    async def __getitem__(self, key):
        """Get an item from the dictionary"""
        return await self.get(key)

    async def update(self, **kwargs):
        """Update the dictionary with items

        Key-value pairs to update should be specified as keyword-arguments
        """
        serialized = self._serialize_dict(self._app, kwargs)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._app.client.stub.DictUpdate(req)

    async def put(self, key, value):
        """Set the specific key/value pair in the dictionary"""
        updates = {key: value}
        serialized = self._serialize_dict(self._app, updates)
        req = api_pb2.DictUpdateRequest(dict_id=self.object_id, updates=serialized)
        await self._app.client.stub.DictUpdate(req)

    # NOTE: setitem only works in a synchronous context.
    async def __setitem__(self, key, value):
        """Set the specific key/value pair in the dictionary

        Only works in a synchronous context
        """
        return await self.put(key, value)

    async def pop(self, key):
        """Remove the specific key from the dictionary"""
        req = api_pb2.DictPopRequest(dict_id=self.object_id, key=self._app.serialize(key))
        resp = await self._app.client.stub.DictPop(req)
        if not resp.found:
            raise KeyError(f"KeyError: {key} not in dict {self.object_id}")
        return self._app.deserialize(resp.value)

    async def __delitem__(self, key):
        """Delete the specific key from the dictionary

        Only works in a synchronous context
        """
        return await self.pop(key)
