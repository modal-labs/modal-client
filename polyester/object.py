import io

from .async_utils import synchronizer
from .client import Client
from .config import logger
from .serialization import Pickler, Unpickler


class ObjectMeta(type):
    type_to_name = {}
    name_to_type = {}

    def __new__(metacls, name, bases, dct):
        # Synchronize class
        new_cls = synchronizer.create_class(metacls, name, bases, dct)

        # Register class as serializable
        ObjectMeta.type_to_name[new_cls] = name
        ObjectMeta.name_to_type[name] = new_cls

        logger.debug(f'Created Object class {name}')
        return new_cls


class Object(metaclass=ObjectMeta):
    # A bit ugly to leverage implemenation inheritance here, but I guess you could
    # roughly think of this class as a mixin
    def __init__(self, client=None, tag=None, server_id=None):
        self.client = client
        self.tag = tag
        self.server_id = server_id

    async def _get_client(self):
        if self.client is None:
            self.client = await Client.from_env()
        return self.client

    def _serialize(self, client, obj):
        ''' Serializes object and replaces all references to the client class by a placeholder.'''
        buf = io.BytesIO()
        Pickler(client, ObjectMeta.type_to_name, buf).dump(obj)
        return buf.getvalue()

    def _deserialize(self, client, s: bytes):
        ''' Deserializes object and replaces all client placeholders by self.'''
        return Unpickler(client, ObjectMeta.name_to_type, io.BytesIO(s)).load()
