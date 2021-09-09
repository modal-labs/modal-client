from .async_utils import synchronizer
from .config import logger


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
            # TODO: fix this circular import later
            from .client import Client
            self.client = await Client.from_env()
        return self.client
