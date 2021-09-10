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

        logger.debug(f"Created Object class {name}")
        return new_cls


class Args:
    def __init__(self, data):
        self.__dict__["data"] = data if data is not None else {}

    def __getattr__(self, k):
        return self.__dict__["data"][k]

    def __setattr__(self, k, v):
        raise AttributeError("Args object is immutable")


class Object(metaclass=ObjectMeta):
    # A bit ugly to leverage implemenation inheritance here, but I guess you could
    # roughly think of this class as a mixin

    def __init__(self, client=None, id=None, args=None):
        self.client = client
        self.id = id
        self.args = Args(args)

    async def _get_client(self):
        if self.client is None:
            # TODO: fix this circular import later
            from .client import Client

            self.client = await Client.from_env()
        return self.client
