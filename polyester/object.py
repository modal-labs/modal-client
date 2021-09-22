import asyncio

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

    def __init__(self, object_id=None, args=None):
        # TODO: should we make these attributes hidden for subclasses?
        # (i.e. "private" not even "protected" to use the C++ terminology)
        # Feels like there could be some benefits of doing so
        self.join_lock = None
        self.object_id = object_id
        if isinstance(args, dict):
            self.args = Args(args)
        elif isinstance(args, Args):
            self.args = args
        elif args is None:
            self.args = None
        else:
            raise Exception(f"{args} of type {type(args)} must be instance of (dict, Args, NoneType)")

    async def _join(self):
        raise NotImplementedError

    async def join(self, client, session):
        if self.object_id is None:
            if self.join_lock is None:
                # There's no race condition here because it's cooperative multithreading
                self.join_lock = asyncio.Lock()
            async with self.join_lock:
                if self.object_id is None:
                    self.object_id = await self._join(client, session)  # TODO: pass it self.args?
                    assert self.object_id
        return self.object_id

    def __setattr__(self, k, v):
        if k not in ["object_id", "args", "join_lock"]:
            raise AttributeError(f"Cannot set attribute {k}")
        self.__dict__[k] = v
