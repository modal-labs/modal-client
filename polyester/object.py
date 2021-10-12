import asyncio
import functools
import inspect

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

    def __init__(self, args=None):
        logger.debug(f"Creating object {self}")

        # TODO: should we make these attributes hidden for subclasses?
        # (i.e. "private" not even "protected" to use the C++ terminology)
        # Feels like there could be some benefits of doing so
        if isinstance(args, dict):
            self.args = Args(args)
        elif isinstance(args, Args):
            self.args = args
        elif args is None:
            self.args = None
        else:
            raise Exception(f"{args} of type {type(args)} must be instance of (dict, Args, NoneType)")

        # Default values for non-created objects
        self.created = False
        self.client = None
        self.session = None

    async def create_or_get(self):
        raise NotImplementedError

    def clone(self):
        cls = type(self)
        new_self = cls.__new__(cls)
        new_self.args = self.args
        return new_self

    async def set_context(self, session, client):
        self.session = session
        self.client = client

    async def create_from_scratch(self):
        self.object_id = await self.create_or_get()
        self.created = True
        return self.object_id

    def create_from_id(self, object_id):
        self.object_id = object_id
        self.created = True


def requires_create(method):
    # TODO: this does not work for generators (need to do `async for z in await f()` )
    # See the old requires_join_generator function for how to make this work

    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if not self.created:
            raise Exception(f"Error running method {method} on object {self}: object is not created yet")
        return method(self, *args, **kwargs)

    return wrapped_method
