import asyncio
import functools
import inspect
import uuid

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


class Object(metaclass=ObjectMeta):
    # A bit ugly to leverage implemenation inheritance here, but I guess you could
    # roughly think of this class as a mixin
    RANDOM_TAG = []  # sentinel

    def __init__(self, tag, session=None):
        logger.debug(f"Creating object {self}")
        if tag is self.RANDOM_TAG:
            tag = str(uuid.uuid4())
        else:
            assert tag

        self.tag = tag
        self.session = session
        if session:
            self.session.register(self.tag, self)

        # TODO: if the session is running, enforce that tag is set
        # TODO: if the object has methods that requires creation, enforce that session is set

    async def create_or_get(self, session):
        raise NotImplementedError

    @property
    def object_id(self):
        return self.session.get_object_id(self.tag)


def requires_create(method):
    # TODO: this does not work for generators (need to do `async for z in await f()` )
    # See the old requires_join_generator function for how to make this work

    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if not self.object_id:
            raise Exception(f"Error running method {method} on object {self}: object is not created yet")
        return method(self, *args, **kwargs)

    return wrapped_method
