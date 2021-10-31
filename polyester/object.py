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
    def __init__(self, tag=None, session=None):
        logger.debug(f"Creating object {self}")

        # If the session is not running, enforce that tag is set
        # This is probably because the object is created in a global scope
        # We need to have a tag set or else different processes won't be able to "reconcile"
        if session and not session.client:  # TODO: dumb check
            if not tag:
                raise Exception("Objects created on non-running sessions need to have a tag set")
        
        # TODO: if the object has methods that requires creation, enforce that session is set

        if tag is None:
            tag = str(uuid.uuid4())

        self.tag = tag
        self.session = session
        if session:
            self.session.register(self.tag, self)

    async def create_or_get(self, session):
        raise NotImplementedError

    async def create(self):
        # TODO: this method name is pretty inconsistent
        return await self.session.create_or_get_object(self)

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
