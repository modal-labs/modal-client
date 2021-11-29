import asyncio
import functools
import inspect

from .async_utils import synchronizer
from .config import logger
from .session_state import SessionState


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
    def __init__(self, session=None, tag=None):
        logger.debug(f"Creating object {self}")

        self._init(session=session, tag=tag)
        if session is not None:
            self._session.register(self)

    def _init(self, session=None, tag=None, share_path=None):
        self._object_id = None
        self._session_id = None
        self.share_path = share_path
        self.tag = tag
        self._session = session

    async def _create_impl(self, session):
        # Overloaded in subclasses to do the actual logic
        raise NotImplementedError

    def set_object_id(self, object_id, session_id):
        self._object_id = object_id
        self._session_id = session_id

    @property
    def object_id(self):
        if self._session_id is not None and self._session is not None and self._session_id == self._session.session_id:
            return self._object_id

    @classmethod
    def new(cls, **kwargs):
        obj = Object.__new__(cls)
        obj._init(**kwargs)
        return obj

    @classmethod
    def use(cls, session, path):
        # TODO: this is a bit ugly, because it circumvents the contructor, which means
        # it might not always work (eg you can't do DebianSlim.use("foo"))
        # This interface is a bit TBD, let's think more about it
        obj = cls.new(session=session, share_path=path)
        if session:
            session.register(obj)  # TODO: is this needed?
        return obj


def requires_create_generator(method):
    @functools.wraps(method)
    async def wrapped_method(self, *args, **kwargs):
        if not self._session:
            raise Exception("Can only run this method on an object with a session set")
        if self._session.state != SessionState.RUNNING:
            raise Exception("Can only run this method on an object with a running session")

        # Flush all objects to the session
        await self._session.flush_objects()

        async for ret in method(self, *args, **kwargs):
            yield ret

    return wrapped_method


def requires_create(method):
    @functools.wraps(method)
    async def wrapped_method(self, *args, **kwargs):
        if not self._session:
            raise Exception("Can only run this method on an object with a session set")
        if self._session.state != SessionState.RUNNING:
            raise Exception("Can only run this method on an object with a running session")

        # Flush all objects to the session
        await self._session.flush_objects()

        return await method(self, *args, **kwargs)

    return wrapped_method
