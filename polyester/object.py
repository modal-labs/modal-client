import asyncio
import functools
import inspect

from .config import logger
from .function_utils import FunctionInfo
from .object_meta import ObjectMeta
from .session_singleton import get_session_singleton
from .session_state import SessionState


class Object(metaclass=ObjectMeta):
    # A bit ugly to leverage implemenation inheritance here, but I guess you could
    # roughly think of this class as a mixin
    def __init__(self, session=None, tag=None):
        logger.debug(f"Creating object {self}")

        self._init(session=session, tag=tag)

        # Fallback to singleton session
        s = session or get_session_singleton()

        if tag is not None:
            # See if we can populate this with an object id
            # (this happens if we're inside the container)
            if s:
                object_id = s.get_object_id_by_tag(tag)
            else:
                object_id = None

            if object_id:
                self._session = s
                self._object_id = object_id
                self._session_id = s.session_id
            elif session:
                # If not, let's register this for creation later
                # Only if this was explicitly created with a session
                self._session = session
                self._session.create_object_later(self)

        elif s:
            # If there's a session around, create this
            self._session = s
            s.create_object_later(self)

    def _init(self, session=None, tag=None, share_path=None):
        self._object_id = None
        self._session_id = None
        self.share_path = share_path
        self.tag = tag
        self._session = session

    async def _create_impl(self, session):
        # Overloaded in subclasses to do the actual logic
        raise NotImplementedError(f"Object of class {type(self)} has no _create_impl method")

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
        # TODO: session should be a 2nd optional arg
        obj = cls.new(session=session, share_path=path)
        if session:
            session.create_object_later(obj)
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


def make_factory(cls):
    assert issubclass(cls, Object)

    class Factory(cls):
        """Acts as a wrapper for a transient Object.

        Puts a tag and optionally a session on it. Otherwise just "steals" the object id from the
        underlying object at construction time.
        """

        def __init__(self, fun, args=None, kwargs=None):  # TODO: session?
            self._fun = fun
            self._args = args
            self._kwargs = kwargs
            function_info = FunctionInfo(fun)
            tag = function_info.get_tag(args, kwargs)
            super().__init__(session=None, tag=tag)

        async def _create_impl(self, session):
            if self._args is not None:
                object = self._fun(*self._args, **self._kwargs)
            else:
                object = self._fun()
            assert isinstance(object, cls)
            object_id = await session.create_object(object)
            # Note that we can "steal" the object id from the other object
            # and set it on this object. This is a general trick we can do
            # to other objects too.
            return object_id

        def __call__(self, *args, **kwargs):
            """Binds arguments to this object."""
            assert self._args is None
            assert self._kwargs is None
            return Factory(self._fun, args=args, kwargs=kwargs)

    # Meant to be used as a decorator
    return Factory
