import asyncio
import functools
import inspect

from ._decorator_utils import decorator_with_options
from ._function_utils import FunctionInfo
from ._object_meta import ObjectMeta
from ._session_singleton import get_session_singleton
from ._session_state import SessionState
from .config import logger


class Object(metaclass=ObjectMeta):
    """The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.

    The Object base class provides some common initialization patterns. There
    are 2 main ways to initialize and use Objects.

    The first pattern is to directly instantiate objects and pass them as
    parameters when required. This is common for data structures (e.g. ``dict_
    = Dict(session=session); main(dict_)``).
    Instances of Object are just handles with an object ID and some associated
    metadata, so they can be safely serialized or passed as parameters to Modal
    functions.

    The second pattern is to declare objects in the global scope. This is most
    common for global and unique objects like Images. In this case, some
    identifier is required to matching up local and remote copies objects and
    to avoid double initialization.

    The solution is to declare objects as "factory functions". A factory
    function is a function decorated with ``@Type.factory`` whose body
    initializes and returns an object of type ``Type``. This object will be
    automatically initialized once and tagged with the function name/module.
    The decorator will convert the function into a proxy object, so it can be
    used directly.
    """

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
            # TODO: tag is only ever set when (a) created from a Factory (see below)
            # and (b) created from a Function: maybe this could be a separate
            # constructor method.
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
        """Set the Modal internal object id"""
        self._object_id = object_id
        self._session_id = session_id

    @property
    def object_id(self):
        """The Modal internal object id"""
        if self._session_id is not None and self._session is not None and self._session_id == self._session.session_id:
            return self._object_id

    @classmethod
    def _new(cls, **kwargs):
        obj = Object.__new__(cls)
        obj._init(**kwargs)
        return obj

    @classmethod
    def use(cls, session, path):
        """Use an object published with :py:meth:`modal.session.Session.share`"""
        # TODO: session should be a 2nd optional arg
        obj = cls._new(session=session, share_path=path)
        if session:
            session.create_object_later(obj)
        return obj

    @classmethod
    @decorator_with_options
    def factory(cls, fun, session=None):
        """Decorator to mark a "factory function".

        Factory functions work like "named promises" for Objects, they are
        automatically tagged by name/path so they will always refer to the same
        underlying Object across machines. The function body should return the
        desired initialized object. The body will only be run on the local
        machine, so it may access local resources/files.

        The decorated function can be used directly as a proxy object (if no
        parameters are needed), or can be called with arguments and will return
        a proxy object.
        """

        if not hasattr(cls, "_factory_class"):
            # TODO: is there some nicer way we could do this rather than creating a class inside a function?
            # Maybe we could use the ObjectMeta meta class?
            class Factory(cls):
                """Acts as a wrapper for a transient Object.

                Puts a tag and optionally a session on it. Otherwise just "steals" the object id from the
                underlying object at construction time.
                """

                def __init__(self, fun, session, args_and_kwargs=None):  # TODO: session?
                    functools.update_wrapper(self, fun)
                    self._fun = fun
                    self._args_and_kwargs = args_and_kwargs
                    self._session = session
                    function_info = FunctionInfo(fun)

                    # This is the only place where tags are being set on objects,
                    # besides Function
                    tag = function_info.get_tag(args_and_kwargs)
                    Object.__init__(self, session=session, tag=tag)

                async def _create_impl(self, session):
                    if self._args_and_kwargs is not None:
                        args, kwargs = self._args_and_kwargs
                        object = self._fun(*args, **kwargs)
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
                    assert self._args_and_kwargs is None
                    return Factory(self._fun, self._session, args_and_kwargs=(args, kwargs))

                def __repr__(self):
                    return "<{}.{} {!r}>".format(
                        type(self).__module__, type(self).__qualname__, getattr(self, "tag", None)
                    )

            Factory.__module__ = cls.__module__
            Factory.__qualname__ = cls.__qualname__ + ".Factory"
            Factory.__doc__ = "\n\n".join(filter(None, [Factory.__doc__, cls.__doc__]))
            cls._factory_class = Factory

        return cls._factory_class(fun, session)


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
