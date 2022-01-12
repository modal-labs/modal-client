import asyncio
import functools
import inspect

from ._decorator_utils import decorator_with_options
from ._function_utils import FunctionInfo
from ._object_meta import ObjectMeta
from ._session_singleton import (
    get_container_session,
    get_default_session,
    get_running_session,
)
from ._session_state import SessionState
from .config import logger
from .proto import api_pb2


class Object(metaclass=ObjectMeta):
    """The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.

    The Object base class provides some common initialization patterns. There
    are 2 main ways to initialize and use Objects.

    The first pattern is to directly instantiate objects and pass them as
    parameters when required. This is common for data structures (e.g. ``dict_
    = Dict(); main(dict_)``).
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

    def __init__(self, *args, **kwargs):
        raise Exception("Direct construction of Object is not possible! Use factories or .create(...)!")

    def _init_attributes(self, tag=None, share_label=None, share_namespace=None):
        """Initialize attributes"""
        self.share_label = share_label
        self.share_namespace = share_namespace
        self.tag = tag
        self._object_id = None
        self._session_id = None
        self._session = None

    @classmethod
    async def create(cls, *args, **kwargs):
        """Creates an object.

        If no session is specified, the object is registered on the default session.
        """
        session = kwargs.pop("session", None)

        # Create object and initialize it
        # TODO(erikbern): I'm trying to minimize code changes in order to get the async
        # constructors working, so I'm simply just reusing existing constructors for now.
        # It's probably much better to get rid of this dumb layer of indirection since
        # pretty much all the constructors do is to save a bunch of values to the object
        # that's only ever used by _create_impl anyway. Let's revisit shortly
        obj = Object.__new__(cls)
        obj._init_attributes()
        obj._init(*args, **kwargs)

        if not session:
            session = get_container_session()
        if not session:
            session = get_default_session()

        # Now, create the object on the server
        object_id = await obj._create_impl(session)
        if object_id is None:
            raise Exception(f"object_id for object of type {type(obj)} is None")

        obj.set_object_id(object_id, session)
        return obj

    def _init_static(self, session, tag, register_on_default_session=False):
        """Create a new tagged object.

        This is only used by the Factory or Function constructors

        register_on_default_session is set to True for Functions
        """

        assert tag is not None
        self._init_attributes(tag=tag)

        container_session = get_container_session()
        if container_session is not None:
            # If we're inside the container, then just lookup the tag and use
            # it if possible.

            session = container_session
            object_id = session.get_object_id_by_tag(tag)
            if object_id is not None:
                self.set_object_id(object_id, session)
        else:
            if not session and register_on_default_session:
                session = get_default_session()

            if session:
                session.register_object(self)

    async def _create_impl(self, session):
        # Overloaded in subclasses to do the actual logic
        raise NotImplementedError(f"Object of class {type(self)} has no _create_impl method")

    def set_object_id(self, object_id, session):
        """Set the Modal internal object id"""
        self._object_id = object_id
        self._session = session
        self._session_id = session.session_id

    @property
    def object_id(self):
        """The Modal internal object id"""
        if self._session_id is not None and self._session is not None and self._session_id == self._session.session_id:
            return self._object_id

    @classmethod
    def use(cls, session, label, namespace=api_pb2.ShareNamespace.ACCOUNT):
        """Use an object published with :py:meth:`modal.session.Session.share`"""
        # TODO: session should be a 2nd optional arg
        obj = Object.__new__(cls)
        obj._init_attributes(share_label=label, share_namespace=namespace)
        if session:
            session.register_object(obj)
        return obj

    @classmethod
    @decorator_with_options
    def factory(cls, fun, session=None):
        """Decorator to mark a "factory function".

        Factory functions work like "named promises" for Objects, they are
        automatically tagged by name/label so they will always refer to the same
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
                    self.function_info = FunctionInfo(fun)

                    # This is the only place where tags are being set on objects,
                    # besides Function
                    tag = self.function_info.get_tag(args_and_kwargs)
                    Object._init_static(self, session=session, tag=tag)

                async def _create_impl(self, session):
                    if get_container_session() is not None:
                        assert False

                    if self._args_and_kwargs is not None:
                        args, kwargs = self._args_and_kwargs
                        obj = self._fun(*args, **kwargs)
                    else:
                        obj = self._fun()
                    if inspect.iscoroutine(obj):
                        obj = await obj
                    if not isinstance(obj, cls):
                        raise TypeError(f"expected {obj} to have type {cls}")
                    object_id = await session.create_object(obj)
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
