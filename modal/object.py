from ._decorator_utils import decorator_with_options
from ._factory import Factory
from ._object_meta import ObjectMeta
from ._session_singleton import (
    get_container_session,
    get_default_session,
    get_running_session,
)
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

    @classmethod
    async def create(cls, *args, **kwargs):
        raise NotImplementedError("Class {cls} does not implement a .create(...) constructor!")

    def _init_attributes(self, tag=None):
        """Initialize attributes"""
        self.tag = tag
        self._object_id = None
        self._session_id = None
        self._session = None

    @classmethod
    def get_session(cls, session=None):
        """Helper method for subclasses."""
        if not session:
            session = get_container_session()
        if not session:
            session = get_default_session()
        return session

    @classmethod
    def create_object_instance(cls, object_id, session):
        """Helper method for subclass constructors."""
        obj = Object.__new__(cls)
        obj._init_attributes()

        if object_id is None:
            raise Exception(f"object_id for object of type {type(obj)} is None")

        obj.set_object_id(object_id, session)
        return obj

    def _init_static(self, session, tag, register_on_default_session=False):
        """Create a new tagged object.

        This is only used by the Factory or Function constructors

        register_on_default_session is set to True for Functions
        """
        # TODO: move this into Factory?

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

    def is_factory(self):
        return isinstance(self, Factory)

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
        a proxy object. In the latter case, the session will be passed as the
        first argument. Factories can be synchronous or asynchronous. however,
        synchronous factories may cause issues if they operate on Modal objects.

        .. code-block:: python

           @Image.factory
           def factory():
               return OtherImage()

        .. code-block:: python

           @Queue.factory
           async def factory(session, initial_value=42):
               q = Queue(session)
               await q.put(initial_value)
               return q
        """
        return cls._user_factory_class(fun, session)

    @classmethod
    def use(cls, session, label, namespace=api_pb2.ShareNamespace.ACCOUNT):
        """Use an object published with :py:meth:`modal.session.Session.share`"""
        return cls._shared_object_factory_class(session, label, namespace)
