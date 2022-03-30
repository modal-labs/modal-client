from typing import NamedTuple, Optional

from modal_proto import api_pb2

from ._app_singleton import get_container_app
from ._app_state import AppState
from ._object_meta import ObjectMeta
from .exception import InvalidError


class ObjectLabel(NamedTuple):
    # Local to this app
    local_tag: str

    # Different app
    app_name: Optional[str] = None
    object_label: Optional[str] = None
    namespace: Optional[int] = None  # api_pb2.DEPLOYMENT_NAMESPACE


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

    def __init__(self, app, tag=None, label=None):
        if app is None:
            raise InvalidError(f"Object {self} created without an app")
        if tag is not None:
            assert isinstance(tag, str)
            label = ObjectLabel(tag)
        self._label = label
        self._app = app
        self._object_id = None
        self._app_id = app.app_id

        # A bunch of initialization specific to objects that are created
        # prior to the app running (we should verify this)
        if label is not None:
            container_app = get_container_app()
            if container_app is not None:
                # If we're inside the container, then just lookup the tag and use
                # it if possible.
                if app != container_app:
                    raise Exception(f"app {app} is not container app {container_app}")
                object_id = app._get_object_id_by_tag(label.local_tag)
                if object_id is not None:
                    self.set_object_id(object_id, app)
            else:
                if app.state == AppState.NONE:
                    app._register_object(self)

    @classmethod
    async def create(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        if obj._app.state != AppState.RUNNING:
            raise InvalidError(f"{cls}.create(...): can only do this on a running app")
        object_id = await obj.load(obj._app)
        obj.set_object_id(object_id, obj._app)
        return obj

    async def load(self, app):
        raise NotImplementedError(f"Object factory of class {type(self)} has no load method")

    def _init_attributes(self, app=None, label=None):
        """Initialize attributes"""
        self._label = label
        self._object_id = None
        self._app_id = app.app_id
        self._app = app

    @classmethod
    def object_type_name(cls):
        return ObjectMeta.prefix_to_type[cls._type_prefix].__name__  # type: ignore

    @classmethod
    def _init_persisted(cls, object_id, app):
        prefix, _ = object_id.split("-")  # TODO: util method
        object_cls = ObjectMeta.prefix_to_type[prefix]
        obj = Object.__new__(object_cls)
        obj._init_attributes(app=app)
        obj.set_object_id(object_id, app)
        return obj

    def set_object_id(self, object_id, app):
        """Set the Modal internal object id"""
        self._object_id = object_id
        self._app = app
        self._app_id = app.app_id

    @property
    def object_id(self):
        """The Modal internal object id"""
        if self._app_id is not None and self._app is not None and self._app_id == self._app.app_id:
            return self._object_id

    @property
    def label(self):
        return self._label

    @property
    def tag(self):
        if self._label is not None:
            return self._label.local_tag
        else:
            return None

    @property
    def app(self):
        return self._app

    @classmethod
    def include(cls, app, app_name, object_label=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
        """Use an object published with `modal.App.deploy`"""
        # TODO: this is somewhat hacky
        # everything needs a local label right now, so let's contruct an artificial one

        local_tag = f"#SHARE({app_name}, {object_label}, {namespace})"
        label = ObjectLabel(local_tag, app_name, object_label, namespace)

        obj = Object.__new__(cls)
        Object.__init__(obj, app, label=label)
        return obj
