from typing import NamedTuple, Optional

from modal_proto import api_pb2

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
    """

    def __init__(self, app, tag=None, label=None, object_id=None):
        if app is None:
            raise InvalidError(f"Object {self} created without an app")
        if tag is not None:
            assert isinstance(tag, str)
            label = ObjectLabel(tag)
        self._label = label
        self._app = app
        self._object_id = object_id
        self._app_id = app.app_id

    @classmethod
    async def create(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        if obj._app.state != AppState.RUNNING:
            raise InvalidError(f"{cls.__name__}.create(...): can only do this on a running app")
        object_id = await obj.load(obj._app, None)
        obj_2 = Object.__new__(cls)
        Object.__init__(obj_2, obj._app, object_id=object_id)
        return obj_2

    async def load(self, app, existing_object_id):
        raise NotImplementedError(f"Object factory of class {type(self)} has no load method")

    @classmethod
    def _init_persisted(cls, object_id, app):
        prefix, _ = object_id.split("-")  # TODO: util method
        object_cls = ObjectMeta.prefix_to_type[prefix]
        obj = Object.__new__(object_cls)
        Object.__init__(obj, app, object_id=object_id)
        return obj

    def set_object_id(self, object_id):
        object_cls = type(self)
        obj = Object.__new__(object_cls)
        Object.__init__(obj, self._app, object_id=object_id)
        return obj

    @property
    def object_id(self):
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

    def get_creating_message(self):
        return None

    def get_created_message(self):
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
