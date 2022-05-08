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

    def __init__(self, app, label=None, object_id=None):
        if app is None:
            raise InvalidError(f"Object {self} created without an app")
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
    def from_id(cls, object_id, app):
        parts = object_id.split("-")
        if len(parts) != 2:
            raise InvalidError(f"Object id {object_id} has no dash in it")
        prefix = parts[0]
        if prefix not in ObjectMeta.prefix_to_type:
            raise InvalidError(f"Object prefix {prefix} does not correspond to a type")
        object_cls = ObjectMeta.prefix_to_type[prefix]
        obj = Object.__new__(object_cls)
        Object.__init__(obj, app, object_id=object_id)
        return obj

    @property
    def object_id(self):
        return self._object_id

    @property
    def label(self):
        return self._label

    def get_creating_message(self) -> Optional[str]:
        return None

    def get_created_message(self) -> Optional[str]:
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
