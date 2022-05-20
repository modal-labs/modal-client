from typing import Optional

from modal_proto import api_pb2

from ._object_meta import ObjectMeta
from .app_singleton import _container_app
from .exception import InvalidError


class Object(metaclass=ObjectMeta):
    """The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.
    """

    def __init__(self, running_app=None, object_id=None):
        self._running_app = running_app
        self._object_id = object_id

    async def load(self, running_app, existing_object_id):
        raise NotImplementedError(f"Object factory of class {type(self)} has no load method")

    @classmethod
    def from_id(cls, object_id, running_app):
        parts = object_id.split("-")
        if len(parts) != 2:
            raise InvalidError(f"Object id {object_id} has no dash in it")
        prefix = parts[0]
        if prefix not in ObjectMeta.prefix_to_type:
            raise InvalidError(f"Object prefix {prefix} does not correspond to a type")
        object_cls = ObjectMeta.prefix_to_type[prefix]
        obj = Object.__new__(object_cls)
        Object.__init__(obj, running_app, object_id=object_id)
        return obj

    async def create(self, running_app=None):
        if running_app is None:
            running_app = _container_app
            if running_app is None:
                raise InvalidError(".create must be passed the app explicitly if not running in a container")
        object_id = await self.load(running_app, None)
        return Object.from_id(object_id, running_app)

    @property
    def object_id(self):
        return self._object_id

    @property
    def running_app(self):
        return self._running_app

    def get_creating_message(self) -> Optional[str]:
        return None

    def get_created_message(self) -> Optional[str]:
        return None

    @classmethod
    def include(cls, app, app_name, object_label=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
        """Use an object published with `modal.App.deploy`"""
        raise InvalidError("The `Object.include` method is gone. Use `modal.ref` instead!")


class Ref(Object):
    def __init__(
        self,
        app_name: Optional[str] = None,  # If it's none then it's the same app
        tag: Optional[str] = None,
        namespace: Optional[int] = None,  # api_pb2.DEPLOYMENT_NAMESPACE
    ):
        self.app_name = app_name
        self.tag = tag
        self.namespace = namespace
        super().__init__()


def ref(app_name: Optional[str], tag: Optional[str] = None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
    # TODO(erikbern): we should probably get rid of this function since it's just a dumb wrapper
    return Ref(app_name, tag, namespace)
