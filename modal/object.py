import uuid
from typing import Optional

from modal_proto import api_pb2

from ._object_meta import ObjectMeta
from .client import _Client
from .exception import InvalidError


class Object(metaclass=ObjectMeta):
    """The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.
    """

    def __init__(self, client=None, object_id=None):
        self._client = client
        self._object_id = object_id
        self._local_uuid = str(uuid.uuid4())

    async def load(
        self,
        client: _Client,
        app_id: str,
        existing_object_id: Optional[str] = None,
    ):
        raise NotImplementedError(f"Object factory of class {type(self)} has no load method")

    @classmethod
    def from_id(cls, object_id, client):
        parts = object_id.split("-")
        if len(parts) != 2:
            raise InvalidError(f"Object id {object_id} has no dash in it")
        prefix = parts[0]
        if prefix not in ObjectMeta.prefix_to_type:
            raise InvalidError(f"Object prefix {prefix} does not correspond to a type")
        object_cls = ObjectMeta.prefix_to_type[prefix]
        obj = Object.__new__(object_cls)
        Object.__init__(obj, client, object_id=object_id)
        return obj

    async def create(self, running_app=None):
        # TOOD: should we just get rid of this one in favor of running_app.create(obj) ?
        from .running_app import _container_app, _RunningApp  # avoid circular import

        if running_app is None:
            running_app = _container_app
            if running_app is None:
                raise InvalidError(".create must be passed the app explicitly if not running in a container")
        assert isinstance(running_app, _RunningApp)
        object_id = await running_app.load(self)
        return Object.from_id(object_id, running_app.client)

    @property
    def object_id(self):
        return self._object_id

    @property
    def local_uuid(self):
        return self._local_uuid

    def get_creating_message(self) -> Optional[str]:
        return None

    def get_created_message(self) -> Optional[str]:
        return None

    @classmethod
    def include(cls, app, app_name, object_label=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
        """Use an object published with `modal.App.deploy`"""
        raise InvalidError("The `Object.include` method is gone. Use `modal.ref` instead!")

    async def persist(self, label: str):
        """Deploy a Modal app containing this object. This object can then be imported from other apps using
        the returned reference, or by calling `modal.ref(label)`.

        **Example Usage**

        ```python
        import modal

        volume = modal.SharedVolume().persist("my-volume")

        app = modal.App()

        # Volume refers to the same object, even across instances of `app`.
        @app.function(shared_volumes={"/vol": volume})
        def f():
            pass
        ```

        """
        return Ref(label, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT, definition=self)

    def __await__(self):
        """Make objects awaitable from the load() method."""
        return (yield self)


class Ref(Object):
    def __init__(
        self,
        app_name: Optional[str] = None,  # If it's none then it's the same app
        tag: Optional[str] = None,
        namespace: Optional[int] = None,  # api_pb2.DEPLOYMENT_NAMESPACE
        definition: Optional[Object] = None,  # Object definition to deploy to this ref.
    ):
        self.app_name = app_name
        self.tag = tag
        self.namespace = namespace
        self.definition = definition
        super().__init__()


def ref(app_name: Optional[str], tag: Optional[str] = None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
    # TODO(erikbern): we should probably get rid of this function since it's just a dumb wrapper
    return Ref(app_name, tag, namespace)
