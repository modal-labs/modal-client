import uuid
from typing import Awaitable, Callable, Generic, Optional, Type, TypeVar

from modal_proto import api_pb2

from ._object_meta import ObjectMeta
from .client import _Client
from .exception import InvalidError, NotFoundError

T = TypeVar("T")


class Handle(metaclass=ObjectMeta):
    """mdmd:hidden The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.
    """

    def __init__(self, client=None, object_id=None):
        self._client = client
        self._object_id = object_id

    @staticmethod
    def _from_id(object_id, client):
        parts = object_id.split("-")
        if len(parts) != 2:
            raise InvalidError(f"Object id {object_id} has no dash in it")
        prefix = parts[0]
        if prefix not in ObjectMeta.prefix_to_type:
            raise InvalidError(f"Object prefix {prefix} does not correspond to a type")
        object_cls = ObjectMeta.prefix_to_type[prefix]
        obj = Handle.__new__(object_cls)
        Handle.__init__(obj, client, object_id=object_id)
        return obj

    @classmethod
    async def from_id(cls: Type[T], object_id: str) -> T:
        client = await _Client.from_env()
        return Handle._from_id(object_id, client)

    @property
    def object_id(self):
        return self._object_id


H = TypeVar("H")


class Provider(Generic[H]):
    def __init__(self):
        self._local_uuid = str(uuid.uuid4())

    @property
    def local_uuid(self):
        return self._local_uuid

    async def persist(self, label: str):
        """Deploy a Modal app containing this object. This object can then be imported from other apps using
        the returned reference, or by calling `modal.ref(label)`.

        **Example Usage**

        ```python
        import modal

        volume = modal.SharedVolume().persist("my-volume")

        stub = modal.Stub()

        # Volume refers to the same object, even across instances of `stub`.
        @stub.function(shared_volumes={"/vol": volume})
        def f():
            pass
        ```

        """
        return PersistedRef(label, definition=self)

    async def _load(
        self,
        client: _Client,
        app_id: str,
        loader: Callable[["Provider"], Awaitable[str]],
        message_callback: Callable[[str], None],
        existing_object_id: Optional[str] = None,
    ) -> H:
        raise NotImplementedError(f"Object factory of class {type(self)} has no load method")


class Ref(Provider[H]):
    pass


class RemoteRef(Ref[H]):
    def __init__(
        self,
        app_name: str,
        tag: Optional[str] = None,
        namespace: Optional[int] = None,  # api_pb2.DEPLOYMENT_NAMESPACE
    ):
        self.app_name = app_name
        self.tag = tag
        self.namespace = namespace
        super().__init__()


class LocalRef(Ref[H]):
    def __init__(self, tag: str):
        self.tag = tag
        super().__init__()

    def __call__(self, *args, **kwargs):
        raise NotFoundError(f"Stub has no function named {self.tag}.")


class PersistedRef(Ref[H]):
    def __init__(self, app_name: str, definition: H):
        self.app_name = app_name
        self.definition = definition
        super().__init__()


def ref(app_name: Optional[str], tag: Optional[str] = None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT) -> Ref:
    """Returns a reference to an Modal object any type

    Useful for referring to already created/deployed objects, e.g., Secrets

    ```python
    import modal

    stub = modal.Stub()

    @stub.function(secret=modal.ref("my-secret-name"))
    def some_function():
        pass
    ```
    """
    # TODO(erikbern): we should probably get rid of this function since it's just a dumb wrapper
    return RemoteRef(app_name, tag, namespace)
