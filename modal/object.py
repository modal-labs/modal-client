# Copyright Modal Labs 2022
import uuid
from typing import (
    Awaitable,
    Callable,
    Generic,
    Optional,
    Type,
    TypeVar,
    cast,
)

from google.protobuf.message import Message

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._resolver import Resolver
from ._object_meta import ObjectMeta
from .client import _Client
from .exception import InvalidError, NotFoundError

H = TypeVar("H", bound="Handle")


class Handle(metaclass=ObjectMeta):
    """mdmd:hidden The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.
    """

    def __init__(self, client=None, object_id=None):
        """mdmd:hidden"""
        self._client = client
        self._object_id = object_id

    def _initialize_from_proto(self, proto: Message):
        pass  # default implementation

    @staticmethod
    def _from_id(object_id: str, client: _Client, proto: Optional[Message]):
        parts = object_id.split("-")
        if len(parts) != 2:
            raise InvalidError(f"Object id {object_id} has no dash in it")
        prefix = parts[0]
        if prefix not in ObjectMeta.prefix_to_type:
            raise InvalidError(f"Object prefix {prefix} does not correspond to a type")
        object_cls = ObjectMeta.prefix_to_type[prefix]
        obj = Handle.__new__(object_cls)
        Handle.__init__(obj, client, object_id=object_id)
        if proto is not None:
            obj._initialize_from_proto(proto)
        return obj

    @property
    def object_id(self):
        return self._object_id

    @classmethod
    async def from_app(
        cls: Type[H],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
        client: Optional[_Client] = None,
    ) -> H:
        """Returns a handle to a tagged object in a deployment on Modal."""
        if client is None:
            client = await _Client.from_env()
        request = api_pb2.AppLookupObjectRequest(
            app_name=app_name,
            object_tag=tag,
            namespace=namespace,
        )
        response = await client.stub.AppLookupObject(request)
        if not response.object_id:
            raise NotFoundError(response.error_message)
        proto = response.function  # TODO: handle different object types
        return Handle._from_id(response.object_id, client, proto)


async def _lookup(
    app_name: str,
    tag: Optional[str] = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
    client: Optional[_Client] = None,
) -> Handle:
    """
    General purpose method to retrieve Modal objects such as
    functions, shared volumes, and secrets.

    ```python notest
    import modal

    square = modal.lookup("my-shared-app", "square")
    assert square(3) == 9

    vol = modal.lookup("my-shared-volume")
    for chunk in vol.read_file("my_db_dump.csv"):
        ...
    ```
    """
    return await Handle.from_app(app_name, tag, namespace, client)


lookup, aio_lookup = synchronize_apis(_lookup)

P = TypeVar("P", bound="Provider")


class Provider(Generic[H]):
    def __init__(self, load: Callable[[Resolver], Awaitable[None]]):
        self._local_uuid = str(uuid.uuid4())
        self._load = load

    @property
    def local_uuid(self):
        return self._local_uuid

    def persist(self, label: str):
        """Deploy a Modal app containing this object. This object can then be imported from other apps using
        the returned reference, or by calling `modal.SharedVolume.from_name(label)` (or the equivalent method
        on respective class).

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

    @classmethod
    def from_name(
        cls: Type[P], app_name: str, tag: Optional[str] = None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT
    ) -> P:
        """Returns a reference to an Modal object of any type

        Useful for referring to already created/deployed objects, e.g., Secrets

        ```python
        import modal

        stub = modal.Stub()

        @stub.function(secret=modal.Secret.from_name("my-secret-name"))
        def some_function():
            pass
        ```
        """
        provider: RemoteRef = RemoteRef(app_name, tag, namespace)
        # TODO(erikbern): this returns an object that looks like a P during static analysis,
        # but is actually a RemoteRef during runtime. This seems pretty confusing and bad:
        # we should return an object that's always P.
        return cast(P, provider)


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
        super().__init__(self._load)

    def __repr__(self):
        return f"Ref({self.app_name})"

    async def _load(
        self,
        resolver: Resolver,
    ) -> H:
        handle = await Handle.from_app(self.app_name, self.tag, self.namespace, resolver.client)
        return cast(H, handle)


class PersistedRef(Ref[H]):
    def __init__(self, app_name: str, definition: H):
        self.app_name = app_name
        self.definition = definition
        super().__init__(self._load)

    def __repr__(self):
        return f"PersistedRef<{self.definition}>({self.app_name})"

    async def _load(
        self,
        resolver: Resolver,
    ) -> H:
        from .stub import _Stub

        _stub = _Stub(self.app_name, _object=self.definition)
        await _stub.deploy(client=resolver.client)
        handle = await Handle.from_app(self.app_name, client=resolver.client)
        return cast(H, handle)
