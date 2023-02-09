# Copyright Modal Labs 2022
import uuid
from typing import Awaitable, Callable, Generic, Optional, Type, TypeVar, cast

from google.protobuf.message import Message

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._object_meta import ObjectMeta
from ._resolver import Resolver
from .client import _Client
from .exception import InvalidError, NotFoundError

H = TypeVar("H", bound="Handle")


class Handle(metaclass=ObjectMeta):
    """mdmd:hidden The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.
    """

    def __init__(self):
        raise Exception("__init__ disallowed, use proper classmethods")

    def _init(self):
        self._client = None
        self._object_id = None

    @classmethod
    def _new(cls):
        obj = Handle.__new__(cls)
        obj._init()
        obj._initialize_from_proto(None)
        return obj

    def _initialize_handle(self, client: _Client, object_id: str):
        """mdmd:hidden"""
        self._client = client
        self._object_id = object_id

    def _initialize_from_proto(self, proto: Optional[Message]):
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
        obj = object_cls._new()
        obj._initialize_handle(client, object_id)
        obj._initialize_from_proto(proto)
        return obj

    @classmethod
    async def from_id(cls, object_id: str, client: Optional[_Client] = None):
        # This is used in a few examples to construct FunctionCall objects
        # TODO(erikbern): doesn't use _initialize_from_proto - let's use AppLookupObjectRequest?
        # TODO(erikbern): this should probably be on the provider?
        if client is None:
            client = await _Client.from_env()
        return cls._from_id(object_id, client, None)

    @property
    def object_id(self):
        return self._object_id

    @classmethod
    async def from_app(
        cls: Type[H],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
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
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    client: Optional[_Client] = None,
) -> Handle:
    """
    General purpose method to retrieve Modal objects such as
    functions, shared volumes, and secrets.

    ```python notest
    import modal

    square = modal.lookup("my-shared-app", "square")
    assert square.call(3) == 9

    vol = modal.lookup("my-shared-volume")
    for chunk in vol.read_file("my_db_dump.csv"):
        ...
    ```
    """
    return await Handle.from_app(app_name, tag, namespace, client)


lookup, aio_lookup = synchronize_apis(_lookup)

P = TypeVar("P", bound="Provider")


class Provider(Generic[H]):
    def _init(self, load, rep):
        self._local_uuid = str(uuid.uuid4())
        self._load = load
        self._rep = rep

    def __init__(self, load: Callable[[Resolver], Awaitable[H]], rep: str):
        # TODO(erikbern): this is semi-deprecated - subclasses should use _from_loader
        self._init(load, rep)

    @classmethod
    def _from_loader(cls, load: Callable[[Resolver], Awaitable[H]], rep: str):
        obj = Handle.__new__(cls)
        obj._init(load, rep)
        return obj

    def __repr__(self):
        return self._rep

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

        async def _load_persisted(resolver: Resolver) -> H:
            from .stub import _Stub

            _stub = _Stub(label, _object=self)
            await _stub.deploy(client=resolver.client)
            handle = await Handle.from_app(label, client=resolver.client)
            return cast(H, handle)

        # Create a class of type cls, but use the base constructor
        cls = type(self)
        obj = cls.__new__(cls)
        rep = f"PersistedRef<{self}>({label})"
        Provider.__init__(obj, _load_persisted, rep)
        return obj

    @classmethod
    def from_name(
        cls: Type[P], app_name: str, tag: Optional[str] = None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE
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

        async def _load_remote(resolver: Resolver) -> H:
            handle = await Handle.from_app(app_name, tag, namespace, resolver.client)
            return cast(H, handle)

        # Create a class of type cls, but use the base constructor
        # TODO(erikbern): No Provider subclass should override __init__
        obj = cls.__new__(cls)
        rep = f"Ref({app_name})"
        Provider.__init__(obj, _load_remote, rep)
        return obj
