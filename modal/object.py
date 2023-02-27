# Copyright Modal Labs 2022
from datetime import date
import uuid
from typing import Awaitable, Callable, Generic, Optional, Type, TypeVar

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis

from ._object_meta import ObjectMeta
from ._resolver import Resolver
from .client import _Client
from .exception import InvalidError, NotFoundError, deprecation_warning

H = TypeVar("H", bound="Handle")


class Handle(metaclass=ObjectMeta):
    """mdmd:hidden The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.
    """

    _type_prefix: str

    def __init__(self):
        raise Exception("__init__ disallowed, use proper classmethods")

    def _init(self):
        self._client = None
        self._object_id = None

    @classmethod
    def _new(cls: Type[H]) -> H:
        obj = Handle.__new__(cls)
        obj._init()
        obj._initialize_from_empty()
        return obj

    def _initialize_handle(self, client: _Client, object_id: str):
        """mdmd:hidden"""
        self._client = client
        self._object_id = object_id

    def _initialize_from_empty(self):
        pass  # default implementation

    def _initialize_from_proto(self, proto: Message):
        pass  # default implementation

    @classmethod
    def _from_id(cls: Type[H], object_id: str, client: _Client, proto: Optional[Message]) -> H:
        if cls._type_prefix is not None:
            # This is called directly on a subclass, e.g. Secret.from_id
            if not object_id.startswith(cls._type_prefix):
                raise InvalidError(f"Object {object_id} does not start with {cls._type_prefix}")
            object_cls = cls
        else:
            # This is called on the base class, e.g. Handle.from_id
            parts = object_id.split("-")
            if len(parts) != 2:
                raise InvalidError(f"Object id {object_id} has no dash in it")
            prefix = parts[0]
            if prefix not in ObjectMeta.prefix_to_type:
                raise InvalidError(f"Object prefix {prefix} does not correspond to a type")
            object_cls = ObjectMeta.prefix_to_type[prefix]

        # Instantiate object and return
        obj = object_cls._new()
        obj._initialize_handle(client, object_id)
        obj._initialize_from_proto(proto)
        return obj

    @classmethod
    async def from_id(cls: Type[H], object_id: str, client: Optional[_Client] = None) -> H:
        # This is used in a few examples to construct FunctionCall objects
        # TODO(erikbern): doesn't use _initialize_from_proto - let's use AppLookupObjectRequest?
        # TODO(erikbern): this should probably be on the provider?
        if client is None:
            client = await _Client.from_env()
        return cls._from_id(object_id, client, None)

    @property
    def object_id(self) -> str:
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
            object_entity=cls._type_prefix,
        )
        try:
            response = await client.stub.AppLookupObject(request)
            if not response.object_id:
                # Legacy error message: remove soon
                raise NotFoundError(response.error_message)
        except GRPCError as exc:
            if exc.status == Status.NOT_FOUND:
                raise NotFoundError(exc.message)
            else:
                raise

        proto = response.function  # TODO: handle different object types
        handle: H = cls._from_id(response.object_id, client, proto)
        return handle


async def _lookup(
    app_name: str,
    tag: Optional[str] = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    client: Optional[_Client] = None,
) -> Handle:
    deprecation_warning(
        date(2023, 2, 11),
        "modal.lookup is deprecated. Use corresponding class methods instead," " e.g. modal.Secret.lookup, etc.",
    )
    return await Handle.from_app(app_name, tag, namespace, client)


lookup, aio_lookup = synchronize_apis(_lookup)

P = TypeVar("P", bound="Provider")


class Provider(Generic[H]):
    def _init(self, load: Callable[[Resolver, str], Awaitable[H]], rep: str):
        self._local_uuid = str(uuid.uuid4())
        self._load = load
        self._rep = rep

    def __init__(self, load: Callable[[Resolver, str], Awaitable[H]], rep: str):
        # TODO(erikbern): this is semi-deprecated - subclasses should use _from_loader
        self._init(load, rep)

    @classmethod
    def _from_loader(cls, load: Callable[[Resolver, str], Awaitable[H]], rep: str):
        obj = Handle.__new__(cls)
        obj._init(load, rep)
        return obj

    @classmethod
    def get_handle_cls(cls):
        (base,) = cls.__orig_bases__  # type: ignore
        (handle_cls,) = base.__args__
        return handle_cls

    def __repr__(self):
        return self._rep

    @property
    def local_uuid(self):
        return self._local_uuid

    async def _deploy(
        self, label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, client: Optional[_Client] = None
    ) -> H:
        """mdmd:hidden

        Note 1: this uses the single-object app method, which we're planning to get rid of later
        Note 2: still considering this an "internal" method, but we'll make it "official" later
        """
        from .stub import _Stub

        if client is None:
            client = await _Client.from_env()

        handle_cls = self.get_handle_cls()
        object_entity = handle_cls._type_prefix
        _stub = _Stub(label, _object=self)
        await _stub.deploy(namespace=namespace, client=client, object_entity=object_entity, show_progress=False)
        handle: H = await handle_cls.from_app(label, namespace=namespace, client=client)
        return handle

    def persist(self, label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE):
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

        async def _load_persisted(resolver: Resolver, existing_object_id: str) -> H:
            return await self._deploy(label, namespace, resolver.client)

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

        async def _load_remote(resolver: Resolver, existing_object_id: str) -> H:
            handle_cls = cls.get_handle_cls()
            handle: H = await handle_cls.from_app(app_name, tag, namespace, client=resolver.client)
            return handle

        # Create a class of type cls, but use the base constructor
        # TODO(erikbern): No Provider subclass should override __init__
        obj = cls.__new__(cls)
        rep = f"Ref({app_name})"
        Provider.__init__(obj, _load_remote, rep)
        return obj

    @classmethod
    async def lookup(
        cls: Type[P],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
    ) -> H:
        """
        General purpose method to retrieve Modal objects such as
        functions, shared volumes, and secrets.
        ```python notest
        import modal
        square = modal.Function.lookup("my-shared-app", "square")
        assert square(3) == 9
        vol = modal.SharedVolume.lookup("my-shared-volume")
        for chunk in vol.read_file("my_db_dump.csv"):
            ...
        ```
        """
        handle_cls = cls.get_handle_cls()
        handle: H = await handle_cls.from_app(app_name, tag, namespace, client)
        return handle

    @classmethod
    async def _exists(
        cls: Type[P],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
    ) -> bool:
        """mdmd:hidden

        Internal for now - will make this "public" later.
        """
        if client is None:
            client = await _Client.from_env()
        handle_cls = cls.get_handle_cls()
        request = api_pb2.AppLookupObjectRequest(
            app_name=app_name,
            object_tag=tag,
            namespace=namespace,
            object_entity=handle_cls._type_prefix,
        )
        try:
            response = await client.stub.AppLookupObject(request)
            return bool(response.object_id)  # old code path - change to `return True` shortly
        except GRPCError as exc:
            if exc.status == Status.NOT_FOUND:
                return False
            else:
                raise
