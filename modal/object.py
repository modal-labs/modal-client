# Copyright Modal Labs 2022
from datetime import date
import uuid
from typing import Awaitable, Callable, Generic, Optional, Type, TypeVar

from google.protobuf.message import Message
from grpclib import GRPCError, Status
from modal._types import typechecked

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors, get_proto_oneof

from ._object_meta import ObjectMeta
from ._resolver import Resolver
from .client import _Client
from .exception import InvalidError, NotFoundError, deprecation_error

H = TypeVar("H", bound="_Handle")

_BLOCKING_H, _ASYNC_H = synchronize_apis(H)


class _Handle(metaclass=ObjectMeta):
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
        self._is_hydrated = False

    @classmethod
    def _new(cls: Type[H]) -> H:
        obj = _Handle.__new__(cls)
        obj._init()
        obj._initialize_from_empty()
        return obj

    def _initialize_from_empty(self):
        pass  # default implementation

    def _hydrate(self, client: _Client, object_id: str, handle_metadata: Optional[Message]):
        self._client = client
        self._object_id = object_id
        if handle_metadata:
            self._hydrate_metadata(handle_metadata)
        self._is_hydrated = True

    def is_hydrated(self) -> bool:
        # A hydrated Handle is fully functional and linked to a live object in an app
        # To hydrate Handles, run an app using stub.run() or look up the object from a running app using <HandleClass>.lookup()
        return self._is_hydrated

    def _hydrate_metadata(self, handle_metadata: Message):
        # override this is subclasses that need additional data (other than an object_id) for a functioning Handle
        pass

    def _get_handle_metadata(self) -> Optional[Message]:
        # return the necessary metadata from this handle to be able to re-hydrate in another context if one is needed
        # used to provide a handle's handle_metadata for serializing/pickling a live handle
        # the object_id is already provided by other means
        return None

    @classmethod
    def _from_id(cls: Type[H], object_id: str, client: _Client, handle_metadata: Optional[Message]) -> H:
        if cls._type_prefix is not None:
            # This is called directly on a subclass, e.g. Secret.from_id
            if not object_id.startswith(cls._type_prefix + "-"):
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
        obj._hydrate(client, object_id, handle_metadata)
        return obj

    @classmethod
    async def from_id(cls: Type[H], object_id: str, client: Optional[_Client] = None) -> H:
        """Get an object of this type from a unique object id (retrieved from `obj.object_id`)"""
        # This is used in a few examples to construct FunctionCall objects
        # TODO(erikbern): this should probably be on the provider?
        if client is None:
            client = await _Client.from_env()
        app_lookup_object_response: api_pb2.AppLookupObjectResponse = await client.stub.AppLookupObject(
            api_pb2.AppLookupObjectRequest(object_id=object_id)
        )

        handle_metadata = get_proto_oneof(app_lookup_object_response, "handle_metadata_oneof")
        return cls._from_id(object_id, client, handle_metadata)

    @property
    def object_id(self) -> str:
        """A unique object id for this instance. Can be used to retrieve the object using `.from_id()`"""
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
            response = await retry_transient_errors(client.stub.AppLookupObject, request)
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


Handle, AioHandle = synchronize_apis(_Handle)


@typechecked
async def _lookup(
    app_name: str,
    tag: Optional[str] = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    client: Optional[_Client] = None,
):
    """Deprecated. Use corresponding class methods instead," " e.g. modal.Secret.lookup, etc."""
    deprecation_error(
        date(2023, 2, 11),
        _lookup.__doc__,
    )


lookup, aio_lookup = synchronize_apis(_lookup)

P = TypeVar("P", bound="_Provider")

_BLOCKING_P, _ASYNC_P = synchronize_apis(P)


class _Provider(Generic[H]):
    _load: Callable[[Resolver, Optional[str]], Awaitable[H]]
    _preload: Optional[Callable[[Resolver, Optional[str]], Awaitable[H]]]

    def _init(
        self,
        load: Callable[[Resolver, Optional[str]], Awaitable[H]],
        rep: str,
        is_persisted_ref: bool = False,
        preload: Optional[Callable[[Resolver, Optional[str]], Awaitable[H]]] = None,
    ):
        self._local_uuid = str(uuid.uuid4())
        self._load = load
        self._preload = preload
        self._rep = rep
        self._is_persisted_ref = is_persisted_ref

    def __init__(
        self,
        load: Callable[[Resolver, Optional[str]], Awaitable[H]],
        rep: str,
        is_persisted_ref: bool = False,
        preload: Optional[Callable[[Resolver, Optional[str]], Awaitable[H]]] = None,
    ):
        # TODO(erikbern): this is semi-deprecated - subclasses should use _from_loader
        self._init(load, rep, is_persisted_ref, preload=preload)

    def _init_from_other(self, other: "_Provider"):
        # Transient use case, see Secret.__inint__
        self._init(other._load, other._rep, other._is_persisted_ref, other._preload)

    @classmethod
    def _from_loader(
        cls,
        load: Callable[[Resolver, Optional[str]], Awaitable[H]],
        rep: str,
        is_persisted_ref: bool = False,
        preload: Optional[Callable[[Resolver, Optional[str]], Awaitable[H]]] = None,
    ):
        obj = _Handle.__new__(cls)
        obj._init(load, rep, is_persisted_ref, preload)
        return obj

    @classmethod
    def _get_handle_cls(cls) -> Type[H]:
        (base,) = cls.__orig_bases__  # type: ignore
        (handle_cls,) = base.__args__
        return handle_cls

    def __repr__(self):
        return self._rep

    @property
    def local_uuid(self):
        """mdmd:hidden"""
        return self._local_uuid

    async def _deploy(
        self,
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
    ) -> H:
        """
        Note 1: this uses the single-object app method, which we're planning to get rid of later
        Note 2: still considering this an "internal" method, but we'll make it "official" later
        """
        from .app import _App

        if client is None:
            client = await _Client.from_env()

        handle_cls = self._get_handle_cls()
        object_entity = handle_cls._type_prefix
        app = await _App._init_from_name(client, label, namespace)
        handle = await app.create_one_object(self)
        await app.deploy(label, namespace, object_entity)  # TODO(erikbern): not needed if the app already existed
        return handle

    @typechecked
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

        async def _load_persisted(resolver: Resolver, existing_object_id: Optional[str]) -> H:
            return await self._deploy(label, namespace, resolver.client)

        cls = type(self)
        rep = f"PersistedRef<{self}>({label})"
        return cls._from_loader(_load_persisted, rep, is_persisted_ref=True)

    @classmethod
    def from_name(
        cls: Type[P],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
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

        async def _load_remote(resolver: Resolver, existing_object_id: Optional[str]) -> H:
            handle_cls = cls._get_handle_cls()
            handle: H = await handle_cls.from_app(app_name, tag, namespace, client=resolver.client)
            return handle

        rep = f"Ref({app_name})"
        return cls._from_loader(_load_remote, rep)

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
        handle_cls = cls._get_handle_cls()
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
        """
        Internal for now - will make this "public" later.
        """
        if client is None:
            client = await _Client.from_env()
        handle_cls = cls._get_handle_cls()
        request = api_pb2.AppLookupObjectRequest(
            app_name=app_name,
            object_tag=tag,
            namespace=namespace,
            object_entity=handle_cls._type_prefix,
        )
        try:
            response = await retry_transient_errors(client.stub.AppLookupObject, request)
            return bool(response.object_id)  # old code path - change to `return True` shortly
        except GRPCError as exc:
            if exc.status == Status.NOT_FOUND:
                return False
            else:
                raise


# Dumb but needed becauase it's in the hierarchy
synchronize_apis(Generic, __name__)  # erases base Generic type...
Provider, AioProvider = synchronize_apis(_Provider, target_module=__name__)
