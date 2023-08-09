# Copyright Modal Labs 2022
import uuid
from datetime import date
from typing import Awaitable, Callable, ClassVar, Dict, Optional, Type, TypeVar

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal._types import typechecked
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import get_proto_oneof, retry_transient_errors

from ._resolver import Resolver
from .client import _Client
from .config import config
from .exception import InvalidError, NotFoundError, deprecation_error

H = TypeVar("H", bound="_Handle")

_BLOCKING_H = synchronize_api(H)


class _Handle:
    """mdmd:hidden The shared base class of any synced/distributed object in Modal.

    Examples of objects include Modal primitives like Images and Functions, as
    well as distributed data structures like Queues or Dicts.
    """

    _type_prefix: ClassVar[Optional[str]] = None
    _prefix_to_type: ClassVar[Dict[str, type]] = {}

    _object_id: str
    _client: _Client
    _is_hydrated: bool

    @classmethod
    def __init_subclass__(cls, type_prefix: Optional[str] = None):
        super().__init_subclass__()
        if type_prefix is not None:
            cls._type_prefix = type_prefix
            cls._prefix_to_type[type_prefix] = cls

    def __init__(self):
        raise Exception("__init__ disallowed, use proper classmethods")

    def _init(self):
        self._object_id = None
        self._client = None
        self._is_hydrated = False

    @classmethod
    def _new(cls: Type[H]) -> H:
        obj = _Handle.__new__(cls)
        obj._init()
        obj._initialize_from_empty()
        return obj

    def _initialize_from_empty(self):
        pass  # default implementation

    def _hydrate(self, object_id: str, client: _Client, metadata: Optional[Message]):
        self._object_id = object_id
        self._client = client
        if metadata:
            self._hydrate_metadata(metadata)
        self._is_hydrated = True

    def _hydrate_from_other(self, other: "_Handle"):
        self._hydrate(other.object_id, other._client, other._get_metadata())

    def is_hydrated(self) -> bool:
        """mdmd:hidden"""

        # A hydrated Handle is fully functional and linked to a live object in an app
        # To hydrate Handles, run an app using stub.run() or look up the object from a running app using <HandleClass>.lookup()
        return self._is_hydrated

    def _hydrate_metadata(self, metadata: Message):
        # override this is subclasses that need additional data (other than an object_id) for a functioning Handle
        pass

    def _get_metadata(self) -> Optional[Message]:
        # return the necessary metadata from this handle to be able to re-hydrate in another context if one is needed
        # used to provide a handle's handle_metadata for serializing/pickling a live handle
        # the object_id is already provided by other means
        return

    @classmethod
    def _new_hydrated(cls: Type[H], object_id: str, client: _Client, handle_metadata: Optional[Message]) -> H:
        """Similar to `_new` and `_hydrate` but does both at the same time."""

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
            if prefix not in cls._prefix_to_type:
                raise InvalidError(f"Object prefix {prefix} does not correspond to a type")
            object_cls = cls._prefix_to_type[prefix]

        # Instantiate object and return
        obj = object_cls._new()
        obj._hydrate(object_id, client, handle_metadata)
        return obj

    @classmethod
    async def from_id(cls: Type[H], object_id: str, client: Optional[_Client] = None) -> H:
        """Get an object of this type from a unique object id (retrieved from `obj.object_id`)"""
        # This is used in a few examples to construct FunctionCall objects
        # TODO(erikbern): this should probably be on the provider?
        if client is None:
            client = await _Client.from_env()
        app_lookup_object_response: api_pb2.AppLookupObjectResponse = await retry_transient_errors(
            client.stub.AppLookupObject, api_pb2.AppLookupObjectRequest(object_id=object_id)
        )

        handle_metadata = get_proto_oneof(app_lookup_object_response, "handle_metadata_oneof")
        return cls._new_hydrated(object_id, client, handle_metadata)

    @property
    def object_id(self) -> str:
        """A unique object id for this instance. Can be used to retrieve the object using `.from_id()`"""
        return self._object_id

    async def _hydrate_from_app(
        self: H,
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> H:
        """Returns a handle to a tagged object in a deployment on Modal."""
        if environment_name is None:
            environment_name = config.get("environment")

        if client is None:
            client = await _Client.from_env()
        request = api_pb2.AppLookupObjectRequest(
            app_name=app_name,
            object_tag=tag,
            namespace=namespace,
            object_entity=self._type_prefix,
            environment_name=environment_name,
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

        handle_metadata = get_proto_oneof(response, "handle_metadata_oneof")
        return self._hydrate(response.object_id, client, handle_metadata)


Handle = synchronize_api(_Handle)


P = TypeVar("P", bound="_Provider")

_BLOCKING_P = synchronize_api(P)


class _Provider:
    _type_prefix: ClassVar[Optional[str]] = None
    _prefix_to_type: ClassVar[Dict[str, type]] = {}

    _load: Optional[Callable[[Resolver, Optional[str], _Handle], Awaitable[None]]]
    _preload: Optional[Callable[[Resolver, Optional[str], _Handle], Awaitable[None]]]
    _handle: _Handle

    @classmethod
    def __init_subclass__(cls, type_prefix: Optional[str] = None):
        super().__init_subclass__()
        if type_prefix is not None:
            cls._type_prefix = type_prefix
            cls._prefix_to_type[type_prefix] = cls

    def __init__(self):
        raise Exception("__init__ disallowed, use proper classmethods")

    @classmethod
    def _get_handle_cls(cls) -> Type[_Handle]:
        return _Handle._prefix_to_type[cls._type_prefix]

    def _init(
        self,
        rep: str,
        load: Optional[Callable[[Resolver, Optional[str], _Handle], Awaitable[None]]] = None,
        is_persisted_ref: bool = False,
        preload: Optional[Callable[[Resolver, Optional[str], _Handle], Awaitable[None]]] = None,
        handle: Optional[_Handle] = None,
    ):
        self._local_uuid = str(uuid.uuid4())
        self._load = load
        self._preload = preload
        self._rep = rep
        self._is_persisted_ref = is_persisted_ref

        if handle is None:
            # Create an unhydrated handle
            handle_cls = self._get_handle_cls()
            handle = handle_cls._new()
        self._handle = handle

    def _init_from_other(self, other: "_Provider"):
        # Transient use case, see Dict, Queue, and SharedVolume
        self._init(other._rep, other._load, other._is_persisted_ref, other._preload)

    @classmethod
    def _from_loader(
        cls,
        load: Callable[[Resolver, Optional[str], _Handle], Awaitable[None]],
        rep: str,
        is_persisted_ref: bool = False,
        preload: Optional[Callable[[Resolver, Optional[str], _Handle], Awaitable[None]]] = None,
    ):
        # TODO(erikbern): flip the order of the two first arguments
        obj = _Provider.__new__(cls)
        obj._init(rep, load, is_persisted_ref, preload)
        return obj

    @classmethod
    def _new_hydrated(cls: Type[P], object_id: str, client: _Client, handle_metadata: Optional[Message]) -> P:
        handle = _Handle._new_hydrated(object_id, client, handle_metadata)
        provider_cls = cls._prefix_to_type[handle._type_prefix]
        obj = _Provider.__new__(provider_cls)
        rep = f"Provider({object_id})"  # TODO(erikbern): dumb
        obj._init(rep, handle=handle)
        return obj

    def __repr__(self):
        return self._rep

    @property
    def local_uuid(self):
        """mdmd:hidden"""
        return self._local_uuid

    @property
    def object_id(self):
        return self._handle.object_id

    def _get_metadata(self) -> Optional[Message]:
        return self._handle._get_metadata()

    def is_hydrated(self) -> bool:
        return self._handle.is_hydrated()

    @property
    def _client(self) -> _Client:
        return self._handle._client

    async def _deploy(
        self,
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> None:
        """
        Note 1: this uses the single-object app method, which we're planning to get rid of later
        Note 2: still considering this an "internal" method, but we'll make it "official" later
        """
        from .app import _App

        if environment_name is None:
            environment_name = config.get("environment")

        if client is None:
            client = await _Client.from_env()

        handle_cls = self._get_handle_cls()
        object_entity = handle_cls._type_prefix
        app = await _App._init_from_name(client, label, namespace, environment_name=environment_name)
        await app.create_one_object(self, environment_name)
        await app.deploy(label, namespace, object_entity)  # TODO(erikbern): not needed if the app already existed

    def persist(
        self, label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None
    ):
        """`Provider.persist` is deprecated for generic objects. See `NetworkFileSystem.persisted` or `Dict.persisted`."""
        # Note: this method is overridden in SharedVolume and Dict to print a warning
        deprecation_error(
            date(2023, 6, 30),
            self.persist.__doc__,
        )

    @typechecked
    def _persist(
        self, label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None
    ):
        if environment_name is None:
            environment_name = config.get("environment")

        async def _load_persisted(resolver: Resolver, existing_object_id: Optional[str], handle: _Handle):
            await self._deploy(label, namespace, resolver.client, environment_name=environment_name)
            handle._hydrate_from_other(self._handle)

        cls = type(self)
        rep = f"PersistedRef<{self}>({label})"
        return cls._from_loader(_load_persisted, rep, is_persisted_ref=True)

    @classmethod
    def from_name(
        cls: Type[P],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
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

        async def _load_remote(resolver: Resolver, existing_object_id: Optional[str], handle: _Handle):
            nonlocal environment_name
            if environment_name is None:
                # resolver always has an environment name, associated with the current app setup
                # fall back on that one if no explicit environment was set in the call itself
                environment_name = resolver.environment_name

            await handle._hydrate_from_app(
                app_name, tag, namespace, client=resolver.client, environment_name=environment_name
            )
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
        environment_name: Optional[str] = None,
    ) -> P:
        """
        General purpose method to retrieve Modal objects such as functions, network file systems, and secrets.
        ```python notest
        import modal
        square = modal.Function.lookup("my-shared-app", "square")
        assert square(3) == 9
        nfs = modal.NetworkFileSystem.lookup("my-nfs")
        for chunk in nfs.read_file("my_db_dump.csv"):
            ...
        ```
        """
        # TODO(erikbern): this code is very duplicated. Clean up once handles are gone.
        handle_cls = cls._get_handle_cls()
        handle: _Handle = handle_cls._new()
        await handle._hydrate_from_app(app_name, tag, namespace, client, environment_name=environment_name)
        obj = _Provider.__new__(cls)
        rep = f"Provider({app_name})"  # TODO(erikbern): dumb
        obj._init(rep, handle=handle)
        return obj

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


Provider = synchronize_api(_Provider, target_module=__name__)
