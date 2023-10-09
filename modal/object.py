# Copyright Modal Labs 2022
import uuid
from datetime import date
from functools import wraps
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
from .exception import ExecutionError, InvalidError, NotFoundError, deprecation_error

O = TypeVar("O", bound="_Object")

_BLOCKING_O = synchronize_api(O)


class _Object:
    _type_prefix: ClassVar[Optional[str]] = None
    _prefix_to_type: ClassVar[Dict[str, type]] = {}

    # For constructors
    _load: Optional[Callable[[O, Resolver, Optional[str]], Awaitable[None]]]
    _preload: Optional[Callable[[O, Resolver, Optional[str]], Awaitable[None]]]

    # For hydrated objects
    _object_id: str
    _client: _Client
    _is_hydrated: bool

    @classmethod
    def __init_subclass__(cls, type_prefix: Optional[str] = None):
        super().__init_subclass__()
        if type_prefix is not None:
            cls._type_prefix = type_prefix
            cls._prefix_to_type[type_prefix] = cls

    def __init__(self, *args, **kwargs):
        raise InvalidError(f"Class {type(self).__name__} has no constructor. Use class constructor methods instead.")

    def _init(
        self,
        rep: str,
        load: Optional[Callable[[O, Resolver, Optional[str]], Awaitable[None]]] = None,
        is_persisted_ref: bool = False,
        preload: Optional[Callable[[O, Resolver, Optional[str]], Awaitable[None]]] = None,
        hydrate_lazily: bool = False,
    ):
        self._local_uuid = str(uuid.uuid4())
        self._load = load
        self._preload = preload
        self._rep = rep
        self._is_persisted_ref = is_persisted_ref
        self._hydrate_lazily = hydrate_lazily

        self._object_id = None
        self._client = None
        self._is_hydrated = False

        self._initialize_from_empty()

    def _unhydrate(self):
        self._object_id = None
        self._client = None
        self._is_hydrated = False

    def _initialize_from_empty(self):
        # default implementation, can be overriden in subclasses
        pass

    def _hydrate(self, object_id: str, client: _Client, metadata: Optional[Message]):
        self._object_id = object_id
        self._client = client
        self._hydrate_metadata(metadata)
        self._is_hydrated = True

    def _hydrate_metadata(self, metadata: Optional[Message]):
        # override this is subclasses that need additional data (other than an object_id) for a functioning Handle
        pass

    def _get_metadata(self) -> Optional[Message]:
        # return the necessary metadata from this handle to be able to re-hydrate in another context if one is needed
        # used to provide a handle's handle_metadata for serializing/pickling a live handle
        # the object_id is already provided by other means
        return

    def _init_from_other(self, other: O):
        # Transient use case, see Dict, Queue, and SharedVolume
        self._init(other._rep, other._load, other._is_persisted_ref, other._preload)

    @classmethod
    def _from_loader(
        cls,
        load: Callable[[O, Resolver, Optional[str]], Awaitable[None]],
        rep: str,
        is_persisted_ref: bool = False,
        preload: Optional[Callable[[O, Resolver, Optional[str]], Awaitable[None]]] = None,
        hydrate_lazily: bool = False,
    ):
        # TODO(erikbern): flip the order of the two first arguments
        obj = _Object.__new__(cls)
        obj._init(rep, load, is_persisted_ref, preload, hydrate_lazily)
        return obj

    @classmethod
    def _new_hydrated(cls: Type[O], object_id: str, client: _Client, handle_metadata: Optional[Message]) -> O:
        if cls._type_prefix is not None:
            # This is called directly on a subclass, e.g. Secret.from_id
            if not object_id.startswith(cls._type_prefix + "-"):
                raise InvalidError(f"Object {object_id} does not start with {cls._type_prefix}")
            prefix = cls._type_prefix
        else:
            # This is called on the base class, e.g. Handle.from_id
            parts = object_id.split("-")
            if len(parts) != 2:
                raise InvalidError(f"Object id {object_id} has no dash in it")
            prefix = parts[0]
            if prefix not in cls._prefix_to_type:
                raise InvalidError(f"Object prefix {prefix} does not correspond to a type")

        # Instantiate provider
        obj_cls = cls._prefix_to_type[prefix]
        obj = _Object.__new__(obj_cls)
        rep = f"Object({object_id})"  # TODO(erikbern): dumb
        obj._init(rep)
        obj._hydrate(object_id, client, handle_metadata)

        return obj

    @classmethod
    async def from_id(cls: Type[O], object_id: str, client: Optional[_Client] = None) -> O:
        """Retrieve an object from its unique ID (accessed through `obj.object_id`)."""
        # This is used in a few examples to construct FunctionCall objects
        if client is None:
            client = await _Client.from_env()
        response: api_pb2.AppLookupObjectResponse = await retry_transient_errors(
            client.stub.AppLookupObject, api_pb2.AppLookupObjectRequest(object_id=object_id)
        )

        handle_metadata = get_proto_oneof(response.object, "handle_metadata_oneof")
        return cls._new_hydrated(object_id, client, handle_metadata)

    async def _hydrate_from_app(
        self,
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ):
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
            if not response.object.object_id:
                # Legacy error message: remove soon
                raise NotFoundError(response.error_message)
        except GRPCError as exc:
            if exc.status == Status.NOT_FOUND:
                raise NotFoundError(exc.message)
            else:
                raise

        handle_metadata = get_proto_oneof(response.object, "handle_metadata_oneof")
        return self._hydrate(response.object.object_id, client, handle_metadata)

    def _hydrate_from_other(self, other: O):
        self._hydrate(other._object_id, other._client, other._get_metadata())

    def __repr__(self):
        return self._rep

    @property
    def local_uuid(self):
        """mdmd:hidden"""
        return self._local_uuid

    @property
    def object_id(self):
        """mdmd:hidden"""
        return self._object_id

    def is_hydrated(self) -> bool:
        """mdmd:hidden"""
        return self._is_hydrated

    async def _try_hydrate(self) -> bool:
        if not self._is_hydrated and self._hydrate_lazily:
            resolver = Resolver()
            await resolver.load(self)

        return self._is_hydrated

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
        from .app import _LocalApp

        if environment_name is None:
            environment_name = config.get("environment")

        if client is None:
            client = await _Client.from_env()

        await _LocalApp._deploy_single_object(self, self._type_prefix, client, label, namespace, environment_name)

    def persist(
        self, label: str, namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, environment_name: Optional[str] = None
    ):
        """`Object.persist` is deprecated for generic objects. See `NetworkFileSystem.persisted` or `Dict.persisted`."""
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

        async def _load_persisted(obj: _Object, resolver: Resolver, existing_object_id: Optional[str]):
            await self._deploy(label, namespace, resolver.client, environment_name=environment_name)
            obj._hydrate_from_other(self)

        cls = type(self)
        rep = f"PersistedRef<{self}>({label})"
        return cls._from_loader(_load_persisted, rep, is_persisted_ref=True)

    @classmethod
    def from_name(
        cls: Type[O],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
    ) -> O:
        """Retrieve an object with a given name and tag.

        Useful for referencing secrets, as well as calling a function from a different app.
        Use this when attaching the object to a stub or function.

        **Examples**

        ```python notest
        # Retrieve a secret
        stub.my_secret = Secret.from_name("my-secret")

        # Retrieve a function from a different app
        stub.other_function = Function.from_name("other-app", "function")

        # Retrieve a persisted Volume, Queue, or Dict
        stub.my_volume = Volume.from_name("my-volume")
        stub.my_queue = Queue.from_name("my-queue")
        stub.my_dict = Dict.from_name("my-dict")
        ```
        """

        async def _load_remote(obj: _Object, resolver: Resolver, existing_object_id: Optional[str]):
            nonlocal environment_name
            if environment_name is None:
                # resolver always has an environment name, associated with the current app setup
                # fall back on that one if no explicit environment was set in the call itself
                environment_name = resolver.environment_name

            await obj._hydrate_from_app(
                app_name, tag, namespace, client=resolver.client, environment_name=environment_name
            )

        rep = f"Ref({app_name})"
        return cls._from_loader(_load_remote, rep)

    @classmethod
    async def lookup(
        cls: Type[O],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> O:
        """Lookup an object with a given name and tag.

        This is a general-purpose method for objects like functions, network file systems,
        and secrets. It gives a reference to the object in a running app.

        **Examples**

        ```python notest
        # Lookup a secret
        my_secret = Secret.lookup("my-secret")

        # Lookup a function from a different app
        other_function = Function.lookup("other-app", "function")

        # Lookup a persisted Volume, Queue, or Dict
        my_volume = Volume.lookup("my-volume")
        my_queue = Queue.lookup("my-queue")
        my_dict = Dict.lookup("my-dict")
        ```
        """
        # TODO(erikbern): this code is very duplicated. Clean up once handles are gone.
        rep = f"Object({app_name})"  # TODO(erikbern): dumb
        obj = _Object.__new__(cls)
        obj._init(rep)
        await obj._hydrate_from_app(app_name, tag, namespace, client, environment_name=environment_name)
        return obj

    @classmethod
    async def _exists(
        cls: Type[O],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> bool:
        """Internal for now - will make this "public" later."""
        if client is None:
            client = await _Client.from_env()
        if environment_name is None:
            environment_name = config.get("environment")
        request = api_pb2.AppLookupObjectRequest(
            app_name=app_name,
            object_tag=tag,
            namespace=namespace,
            object_entity=cls._type_prefix,
            environment_name=environment_name,
        )
        try:
            response = await retry_transient_errors(client.stub.AppLookupObject, request)
            return bool(response.object.object_id)  # old code path - change to `return True` shortly
        except GRPCError as exc:
            if exc.status == Status.NOT_FOUND:
                return False
            else:
                raise


Object = synchronize_api(_Object, target_module=__name__)


def live_method(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        if not await self._try_hydrate():
            raise ExecutionError(f"Calling method `{method.__name__}` requires the object to be hydrated.")
        return await method(self, *args, **kwargs)

    return wrapped


def live_method_gen(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        if not await self._try_hydrate():
            raise ExecutionError(f"Calling method `{method.__name__}` requires the object to be hydrated.")
        async for item in method(self, *args, **kwargs):
            yield item

    return wrapped
