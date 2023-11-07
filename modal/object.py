# Copyright Modal Labs 2022
import uuid
from datetime import date
from functools import wraps
from typing import Awaitable, Callable, ClassVar, Dict, List, Optional, Type, TypeVar

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal._types import typechecked
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import get_proto_oneof, retry_transient_errors

from ._deployments import deploy_single_object
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
    _rep: str
    _is_another_app: bool
    _hydrate_lazily: bool
    _deps: Optional[Callable[..., List["_Object"]]]

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
        is_another_app: bool = False,
        preload: Optional[Callable[[O, Resolver, Optional[str]], Awaitable[None]]] = None,
        hydrate_lazily: bool = False,
        deps: Optional[Callable[..., List["_Object"]]] = None,
    ):
        self._local_uuid = str(uuid.uuid4())
        self._load = load
        self._preload = preload
        self._rep = rep
        self._is_another_app = is_another_app
        self._hydrate_lazily = hydrate_lazily
        self._deps = deps

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
        assert isinstance(object_id, str)
        if not object_id.startswith(self._type_prefix):
            raise ExecutionError(
                f"Can not hydrate {type(self)}:"
                f" it has type prefix {self._type_prefix}"
                f" but the object_id starts with {object_id[:3]}"
            )
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
        self._init(other._rep, other._load, other._is_another_app, other._preload)

    @classmethod
    def _from_loader(
        cls,
        load: Callable[[O, Resolver, Optional[str]], Awaitable[None]],
        rep: str,
        is_another_app: bool = False,
        preload: Optional[Callable[[O, Resolver, Optional[str]], Awaitable[None]]] = None,
        hydrate_lazily: bool = False,
        deps: Optional[Callable[..., List["_Object"]]] = None,
    ):
        # TODO(erikbern): flip the order of the two first arguments
        obj = _Object.__new__(cls)
        obj._init(rep, load, is_another_app, preload, hydrate_lazily, deps)
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

    @property
    def is_hydrated(self) -> bool:
        """mdmd:hidden"""
        return self._is_hydrated

    @property
    def deps(self) -> Callable[..., List["_Object"]]:
        return self._deps if self._deps is not None else lambda: []

    async def resolve(self):
        """mdmd:hidden"""
        if self._is_hydrated:
            return
        elif not self._hydrate_lazily:
            raise ExecutionError(
                "Object has not been hydrated and doesn't support lazy hydration."
                " This might happen if an object is defined on a different stub,"
                " or if it's on the same stub but it didn't get created because it"
                " wasn't defined in global scope."
            )
        else:
            resolver = Resolver()
            await resolver.load(self)

    async def _deploy(
        self,
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> None:
        """
        Note: still considering this an "internal" method, but we'll make it "official" later
        """
        if environment_name is None:
            environment_name = config.get("environment")

        if client is None:
            client = await _Client.from_env()

        await deploy_single_object(self, self._type_prefix, client, label, namespace, environment_name)

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
        return cls._from_loader(_load_persisted, rep, is_another_app=True)

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
                # fall back if no explicit environment was set in the call itself.
                # If there is a current app setup, the resolver has the env name. If doing a .lookup
                # with no associated app, must fetch environment from config.
                environment_name = resolver.environment_name or config.get("environment")

            request = api_pb2.AppLookupObjectRequest(
                app_name=app_name,
                object_tag=tag,
                namespace=namespace,
                object_entity=cls._type_prefix,
                environment_name=environment_name,
            )
            try:
                response = await retry_transient_errors(resolver.client.stub.AppLookupObject, request)
            except GRPCError as exc:
                if exc.status == Status.NOT_FOUND:
                    raise NotFoundError(exc.message)
                else:
                    raise

            handle_metadata = get_proto_oneof(response.object, "handle_metadata_oneof")
            obj._hydrate(response.object.object_id, resolver.client, handle_metadata)

        rep = f"Ref({app_name})"
        return cls._from_loader(_load_remote, rep, is_another_app=True)

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
        obj = cls.from_name(app_name, tag, namespace=namespace, environment_name=environment_name)
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
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
        await self.resolve()
        return await method(self, *args, **kwargs)

    return wrapped


def live_method_gen(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        await self.resolve()
        async for item in method(self, *args, **kwargs):
            yield item

    return wrapped
