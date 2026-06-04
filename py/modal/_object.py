# Copyright Modal Labs 2022
import contextlib
import typing
import uuid
from collections.abc import Awaitable, Callable, Hashable, Sequence
from functools import wraps
from typing import ClassVar

from google.protobuf.message import Message
from typing_extensions import Self

from modal._traceback import suppress_tb_frame

from ._load_context import LoadContext
from ._resolver import Resolver
from ._utils.async_utils import TaskContext, aclosing
from .client import _Client
from .config import config, logger
from .exception import ExecutionError, InvalidError

EPHEMERAL_OBJECT_HEARTBEAT_SLEEP: int = 300


def _get_environment_name(
    environment_name: str | None = None,
) -> str | None:
    """Get environment name from various sources.

    Args:
        environment_name: Explicitly provided environment name (highest priority)

    Returns:
        Environment name from first available source, or config default
    """
    if environment_name:
        return environment_name
    else:
        return config.get("environment")


def live_method(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        await self.hydrate()
        return await method(self, *args, **kwargs)

    return wrapped


def live_method_gen(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        await self.hydrate()
        async with aclosing(method(self, *args, **kwargs)) as stream:
            async for item in stream:
                yield item

    return wrapped


def live_method_contextmanager(method):
    # make sure a wrapped function returning an async context manager
    # will not require both an `await func.aio()` and `async with`
    # which would have been the case if it was wrapped in live_method

    @wraps(method)
    @contextlib.asynccontextmanager
    async def wrapped(self, *args, **kwargs):
        await self.hydrate()
        async with method(self, *args, **kwargs) as ctx:
            yield ctx

    return wrapped


class _Object:
    _type_prefix: ClassVar[str | None] = None
    _prefix_to_type: ClassVar[dict[str, type]] = {}

    # For constructors
    _load: Callable[[Self, Resolver, LoadContext, str | None], Awaitable[None]] | None = None
    _preload: Callable[[Self, Resolver, LoadContext, str | None], Awaitable[None]] | None
    _rep: str
    _skip_reload: bool
    _hydrate_lazily: bool
    _deps: Callable[..., Sequence["_Object"]] | None
    _deduplication_key: Callable[[], Awaitable[Hashable]] | None = None
    _load_context_overrides: LoadContext

    # For hydrated objects
    _object_id: str | None
    _client: _Client | None
    _is_hydrated: bool
    _is_rehydrated: bool

    # Not all object subclasses have a meaningful "name" concept
    # So whether they expose this is a matter of having a name property
    _name: str | None

    @classmethod
    def __init_subclass__(cls, type_prefix: str | None = None):
        super().__init_subclass__()
        if type_prefix is not None:
            cls._type_prefix = type_prefix
            cls._prefix_to_type[type_prefix] = cls

    def __init__(self, *args, **kwargs):
        """mdmd:hidden"""
        raise InvalidError(f"Class {type(self).__name__} has no constructor. Use class constructor methods instead.")

    def _init(
        self,
        rep: str,
        load: Callable[[Self, Resolver, LoadContext, str | None], Awaitable[None]] | None = None,
        preload: Callable[[Self, Resolver, LoadContext, str | None], Awaitable[None]] | None = None,
        hydrate_lazily: bool = False,
        skip_reload: bool = False,
        deps: Callable[..., Sequence["_Object"]] | None = None,
        deduplication_key: Callable[[], Awaitable[Hashable]] | None = None,
        name: str | None = None,
        *,
        load_context_overrides: LoadContext | None = None,
    ):
        self._local_uuid = str(uuid.uuid4())
        self._rep = rep
        self._load = load
        self._preload = preload
        self._skip_reload = skip_reload
        self._hydrate_lazily = hydrate_lazily
        self._deps = deps
        self._deduplication_key = deduplication_key
        self._load_context_overrides = (
            load_context_overrides if load_context_overrides is not None else LoadContext.empty()
        )

        self._object_id = None
        self._client = None
        self._is_hydrated = False
        self._is_rehydrated = False

        self._name = name

        self._initialize_from_empty()

    def _unhydrate(self):
        self._object_id = None
        self._client = None
        self._is_hydrated = False

    def _initialize_from_empty(self):
        # default implementation, can be overriden in subclasses
        pass

    def _initialize_from_other(self, other):
        # default implementation, can be overriden in subclasses
        self._object_id = other._object_id
        self._is_hydrated = other._is_hydrated
        self._client = other._client

    def _hydrate(self, object_id: str, client: _Client, metadata: Message | None):
        assert isinstance(object_id, str) and self._type_prefix is not None
        if not object_id.startswith(self._type_prefix):
            raise ExecutionError(
                f"Can not hydrate {type(self)}: "
                f" it has type prefix {self._type_prefix}"
                f" but the object_id starts with {object_id[:3]}. "
                "This usually means the object name was previously used for a different type. "
                "Rename the object/app or stop the previous deployment and redeploy."
            )
        self._object_id = object_id
        self._client = client
        self._hydrate_metadata(metadata)
        self._is_hydrated = True

    def _hydrate_metadata(self, metadata: Message | None):
        # override this if it's a subclass that needs additional data (other than an object_id) for a functioning Handle
        pass

    def _get_metadata(self) -> Message | None:
        # return the necessary metadata from this handle to be able to re-hydrate in another context if one is needed
        # used to provide a handle's handle_metadata for serializing/pickling a live handle
        # the object_id is already provided by other means
        return None

    def _validate_is_hydrated(self):
        if not self._is_hydrated:
            object_type = self.__class__.__name__.strip("_")
            if hasattr(self, "_app") and getattr(self._app, "_running_app", "") is None:  # type: ignore
                # The most common cause of this error: e.g., user called a Function without using App.run()
                reason = ", because the App it is defined on is not running"
            else:
                # Technically possible, but with an ambiguous cause.
                reason = ""
            raise ExecutionError(
                f"{object_type} has not been hydrated with the metadata it needs to run on Modal{reason}."
            )

    @classmethod
    def _from_loader(
        cls,
        load: Callable[[Self, Resolver, LoadContext, str | None], Awaitable[None]],
        rep: str,
        skip_reload: bool = False,
        preload: Callable[[Self, Resolver, LoadContext, str | None], Awaitable[None]] | None = None,
        hydrate_lazily: bool = False,
        deps: Callable[..., Sequence["_Object"]] | None = None,
        deduplication_key: Callable[[], Awaitable[Hashable]] | None = None,
        name: str | None = None,
        *,
        load_context_overrides: LoadContext,
    ):
        # TODO(erikbern): flip the order of the two first arguments
        obj = _Object.__new__(cls)
        obj._init(
            rep,
            load,
            skip_reload=skip_reload,
            preload=preload,
            hydrate_lazily=hydrate_lazily,
            deps=deps,
            deduplication_key=deduplication_key,
            name=name,
            load_context_overrides=load_context_overrides,
        )
        return obj

    @staticmethod
    def _get_type_from_id(object_id: str) -> type["_Object"]:
        parts = object_id.split("-")
        if len(parts) != 2:
            raise InvalidError(f"Object id {object_id} has no dash in it")
        prefix = parts[0]
        if prefix not in _Object._prefix_to_type:
            raise InvalidError(f"Object prefix {prefix} does not correspond to a type")
        return _Object._prefix_to_type[prefix]

    @classmethod
    def _is_id_type(cls, object_id) -> bool:
        return cls._get_type_from_id(object_id) == cls

    @classmethod
    def _repr(cls, name: str, environment_name: str | None = None) -> str:
        public_cls = cls.__name__.strip("_")
        environment_repr = f", environment_name={environment_name!r}" if environment_name else ""
        return f"modal.{public_cls}.from_name({name!r}{environment_repr})"

    @classmethod
    def _new_hydrated(
        cls,
        object_id: str,
        client: _Client,
        handle_metadata: Message | None,
        skip_reload: bool = False,
        rep: str | None = None,
    ) -> Self:
        obj_cls: type[Self]
        if cls._type_prefix is not None:
            # This is called directly on a subclass, e.g. Secret.from_id
            # validate the id matching the expected id type of the Object subclass
            if not object_id.startswith(cls._type_prefix + "-"):
                raise InvalidError(f"Object {object_id} does not start with {cls._type_prefix}")

            obj_cls = cls
        else:
            # this means the method is used directly on _Object
            # typically during deserialization of objects
            obj_cls = typing.cast(type[Self], cls._get_type_from_id(object_id))

        # Instantiate provider
        obj = _Object.__new__(obj_cls)
        rep = rep or f"modal.{obj_cls.__name__.strip('_')}.from_id({object_id!r})"
        obj._init(rep, skip_reload=skip_reload)
        obj._hydrate(object_id, client, handle_metadata)

        return obj

    def _hydrate_from_other(self, other: Self):
        self._hydrate(other.object_id, other.client, other._get_metadata())

    def __repr__(self):
        return self._rep

    @property
    def local_uuid(self):
        """mdmd:hidden"""
        return self._local_uuid

    @property
    def object_id(self) -> str:
        """mdmd:hidden"""
        if self._object_id is None:
            raise AttributeError(f"Attempting to get object_id of unhydrated {self}")
        return self._object_id

    @live_method
    async def get_dashboard_url(self) -> str:
        """mdmd:hidden"""
        return f"https://modal.com/id/{self.object_id}"

    @property
    def client(self) -> _Client:
        """mdmd:hidden"""
        if self._client is None:
            raise AttributeError(f"Attempting to get client of unhydrated {self}")
        return self._client

    @property
    def is_hydrated(self) -> bool:
        """mdmd:hidden"""
        return self._is_hydrated

    @property
    def deps(self) -> Callable[..., Sequence["_Object"]]:
        """mdmd:hidden"""

        def default_deps(*args, **kwargs) -> Sequence["_Object"]:
            return []

        return self._deps if self._deps is not None else default_deps

    async def hydrate(self, client: _Client | None = None) -> Self:
        """Synchronize the local object with its identity on the Modal server.

        It is rarely necessary to call this method explicitly, as most operations
        will lazily hydrate when needed. The main use case is when you need to
        access object metadata, such as its ID.

        *Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.
        """
        # TODO: add deprecation for the client argument here - should be added in constructors instead
        if self._is_hydrated:
            if self.client._snapshotted and not self._is_rehydrated:
                # memory snapshots capture references which must be rehydrated
                # on restore to handle staleness.
                logger.debug(f"rehydrating {self} after snapshot")
                if self._hydrate_lazily:
                    logger.debug(f"reloading lazy {self} from server")
                    self._is_hydrated = False  # un-hydrate and re-resolve
                    # we don't set an explicit Client here, relying on the default
                    # env client to be applied by LoadContext.apply_default
                    resolver = Resolver()
                    async with TaskContext() as tc:
                        root_load_context = LoadContext(task_context=tc)
                        await resolver.load(typing.cast(_Object, self), root_load_context)
                else:
                    logger.debug(f"reloading non-lazy {self} by replacing client")
                    self._client = client or await _Client.from_env()
                self._is_rehydrated = True
                logger.debug(f"rehydrated {self} with client {id(self.client)}")
        elif not self._hydrate_lazily:
            self._validate_is_hydrated()
        else:
            # Set the client on LoadContext before loading, with a TaskContext for proper
            # exception handling when loading shared dependencies
            resolver = Resolver()
            async with TaskContext() as tc:
                root_load_context = LoadContext(client=client, task_context=tc)
                with suppress_tb_frame():  # skip this frame by default
                    await resolver.load(self, root_load_context)
        return self
