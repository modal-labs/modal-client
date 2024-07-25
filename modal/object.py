# Copyright Modal Labs 2022
import uuid
from functools import wraps
from typing import Awaitable, Callable, ClassVar, Dict, Hashable, List, Optional, Sequence, Type, TypeVar

from google.protobuf.message import Message

from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from .client import _Client
from .config import config
from .exception import ExecutionError, InvalidError

O = TypeVar("O", bound="_Object")

_BLOCKING_O = synchronize_api(O)

EPHEMERAL_OBJECT_HEARTBEAT_SLEEP = 300


def _get_environment_name(environment_name: Optional[str], resolver: Optional[Resolver] = None) -> Optional[str]:
    if environment_name:
        return environment_name
    elif resolver and resolver.environment_name:
        return resolver.environment_name
    else:
        return config.get("environment")


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
    _deduplication_key: Optional[Callable[[], Awaitable[Hashable]]] = None

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
        deduplication_key: Optional[Callable[[], Awaitable[Hashable]]] = None,
    ):
        self._local_uuid = str(uuid.uuid4())
        self._load = load
        self._preload = preload
        self._rep = rep
        self._is_another_app = is_another_app
        self._hydrate_lazily = hydrate_lazily
        self._deps = deps
        self._deduplication_key = deduplication_key

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

    def _initialize_from_other(self, other):
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

    def _validate_is_hydrated(self: O):
        if not self._is_hydrated:
            object_type = self.__class__.__name__.strip("_")
            if hasattr(self, "_app") and getattr(self._app, "_running_app", "") is None:
                # The most common cause of this error: e.g., user called a Function without using App.run()
                reason = ", because the App it is defined on is not running."
            else:
                # Technically possible, but with an ambiguous cause.
                reason = ""
            raise ExecutionError(
                f"{object_type} has not been hydrated with the metadata it needs to run on Modal{reason}."
            )

    def clone(self: O) -> O:
        """mdmd:hidden Clone a given hydrated object."""

        # Object to clone must already be hydrated, otherwise from_loader is more suitable.
        self._validate_is_hydrated()
        obj = _Object.__new__(type(self))
        obj._initialize_from_other(self)
        return obj

    @classmethod
    def _from_loader(
        cls,
        load: Callable[[O, Resolver, Optional[str]], Awaitable[None]],
        rep: str,
        is_another_app: bool = False,
        preload: Optional[Callable[[O, Resolver, Optional[str]], Awaitable[None]]] = None,
        hydrate_lazily: bool = False,
        deps: Optional[Callable[..., Sequence["_Object"]]] = None,
        deduplication_key: Optional[Callable[[], Awaitable[Hashable]]] = None,
    ):
        # TODO(erikbern): flip the order of the two first arguments
        obj = _Object.__new__(cls)
        obj._init(rep, load, is_another_app, preload, hydrate_lazily, deps, deduplication_key)
        return obj

    @classmethod
    def _new_hydrated(
        cls: Type[O], object_id: str, client: _Client, handle_metadata: Optional[Message], is_another_app: bool = False
    ) -> O:
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
        obj._init(rep, is_another_app=is_another_app)
        obj._hydrate(object_id, client, handle_metadata)

        return obj

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
        """mdmd:hidden"""
        return self._deps if self._deps is not None else lambda: []

    async def resolve(self):
        """mdmd:hidden"""
        if self._is_hydrated:
            return
        elif not self._hydrate_lazily:
            self._validate_is_hydrated()
        else:
            # TODO: this client and/or resolver can't be changed by a caller to X.from_name()
            resolver = Resolver(await _Client.from_env())
            await resolver.load(self)


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
