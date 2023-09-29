# Copyright Modal Labs 2022
import asyncio
import pickle
import warnings
from datetime import date
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from google.protobuf.message import Message

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api

from ._output import OutputManager
from ._resolver import Resolver
from .exception import deprecation_warning
from .functions import _Function
from .object import _Object

T = TypeVar("T")


class ClsMixin:
    def __init_subclass__(cls):
        deprecation_warning(date(2023, 9, 1), "`ClsMixin` is deprecated and can be safely removed.")

    @classmethod
    def remote(cls: Type[T], *args, **kwargs) -> T:
        ...

    @classmethod
    async def aio_remote(cls: Type[T], *args, **kwargs) -> T:
        ...


def check_picklability(key, arg):
    try:
        pickle.dumps(arg)
    except Exception:
        raise ValueError(
            f"Only pickle-able types are allowed in remote class constructors: argument {key} of type {type(arg)}."
        )


class _Obj:
    """An instance of a `Cls`, i.e. `Cls("foo", 42)` returns an `Obj`.

    All this class does is to return `Function` objects."""

    _functions: Dict[str, _Function]
    _has_local_obj: bool
    _local_obj: Any
    _local_obj_constr: Optional[Callable[[], Any]]

    def __init__(
        self, user_cls: type, output_mgr: Optional[OutputManager], base_functions: Dict[str, _Function], args, kwargs
    ):
        for i, arg in enumerate(args):
            check_picklability(i + 1, arg)
        for key, kwarg in kwargs.items():
            check_picklability(key, kwarg)

        self._functions = {}
        for k, fun in base_functions.items():
            self._functions[k] = fun.from_parametrized(self, args, kwargs)
            self._functions[k]._set_output_mgr(output_mgr)

        # Used for construction local object lazily
        self._has_local_obj = False
        self._local_obj = None
        if user_cls:
            self._local_obj_constr = lambda: user_cls(*args, **kwargs)
        else:
            self._local_obj_constr = None

    def get_obj(self):
        """Constructs obj without any caching. Used by container entrypoint."""
        self._local_obj = self._local_obj_constr()
        setattr(self._local_obj, "_modal_functions", self._functions)  # Needed for PartialFunction.__get__
        return self._local_obj

    def get_local_obj(self):
        """Construct local object lazily. Used for .local() calls."""
        if not self._has_local_obj:
            self.get_obj()  # Instantiate object
            if hasattr(self._local_obj, "__enter__"):
                self._local_obj.__enter__()
            elif hasattr(self._local_obj, "__aenter__"):
                warnings.warn("Not running asynchronous enter handlers on local objects")
            self._has_local_obj = True

        return self._local_obj

    def __getattr__(self, k):
        if k in self._functions:
            return self._functions[k]
        elif self._local_obj_constr:
            obj = self.get_local_obj()
            return getattr(obj, k)
        else:
            raise AttributeError(k)


Obj = synchronize_api(_Obj)


class _Cls(_Object, type_prefix="cs"):
    _user_cls: Optional[type]
    _functions: Dict[str, _Function]

    def _initialize_from_empty(self):
        self._user_cls = None
        self._base_functions = {}
        self._output_mgr: Optional[OutputManager] = None

    def _set_output_mgr(self, output_mgr: OutputManager):
        self._output_mgr = output_mgr

    def _hydrate_metadata(self, metadata: Message):
        for method in metadata.methods:
            if method.function_name in self._base_functions:
                self._base_functions[method.function_name]._hydrate(
                    method.function_id, self._client, method.function_handle_metadata
                )
            else:
                self._base_functions[method.function_name] = _Function._new_hydrated(
                    method.function_id, self._client, method.function_handle_metadata
                )

    def _get_metadata(self) -> api_pb2.ClassHandleMetadata:
        class_handle_metadata = api_pb2.ClassHandleMetadata()
        for f_name, f in self._base_functions.items():
            class_handle_metadata.methods.append(
                api_pb2.ClassMethod(
                    function_name=f_name, function_id=f.object_id, function_handle_metadata=f._get_metadata()
                )
            )
        return class_handle_metadata

    @staticmethod
    def from_local(user_cls, base_functions: Dict[str, _Function]) -> "_Cls":
        async def _load(provider: _Object, resolver: Resolver, existing_object_id: Optional[str]):
            # Make sure all functions are loaded
            await asyncio.gather(*[resolver.load(function) for function in base_functions.values()])

            # Create class remotely
            req = api_pb2.ClassCreateRequest(app_id=resolver.app_id, existing_class_id=existing_object_id)
            for f_name, f in base_functions.items():
                req.methods.append(api_pb2.ClassMethod(function_name=f_name, function_id=f.object_id))
            resp = await resolver.client.stub.ClassCreate(req)
            provider._hydrate(resp.class_id, resolver.client, resp.handle_metadata)

        rep = f"Cls({user_cls.__name__})"
        cls = _Cls._from_loader(_load, rep)
        cls._user_cls = user_cls
        cls._base_functions = base_functions
        setattr(cls._user_cls, "_modal_functions", base_functions)  # Needed for PartialFunction.__get__
        return cls

    def get_user_cls(self):
        # Used by the container entrypoint
        return self._user_cls

    def get_base_function(self, k: str) -> _Function:
        return self._base_functions[k]

    def __call__(self, *args, **kwargs) -> _Obj:
        """This acts as the class constructor."""
        return _Obj(self._user_cls, self._output_mgr, self._base_functions, args, kwargs)

    async def remote(self, *args, **kwargs) -> _Obj:
        deprecation_warning(
            date(2023, 9, 1), "`Cls.remote(...)` on classes is deprecated. Use the constructor: `Cls(...)`."
        )
        return self(*args, **kwargs)

    def __getattr__(self, k):
        # Used by CLI and container entrypoint
        if k in self._base_functions:
            return self._base_functions[k]
        return getattr(self._user_cls, k)


Cls = synchronize_api(_Cls)
