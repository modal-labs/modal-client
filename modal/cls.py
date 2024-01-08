# Copyright Modal Labs 2022
import pickle
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api, synchronizer
from modal_utils.grpc_utils import retry_transient_errors

from ._output import OutputManager
from ._resolver import Resolver
from .client import _Client
from .exception import NotFoundError, deprecation_error
from .functions import _Function
from .object import _get_environment_name, _Object
from .partial_function import (
    PartialFunction,
    _find_callables_for_cls,
    _find_partial_methods_for_cls,
    _PartialFunctionFlags,
)

T = TypeVar("T")


class ClsMixin:
    def __init_subclass__(cls):
        deprecation_error(date(2023, 9, 1), "`ClsMixin` is deprecated and can be safely removed.")


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
    _inited: bool
    _entered: bool
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
        self._inited = False
        self._entered = False
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
        if not self._inited:
            self.get_obj()  # Instantiate object
            self._inited = True

        return self._local_obj

    def enter(self):
        if not self._entered:
            if hasattr(self._local_obj, "__enter__"):
                self._local_obj.__enter__()
        self._entered = True

    @property
    def entered(self):
        # needed because aenter is nowrap
        return self._entered

    @entered.setter
    def entered(self, val):
        self._entered = val

    @synchronizer.nowrap
    async def aenter(self):
        if not self.entered:
            local_obj = self.get_local_obj()
            if hasattr(local_obj, "__aenter__"):
                await local_obj.__aenter__()
            elif hasattr(local_obj, "__enter__"):
                local_obj.__enter__()
        self.entered = True

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
    _callables: Dict[str, Callable]

    def _initialize_from_empty(self):
        self._user_cls = None
        self._functions = {}
        self._callables = {}
        self._output_mgr: Optional[OutputManager] = None

    def _set_output_mgr(self, output_mgr: OutputManager):
        self._output_mgr = output_mgr

    def _hydrate_metadata(self, metadata: Message):
        for method in metadata.methods:
            if method.function_name in self._functions:
                self._functions[method.function_name]._hydrate(
                    method.function_id, self._client, method.function_handle_metadata
                )
            else:
                self._functions[method.function_name] = _Function._new_hydrated(
                    method.function_id, self._client, method.function_handle_metadata
                )

    def _get_metadata(self) -> api_pb2.ClassHandleMetadata:
        class_handle_metadata = api_pb2.ClassHandleMetadata()
        for f_name, f in self._functions.items():
            class_handle_metadata.methods.append(
                api_pb2.ClassMethod(
                    function_name=f_name, function_id=f.object_id, function_handle_metadata=f._get_metadata()
                )
            )
        return class_handle_metadata

    @staticmethod
    def from_local(user_cls, stub, decorator: Callable[[PartialFunction, type], _Function]) -> "_Cls":
        functions: Dict[str, _Function] = {}
        for k, partial_function in _find_partial_methods_for_cls(user_cls, _PartialFunctionFlags.FUNCTION).items():
            functions[k] = decorator(partial_function, user_cls)

        # Disable the warning that these are not wrapped
        for partial_function in _find_partial_methods_for_cls(user_cls, ~_PartialFunctionFlags.FUNCTION).values():
            partial_function.wrapped = True

        # Get all callables
        callables: Dict[str, Callable] = _find_callables_for_cls(user_cls, ~_PartialFunctionFlags(0))

        def _deps() -> List[_Function]:
            return list(functions.values())

        async def _load(provider: "_Cls", resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.ClassCreateRequest(app_id=resolver.app_id, existing_class_id=existing_object_id)
            for f_name, f in functions.items():
                req.methods.append(api_pb2.ClassMethod(function_name=f_name, function_id=f.object_id))
            resp = await resolver.client.stub.ClassCreate(req)
            provider._hydrate(resp.class_id, resolver.client, resp.handle_metadata)

        rep = f"Cls({user_cls.__name__})"
        cls = _Cls._from_loader(_load, rep, deps=_deps)
        cls._stub = stub
        cls._user_cls = user_cls
        cls._functions = functions
        cls._callables = callables
        setattr(cls._user_cls, "_modal_functions", functions)  # Needed for PartialFunction.__get__
        return cls

    @classmethod
    def from_name(
        cls: Type["_Cls"],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
    ) -> "_Cls":
        """Retrieve a class with a given name and tag.

        ```python
        Class = modal.Cls.from_name("other-app", "Class")
        ```
        """

        async def _load_remote(obj: _Object, resolver: Resolver, existing_object_id: Optional[str]):
            request = api_pb2.ClassGetRequest(
                app_name=app_name,
                object_tag=tag,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
            )
            try:
                response = await retry_transient_errors(resolver.client.stub.ClassGet, request)
            except GRPCError as exc:
                if exc.status == Status.NOT_FOUND:
                    raise NotFoundError(exc.message)
                else:
                    raise

            obj._hydrate(response.class_id, resolver.client, response.handle_metadata)

        rep = f"Ref({app_name})"
        return cls._from_loader(_load_remote, rep, is_another_app=True)

    @classmethod
    async def lookup(
        cls: Type["_Cls"],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ) -> "_Cls":
        """Lookup a class with a given name and tag.

        ```python
        Class = modal.Cls.lookup("other-app", "Class")
        ```
        """
        obj = cls.from_name(app_name, tag, namespace=namespace, environment_name=environment_name)
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

    def __call__(self, *args, **kwargs) -> _Obj:
        """This acts as the class constructor."""
        return _Obj(self._user_cls, self._output_mgr, self._functions, args, kwargs)

    async def remote(self, *args, **kwargs):
        deprecation_error(
            date(2023, 9, 1), "`Cls.remote(...)` on classes is deprecated. Use the constructor: `Cls(...)`."
        )

    def __getattr__(self, k):
        # Used by CLI and container entrypoint
        if k in self._functions:
            return self._functions[k]
        return getattr(self._user_cls, k)


Cls = synchronize_api(_Cls)
