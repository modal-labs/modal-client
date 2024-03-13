# Copyright Modal Labs 2022
import os
import pickle
import typing
from typing import Any, Callable, Collection, Dict, List, Optional, Type, TypeVar, Union

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal_proto import api_pb2

from ._output import OutputManager
from ._resolver import Resolver
from ._utils.async_utils import synchronize_api, synchronizer
from ._utils.grpc_utils import retry_transient_errors
from ._utils.mount_utils import validate_volumes
from .client import _Client
from .exception import InvalidError, NotFoundError, deprecation_error
from .functions import (
    _parse_retries,
)
from .gpu import GPU_T, parse_gpu_config
from .object import _get_environment_name, _Object
from .partial_function import (
    PartialFunction,
    _find_callables_for_cls,
    _find_partial_methods_for_cls,
    _Function,
    _PartialFunctionFlags,
)
from .retries import Retries
from .secret import _Secret
from .volume import _Volume

T = TypeVar("T")


if typing.TYPE_CHECKING:
    import modal.stub


class ClsMixin:
    def __init_subclass__(cls):
        deprecation_error((2023, 9, 1), "`ClsMixin` is deprecated and can be safely removed.")


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
        self,
        user_cls: type,
        output_mgr: Optional[OutputManager],
        base_functions: Dict[str, _Function],
        from_other_workspace: bool,
        options: Optional[api_pb2.FunctionOptions],
        args,
        kwargs,
    ):
        for i, arg in enumerate(args):
            check_picklability(i + 1, arg)
        for key, kwarg in kwargs.items():
            check_picklability(key, kwarg)

        self._functions = {}
        for k, fun in base_functions.items():
            self._functions[k] = fun.from_parametrized(self, from_other_workspace, options, args, kwargs)
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
    _options: Optional[api_pb2.FunctionOptions]
    _callables: Dict[str, Callable]
    _from_other_workspace: Optional[bool]  # Functions require FunctionBindParams before invocation.
    _stub: Optional["modal.stub._Stub"] = None  # not set for lookups

    def _initialize_from_empty(self):
        self._user_cls = None
        self._functions = {}
        self._options = None
        self._callables = {}
        self._from_other_workspace = None
        self._output_mgr: Optional[OutputManager] = None

    def _initialize_from_other(self, other: "_Cls"):
        self._user_cls = other._user_cls
        self._functions = other._functions
        self._options = other._options
        self._callables = other._callables
        self._from_other_workspace = other._from_other_workspace
        self._output_mgr: Optional[OutputManager] = other._output_mgr

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
        """mdmd:hidden"""
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

        async def _load(self: "_Cls", resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.ClassCreateRequest(app_id=resolver.app_id, existing_class_id=existing_object_id)
            for f_name, f in functions.items():
                req.methods.append(api_pb2.ClassMethod(function_name=f_name, function_id=f.object_id))
            resp = await resolver.client.stub.ClassCreate(req)
            self._hydrate(resp.class_id, resolver.client, resp.handle_metadata)

        rep = f"Cls({user_cls.__name__})"
        cls = _Cls._from_loader(_load, rep, deps=_deps)
        cls._stub = stub
        cls._user_cls = user_cls
        cls._functions = functions
        cls._callables = callables
        cls._from_other_workspace = False
        setattr(cls._user_cls, "_modal_functions", functions)  # Needed for PartialFunction.__get__
        return cls

    @classmethod
    def from_name(
        cls: Type["_Cls"],
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        workspace: Optional[str] = None,
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
                lookup_published=workspace is not None,
                workspace_name=workspace,
            )
            try:
                response = await retry_transient_errors(resolver.client.stub.ClassGet, request)
            except GRPCError as exc:
                if exc.status == Status.NOT_FOUND:
                    raise NotFoundError(exc.message)
                elif exc.status == Status.FAILED_PRECONDITION:
                    raise InvalidError(exc.message)
                else:
                    raise

            obj._hydrate(response.class_id, resolver.client, response.handle_metadata)

        rep = f"Ref({app_name})"
        cls = cls._from_loader(_load_remote, rep, is_another_app=True)
        cls._from_other_workspace = bool(workspace is not None)
        return cls

    def with_options(
        self: "_Cls",
        cpu: Optional[float] = None,
        memory: Optional[int] = None,
        gpu: GPU_T = None,
        secrets: Collection[_Secret] = (),
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},
        retries: Optional[Union[int, Retries]] = None,
        timeout: Optional[int] = None,
        concurrency_limit: Optional[int] = None,
        allow_concurrent_inputs: Optional[int] = None,
        container_idle_timeout: Optional[int] = None,
        allow_background_volume_commits: bool = False,
    ) -> "_Cls":
        """
        Allows for the runtime modification of a modal.Cls's configuration.
        Designed for usage in the [MK1 Flywheel](/docs/guide/mk1).

        **Usage:**

        ```python notest
        import modal
        Model = modal.Cls.lookup(
            "flywheel-generic", "Model", workspace="mk-1"
        )
        Model2 = Model.with_options(
            gpu=modal.gpu.A100(memory=40),
            volumes={"/models": models_vol}
        )
        Model2().generate.remote(42)
        ```
        """
        retry_policy = _parse_retries(retries)
        if gpu or cpu or memory:
            milli_cpu = int(1000 * cpu) if cpu is not None else None
            gpu_config = parse_gpu_config(gpu)
            resources = api_pb2.Resources(milli_cpu=milli_cpu, gpu_config=gpu_config, memory_mb=memory)
        else:
            resources = None

        volume_mounts = [
            api_pb2.VolumeMount(
                mount_path=path,
                volume_id=volume.object_id,
                allow_background_commits=allow_background_volume_commits,
            )
            for path, volume in validate_volumes(volumes)
        ]
        replace_volume_mounts = len(volume_mounts) > 0

        cls = self.clone()
        cls._options = api_pb2.FunctionOptions(
            replace_secret_ids=bool(secrets),
            secret_ids=[secret.object_id for secret in secrets],
            resources=resources,
            retry_policy=retry_policy,
            concurrency_limit=concurrency_limit,
            timeout_secs=timeout,
            task_idle_timeout_secs=container_idle_timeout,
            replace_volume_mounts=replace_volume_mounts,
            volume_mounts=volume_mounts,
            allow_concurrent_inputs=allow_concurrent_inputs,
        )

        return cls

    @staticmethod
    async def lookup(
        app_name: str,
        tag: Optional[str] = None,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> "_Cls":
        """Lookup a class with a given name and tag.

        ```python
        Class = modal.Cls.lookup("other-app", "Class")
        ```
        """
        obj = _Cls.from_name(app_name, tag, namespace=namespace, environment_name=environment_name, workspace=workspace)
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

    def __call__(self, *args, **kwargs) -> _Obj:
        """This acts as the class constructor."""
        return _Obj(
            self._user_cls, self._output_mgr, self._functions, self._from_other_workspace, self._options, args, kwargs
        )

    async def remote(self, *args, **kwargs):
        """`Cls.remote(...)` on classes is deprecated. Use the constructor: `Cls(...)`."""
        deprecation_error((2023, 9, 1), "`Cls.remote(...)` on classes is deprecated. Use the constructor: `Cls(...)`.")

    def __getattr__(self, k):
        # Used by CLI and container entrypoint
        if k in self._functions:
            return self._functions[k]
        return getattr(self._user_cls, k)


Cls = synchronize_api(_Cls)
