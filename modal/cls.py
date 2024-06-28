# Copyright Modal Labs 2022
import os
import typing
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Type, TypeVar, Union

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal_proto import api_pb2

from ._output import OutputManager
from ._resolver import Resolver
from ._resources import convert_fn_config_to_resources_config
from ._serialization import check_valid_cls_constructor_arg
from ._utils.async_utils import synchronize_api, synchronizer
from ._utils.grpc_utils import retry_transient_errors
from ._utils.mount_utils import validate_volumes
from .client import _Client
from .exception import InvalidError, NotFoundError, VersionError
from .functions import (
    _Function,
    _parse_retries,
)
from .gpu import GPU_T
from .object import _get_environment_name, _Object
from .partial_function import (
    _find_callables_for_cls,
    _find_callables_for_obj,
    _find_partial_methods_for_user_cls,
    _PartialFunction,
    _PartialFunctionFlags,
)
from .retries import Retries
from .secret import _Secret
from .volume import _Volume

T = TypeVar("T")


if typing.TYPE_CHECKING:
    import modal.app


class _Obj:
    """An instance of a `Cls`, i.e. `Cls("foo", 42)` returns an `Obj`.

    All this class does is to return `Function` objects."""

    _functions: Dict[str, _Function]
    _inited: bool
    _entered: bool
    _user_cls_instance: Optional[Any]
    _user_cls_instance_constr: Optional[Callable[[], Any]]

    _instance_service_function: Optional[_Function]

    def _uses_common_service_function(self):
        # Used for backwards compatibility checks with pre v0.63 classes
        return self._instance_service_function is not None

    def __init__(
        self,
        user_cls: type,
        output_mgr: Optional[OutputManager],
        class_service_function: Optional[_Function],  # only None for <v0.63 classes
        classbound_methods: Dict[str, _Function],
        from_other_workspace: bool,
        options: Optional[api_pb2.FunctionOptions],
        args,
        kwargs,
    ):
        for i, arg in enumerate(args):
            check_valid_cls_constructor_arg(i + 1, arg)
        for key, kwarg in kwargs.items():
            check_valid_cls_constructor_arg(key, kwarg)

        self._method_functions = {}
        if class_service_function:
            # >= v0.63 classes
            # first create the singular object function used by all methods on this parameterization
            self._instance_service_function = class_service_function._bind_parameters(
                self, from_other_workspace, options, args, kwargs
            )
            for method_name, class_bound_method in classbound_methods.items():
                method = self._instance_service_function._bind_instance_method(class_bound_method)
                method._set_output_mgr(output_mgr)
                self._method_functions[method_name] = method
        else:
            # <v0.63 classes - bind each individual method to the new parameters
            self._instance_service_function = None
            for method_name, class_bound_method in classbound_methods.items():
                method = class_bound_method._bind_parameters(self, from_other_workspace, options, args, kwargs)
                method._set_output_mgr(output_mgr)
                self._method_functions[method_name] = method

        # Used for construction local object lazily
        self._inited = False
        self._entered = False
        self._local_user_cls_instance = None

        if user_cls:
            self._user_cls_instance_constr = lambda: user_cls(*args, **kwargs)
        else:
            self._user_cls_instance_constr = None

    async def keep_warm(self, warm_pool_size: int) -> None:
        """Set the warm pool size for the class containers

        Please exercise care when using this advanced feature!
        Setting and forgetting a warm pool on functions can lead to increased costs.

        Note that all Modal methods and web endpoints of a class share the same set
        of containers and the warm_pool_size affects that common container pool.

        ```python
        # Usage on a parametrized function.
        Model = modal.Cls.lookup("my-app", "Model")
        Model("fine-tuned-model").keep_warm(2)
        ```
        """
        if not self._uses_common_service_function():
            raise VersionError(
                "Class instance `.keep_warm(...)` can't be used on classes deployed using client version <v0.63"
            )
        await self._instance_service_function.keep_warm(warm_pool_size)

    def _create_user_cls_instance(self) -> Any:
        """Constructs user cls instance without any caching and inserts hydrated methods

        Used by container entrypoint."""
        self._user_cls_instance = self._user_cls_instance_constr()
        setattr(
            self._user_cls_instance, "_modal_functions", self._method_functions
        )  # Needed for PartialFunction.__get__
        return self._user_cls_instance

    def _get_user_cls_instance(self):
        """Construct local object lazily. Used for .local() calls."""
        if not self._inited:
            self._create_user_cls_instance()  # Instantiate object
            self._inited = True

        return self._user_cls_instance

    def enter(self):
        if not self._entered:
            if hasattr(self._user_cls_instance, "__enter__"):
                self._user_cls_instance.__enter__()

            for method_flag in (
                _PartialFunctionFlags.ENTER_PRE_SNAPSHOT,
                _PartialFunctionFlags.ENTER_POST_SNAPSHOT,
            ):
                for enter_method in _find_callables_for_obj(self._user_cls_instance, method_flag).values():
                    enter_method()

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
            user_cls_instance = self._get_user_cls_instance()
            if hasattr(user_cls_instance, "__aenter__"):
                await user_cls_instance.__aenter__()
            elif hasattr(user_cls_instance, "__enter__"):
                user_cls_instance.__enter__()
        self.entered = True

    def __getattr__(self, k):
        if k in self._method_functions:
            return self._method_functions[k]
        elif self._user_cls_instance_constr:
            user_cls_instance = self._get_user_cls_instance()
            return getattr(user_cls_instance, k)
        else:
            raise AttributeError(k)


Obj = synchronize_api(_Obj)


class _Cls(_Object, type_prefix="cs"):
    _user_cls: Optional[type]
    _class_service_function: Optional[
        _Function
    ]  # The _Function serving *all* methods of the class, used for version >=v0.63
    _method_functions: Dict[str, _Function]  # Placeholder _Functions for each method
    _options: Optional[api_pb2.FunctionOptions]
    _callables: Dict[str, Callable]
    _from_other_workspace: Optional[bool]  # Functions require FunctionBindParams before invocation.
    _app: Optional["modal.app._App"] = None  # not set for lookups

    def _initialize_from_empty(self):
        self._user_cls = None
        self._class_service_function = None
        self._method_functions = {}
        self._options = None
        self._callables = {}
        self._from_other_workspace = None
        self._output_mgr: Optional[OutputManager] = None

    def _initialize_from_other(self, other: "_Cls"):
        self._user_cls = other._user_cls
        self._class_service_function = other._class_service_function
        self._method_functions = other._method_functions
        self._options = other._options
        self._callables = other._callables
        self._from_other_workspace = other._from_other_workspace
        self._output_mgr: Optional[OutputManager] = other._output_mgr

    def _set_output_mgr(self, output_mgr: OutputManager):
        self._output_mgr = output_mgr

    def _get_partial_functions(self) -> Dict[str, _PartialFunction]:
        if not self._user_cls:
            raise AttributeError("You can only get the partial functions of a local Cls instance")
        return _find_partial_methods_for_user_cls(self._user_cls, _PartialFunctionFlags.all())

    def _hydrate_metadata(self, metadata: Message):
        assert isinstance(metadata, api_pb2.ClassHandleMetadata)

        for method in metadata.methods:
            if method.function_name in self._method_functions:
                # This happens when the class is loaded locally
                # since each function will already be a loaded dependency _Function
                self._method_functions[method.function_name]._hydrate(
                    method.function_id, self._client, method.function_handle_metadata
                )
            else:
                self._method_functions[method.function_name] = _Function._new_hydrated(
                    method.function_id, self._client, method.function_handle_metadata
                )

    def _get_metadata(self) -> api_pb2.ClassHandleMetadata:
        class_handle_metadata = api_pb2.ClassHandleMetadata()
        for f_name, f in self._method_functions.items():
            class_handle_metadata.methods.append(
                api_pb2.ClassMethod(
                    function_name=f_name, function_id=f.object_id, function_handle_metadata=f._get_metadata()
                )
            )
        return class_handle_metadata

    @staticmethod
    def from_local(user_cls, app: "modal.app._App", class_service_function: _Function) -> "_Cls":
        """mdmd:hidden"""
        functions: Dict[str, _Function] = {}
        partial_functions: Dict[str, _PartialFunction] = _find_partial_methods_for_user_cls(
            user_cls, _PartialFunctionFlags.FUNCTION
        )

        for method_name, partial_function in partial_functions.items():
            method_function = class_service_function._bind_method(user_cls, method_name, partial_function)
            app._add_function(method_function, is_web_endpoint=partial_function.webhook_config is not None)
            partial_function.wrapped = True
            functions[method_name] = method_function

        # Disable the warning that these are not wrapped
        for partial_function in _find_partial_methods_for_user_cls(user_cls, ~_PartialFunctionFlags.FUNCTION).values():
            partial_function.wrapped = True

        # Get all callables
        callables: Dict[str, Callable] = _find_callables_for_cls(user_cls, ~_PartialFunctionFlags(0))

        def _deps() -> List[_Function]:
            return [class_service_function] + list(functions.values())

        async def _load(self: "_Cls", resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.ClassCreateRequest(app_id=resolver.app_id, existing_class_id=existing_object_id)
            for f_name, f in self._method_functions.items():
                req.methods.append(
                    api_pb2.ClassMethod(
                        function_name=f_name, function_id=f.object_id, function_handle_metadata=f._get_metadata()
                    )
                )
            resp = await resolver.client.stub.ClassCreate(req)
            self._hydrate(resp.class_id, resolver.client, resp.handle_metadata)

        rep = f"Cls({user_cls.__name__})"
        cls: _Cls = _Cls._from_loader(_load, rep, deps=_deps)
        cls._app = app
        cls._user_cls = user_cls
        cls._class_service_function = class_service_function
        cls._method_functions = functions
        cls._callables = callables
        cls._from_other_workspace = False
        return cls

    def _uses_common_service_function(self):
        # Used for backwards compatibility with version < 0.63
        # where methods had individual top level functions
        return self._class_service_function is not None

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
            _environment_name = _get_environment_name(environment_name, resolver)
            request = api_pb2.ClassGetRequest(
                app_name=app_name,
                object_tag=tag,
                namespace=namespace,
                environment_name=_environment_name,
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

            class_function_tag = f"{tag}.*"  # special name of the base service function for the class

            class_service_function = _Function.from_name(
                app_name,
                class_function_tag,
                environment_name=_environment_name,
            )
            try:
                obj._class_service_function = await resolver.load(class_service_function)
            except modal.exception.NotFoundError:
                # this happens when looking up classes deployed using <v0.63
                # This try-except block can be removed when min supported version >= 0.63
                pass

            obj._hydrate(response.class_id, resolver.client, response.handle_metadata)

        rep = f"Ref({app_name})"
        cls = cls._from_loader(_load_remote, rep, is_another_app=True)
        cls._from_other_workspace = bool(workspace is not None)
        return cls

    def with_options(
        self: "_Cls",
        cpu: Optional[float] = None,
        memory: Optional[Union[int, Tuple[int, int]]] = None,
        gpu: GPU_T = None,
        secrets: Collection[_Secret] = (),
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},
        retries: Optional[Union[int, Retries]] = None,
        timeout: Optional[int] = None,
        concurrency_limit: Optional[int] = None,
        allow_concurrent_inputs: Optional[int] = None,
        container_idle_timeout: Optional[int] = None,
        allow_background_volume_commits: Optional[bool] = None,
    ) -> "_Cls":
        """
        Beta: Allows for the runtime modification of a modal.Cls's configuration.

        This is a beta feature and may be unstable.

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
        retry_policy = _parse_retries(retries, f"Class {self.__name__}" if self._user_cls else "")
        if gpu or cpu or memory:
            resources = convert_fn_config_to_resources_config(cpu=cpu, memory=memory, gpu=gpu, ephemeral_disk=None)
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
            self._user_cls,
            self._output_mgr,
            self._class_service_function,
            self._method_functions,
            self._from_other_workspace,
            self._options,
            args,
            kwargs,
        )

    def __getattr__(self, k):
        # Used by CLI and container entrypoint
        if k in self._method_functions:
            return self._method_functions[k]
        return getattr(self._user_cls, k)


Cls = synchronize_api(_Cls)
