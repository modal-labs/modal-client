# Copyright Modal Labs 2022
import inspect
import os
import typing
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Type, TypeVar, Union

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal._utils.function_utils import CLASS_PARAM_TYPE_MAP
from modal_proto import api_pb2

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


def _use_annotation_parameters(user_cls) -> bool:
    has_parameters = any(is_parameter(cls_member) for cls_member in user_cls.__dict__.values())
    has_explicit_constructor = user_cls.__init__ != object.__init__
    return has_parameters and not has_explicit_constructor


def _get_class_constructor_signature(user_cls: type) -> inspect.Signature:
    if not _use_annotation_parameters(user_cls):
        return inspect.signature(user_cls)
    else:
        constructor_parameters = []
        for name, annotation_value in user_cls.__dict__.get("__annotations__", {}).items():
            if hasattr(user_cls, name):
                parameter_spec = getattr(user_cls, name)
                if is_parameter(parameter_spec):
                    maybe_default = {}
                    if not isinstance(parameter_spec.default, _NO_DEFAULT):
                        maybe_default["default"] = parameter_spec.default

                    param = inspect.Parameter(
                        name=name,
                        annotation=annotation_value,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        **maybe_default,
                    )
                    constructor_parameters.append(param)

        return inspect.Signature(constructor_parameters)


class _Obj:
    """An instance of a `Cls`, i.e. `Cls("foo", 42)` returns an `Obj`.

    All this class does is to return `Function` objects."""

    _functions: Dict[str, _Function]
    _entered: bool
    _user_cls_instance: Optional[Any] = None
    _construction_args: Tuple[tuple, Dict[str, Any]]

    _instance_service_function: Optional[_Function]

    def _uses_common_service_function(self):
        # Used for backwards compatibility checks with pre v0.63 classes
        return self._instance_service_function is not None

    def __init__(
        self,
        user_cls: type,
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
                self._method_functions[method_name] = method
        else:
            # <v0.63 classes - bind each individual method to the new parameters
            self._instance_service_function = None
            for method_name, class_bound_method in classbound_methods.items():
                method = class_bound_method._bind_parameters(self, from_other_workspace, options, args, kwargs)
                self._method_functions[method_name] = method

        # Used for construction local object lazily
        self._entered = False
        self._local_user_cls_instance = None
        self._user_cls = user_cls
        self._construction_args = (args, kwargs)  # used for lazy construction in case of explicit constructors

    def _user_cls_instance_constr(self):
        args, kwargs = self._construction_args
        if not _use_annotation_parameters(self._user_cls):
            # TODO(elias): deprecate this code path eventually
            user_cls_instance = self._user_cls(*args, **kwargs)
        else:
            # set the attributes on the class corresponding to annotations
            # with = parameter() specifications
            sig = _get_class_constructor_signature(self._user_cls)
            bound_vars = sig.bind(*args, **kwargs)
            bound_vars.apply_defaults()
            user_cls_instance = self._user_cls.__new__(self._user_cls)  # new instance without running __init__
            user_cls_instance.__dict__.update(bound_vars.arguments)

        user_cls_instance._modal_functions = self._method_functions  # Needed for PartialFunction.__get__
        return user_cls_instance

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

    def _get_user_cls_instance(self):
        """Construct local object lazily. Used for .local() calls."""
        if not self._user_cls_instance:
            self._user_cls_instance = self._user_cls_instance_constr()  # Instantiate object

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
            # if we know the user is accessing a method, we don't have to create an instance
            # yet, since the user might just call `.remote()` on it which doesn't require
            # a local instance (in case __init__ does stuff that can't locally)
            return self._method_functions[k]
        elif self._user_cls_instance_constr:
            # if it's *not* a method
            # TODO: To get lazy loading (from_name) of classes to work, we need to avoid
            #  this path, otherwise local initialization will happen regardless if user
            #  only runs .remote(), since we don't know methods for the class until we
            #  load it
            user_cls_instance = self._get_user_cls_instance()
            return getattr(user_cls_instance, k)
        else:
            raise AttributeError(k)


Obj = synchronize_api(_Obj)


class _Cls(_Object, type_prefix="cs"):
    """
    Cls adds method pooling and [lifecycle hook](/docs/guide/lifecycle-functions) behavior
    to [modal.Function](/docs/reference/modal.Function).

    Generally, you will not construct a Cls directly.
    Instead, use the [`@app.cls()`](/docs/reference/modal.App#cls) decorator on the App object.
    """

    _user_cls: Optional[type]
    _class_service_function: Optional[
        _Function
    ]  # The _Function serving *all* methods of the class, used for version >=v0.63
    _method_functions: Dict[str, _Function]  # Placeholder _Functions for each method
    _options: Optional[api_pb2.FunctionOptions]
    _callables: Dict[str, Callable[..., Any]]
    _from_other_workspace: Optional[bool]  # Functions require FunctionBindParams before invocation.
    _app: Optional["modal.app._App"] = None  # not set for lookups

    def _initialize_from_empty(self):
        self._user_cls = None
        self._class_service_function = None
        self._method_functions = {}
        self._options = None
        self._callables = {}
        self._from_other_workspace = None

    def _initialize_from_other(self, other: "_Cls"):
        self._user_cls = other._user_cls
        self._class_service_function = other._class_service_function
        self._method_functions = other._method_functions
        self._options = other._options
        self._callables = other._callables
        self._from_other_workspace = other._from_other_workspace

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
    def validate_construction_mechanism(user_cls):
        """mdmd:hidden"""
        params = {k: v for k, v in user_cls.__dict__.items() if is_parameter(v)}
        has_custom_constructor = user_cls.__init__ != object.__init__
        if params and has_custom_constructor:
            raise InvalidError(
                "A class can't have both a custom __init__ constructor "
                "and dataclass-style modal.parameter() annotations"
            )

        annotations = user_cls.__dict__.get("__annotations__", {})  # compatible with older pythons
        missing_annotations = params.keys() - annotations.keys()
        if missing_annotations:
            raise InvalidError("All modal.parameter() specifications need to be type annotated")

        annotated_params = {k: t for k, t in annotations.items() if k in params}
        for k, t in annotated_params.items():
            if t not in CLASS_PARAM_TYPE_MAP:
                t_name = getattr(t, "__name__", repr(t))
                supported = ", ".join(t.__name__ for t in CLASS_PARAM_TYPE_MAP.keys())
                raise InvalidError(
                    f"{user_cls.__name__}.{k}: {t_name} is not a supported parameter type. Use one of: {supported}"
                )

    @staticmethod
    def from_local(user_cls, app: "modal.app._App", class_service_function: _Function) -> "_Cls":
        """mdmd:hidden"""
        # validate signature
        _Cls.validate_construction_mechanism(user_cls)

        functions: Dict[str, _Function] = {}
        partial_functions: Dict[str, _PartialFunction] = _find_partial_methods_for_user_cls(
            user_cls, _PartialFunctionFlags.FUNCTION
        )

        for method_name, partial_function in partial_functions.items():
            method_function = class_service_function._bind_method_old(user_cls, method_name, partial_function)
            app._add_function(method_function, is_web_endpoint=partial_function.webhook_config is not None)
            partial_function.wrapped = True
            functions[method_name] = method_function

        # Disable the warning that these are not wrapped
        for partial_function in _find_partial_methods_for_user_cls(user_cls, ~_PartialFunctionFlags.FUNCTION).values():
            partial_function.wrapped = True

        # Get all callables
        callables: Dict[str, Callable] = {
            k: pf.raw_f for k, pf in _find_partial_methods_for_user_cls(user_cls, ~_PartialFunctionFlags(0)).items()
        }

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
            # Even though we already have the function_handle_metadata for this method locally,
            # The RPC is going to replace it with function_handle_metadata derived from the server.
            # We need to overwrite the definition_id sent back from the server here with the definition_id
            # previously stored in function metadata, which may have been sent back from FunctionCreate.
            # The problem is that this metadata propagates back and overwrites the metadata on the Function
            # object itself. This is really messy. Maybe better to exclusively populate the method metadata
            # from the function metadata we already have locally? Really a lot to clean up here...
            for method in resp.handle_metadata.methods:
                f_metadata = self._method_functions[method.function_name]._get_metadata()
                method.function_handle_metadata.definition_id = f_metadata.definition_id
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
        tag: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> "_Cls":
        """Reference a Cls from a deployed App by its name.

        In contrast to `modal.Cls.lookup`, this is a lazy method
        that defers hydrating the local object with metadata from
        Modal servers until the first time it is actually used.

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
        cpu: Optional[Union[float, Tuple[float, float]]] = None,
        memory: Optional[Union[int, Tuple[int, int]]] = None,
        gpu: GPU_T = None,
        secrets: Collection[_Secret] = (),
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},
        retries: Optional[Union[int, Retries]] = None,
        timeout: Optional[int] = None,
        concurrency_limit: Optional[int] = None,
        allow_concurrent_inputs: Optional[int] = None,
        container_idle_timeout: Optional[int] = None,
    ) -> "_Cls":
        """
        **Beta:** Allows for the runtime modification of a modal.Cls's configuration.

        This is a beta feature and may be unstable.

        **Usage:**

        ```python notest
        Model = modal.Cls.lookup("my_app", "Model")
        ModelUsingGPU = Model.with_options(gpu="A100")
        ModelUsingGPU().generate.remote(42)  # will run with an A100 GPU
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
                allow_background_commits=True,
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
            target_concurrent_inputs=allow_concurrent_inputs,
        )

        return cls

    @staticmethod
    async def lookup(
        app_name: str,
        tag: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> "_Cls":
        """Lookup a Cls from a deployed App by its name.

        In contrast to `modal.Cls.from_name`, this is an eager method
        that will hydrate the local object with metadata from Modal servers.

        ```python
        Class = modal.Cls.lookup("other-app", "Class")
        obj = Class()
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


class _NO_DEFAULT:
    def __repr__(self):
        return "modal.cls._NO_DEFAULT()"


_no_default = _NO_DEFAULT()


class _Parameter:
    default: Any
    init: bool

    def __init__(self, default: Any, init: bool):
        self.default = default
        self.init = init

    def __get__(self, obj, obj_type=None) -> Any:
        if obj:
            if self.default is _no_default:
                raise AttributeError("field has no default value and no specified value")
            return self.default
        return self


def is_parameter(p: Any) -> bool:
    return isinstance(p, _Parameter) and p.init


def parameter(*, default: Any = _no_default, init: bool = True) -> Any:
    """Used to specify options for modal.cls parameters, similar to dataclass.field for dataclasses
    ```
    class A:
        a: str = modal.parameter()

    ```

    If `init=False` is specified, the field is not considered a parameter for the
    Modal class and not used in the synthesized constructor. This can be used to
    optionally annotate the type of a field that's used internally, for example values
    being set by @enter lifecycle methods, without breaking type checkers, but it has
    no runtime effect on the class.
    """
    # has to return Any to be assignable to any annotation (https://github.com/microsoft/pyright/issues/5102)
    return _Parameter(default=default, init=init)
