# Copyright Modal Labs 2022
import inspect
import os
import typing
from collections.abc import Collection
from typing import Any, Callable, Optional, TypeVar, Union

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal._utils.function_utils import CLASS_PARAM_TYPE_MAP
from modal_proto import api_pb2

from ._functions import _Function, _parse_retries
from ._object import _get_environment_name, _Object
from ._resolver import Resolver
from ._resources import convert_fn_config_to_resources_config
from ._serialization import check_valid_cls_constructor_arg
from ._traceback import print_server_warnings
from ._utils.async_utils import synchronize_api, synchronizer
from ._utils.deprecation import deprecation_warning, renamed_parameter
from ._utils.grpc_utils import retry_transient_errors
from ._utils.mount_utils import validate_volumes
from .client import _Client
from .exception import ExecutionError, InvalidError, NotFoundError, VersionError
from .gpu import GPU_T
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


def _use_annotation_parameters(user_cls: type) -> bool:
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


def _bind_instance_method(service_function: _Function, class_bound_method: _Function):
    """Binds an "instance service function" to a specific method name.
    This "dummy" _Function gets no unique object_id and isn't backend-backed at the moment, since all
    it does it forward invocations to the underlying instance_service_function with the specified method,
    and we don't support web_config for parametrized methods at the moment.
    """
    # TODO(elias): refactor to not use `_from_loader()` as a crutch for lazy-loading the
    #   underlying instance_service_function. It's currently used in order to take advantage
    #   of resolver logic and get "chained" resolution of lazy loads, even though this thin
    #   object itself doesn't need any "loading"
    assert service_function._obj
    method_name = class_bound_method._use_method_name

    def hydrate_from_instance_service_function(method_placeholder_fun):
        method_placeholder_fun._hydrate_from_other(service_function)
        method_placeholder_fun._obj = service_function._obj
        method_placeholder_fun._web_url = (
            class_bound_method._web_url
        )  # TODO: this shouldn't be set when actual parameters are used
        method_placeholder_fun._function_name = f"{class_bound_method._function_name}[parametrized]"
        method_placeholder_fun._is_generator = class_bound_method._is_generator
        method_placeholder_fun._cluster_size = class_bound_method._cluster_size
        method_placeholder_fun._use_method_name = method_name
        method_placeholder_fun._is_method = True

    async def _load(fun: "_Function", resolver: Resolver, existing_object_id: Optional[str]):
        # there is currently no actual loading logic executed to create each method on
        # the *parametrized* instance of a class - it uses the parameter-bound service-function
        # for the instance. This load method just makes sure to set all attributes after the
        # `service_function` has been loaded (it's in the `_deps`)
        hydrate_from_instance_service_function(fun)

    def _deps():
        if service_function.is_hydrated:
            # without this check, the common service_function will be reloaded by all methods
            # TODO(elias): Investigate if we can fix this multi-loader in the resolver - feels like a bug?
            return []
        return [service_function]

    rep = f"Method({method_name})"

    fun = _Function._from_loader(
        _load,
        rep,
        deps=_deps,
        hydrate_lazily=True,
    )
    if service_function.is_hydrated:
        # Eager hydration (skip load) if the instance service function is already loaded
        hydrate_from_instance_service_function(fun)

    fun._info = class_bound_method._info
    fun._obj = service_function._obj
    fun._is_method = True
    fun._app = class_bound_method._app
    fun._spec = class_bound_method._spec
    fun._is_web_endpoint = class_bound_method._is_web_endpoint
    return fun


class _Obj:
    """An instance of a `Cls`, i.e. `Cls("foo", 42)` returns an `Obj`.

    All this class does is to return `Function` objects."""

    _cls: "_Cls"  # parent
    _functions: dict[str, _Function]
    _has_entered: bool
    _user_cls_instance: Optional[Any] = None
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]

    _instance_service_function: Optional[_Function] = None  # this gets set lazily

    def _uses_common_service_function(self):
        # Used for backwards compatibility checks with pre v0.63 classes
        return self._cls._class_service_function is not None

    def __init__(
        self,
        cls: "_Cls",
        user_cls: Optional[type],  # this would be None in case of lookups
        options: Optional[api_pb2.FunctionOptions],
        args,
        kwargs,
    ):
        for i, arg in enumerate(args):
            check_valid_cls_constructor_arg(i + 1, arg)
        for key, kwarg in kwargs.items():
            check_valid_cls_constructor_arg(key, kwarg)
        self._cls = cls

        # Used for construction local object lazily
        self._has_entered = False
        self._user_cls = user_cls

        # used for lazy construction in case of explicit constructors
        self._args = args
        self._kwargs = kwargs
        self._options = options

    def _cached_service_function(self) -> "modal.functions._Function":
        # Returns a service function for this _Obj, serving all its methods
        # In case of methods without parameters or options, this is simply proxying to the class service function

        # only safe to call for 0.63+ classes (before then, all methods had their own services)
        if not self._instance_service_function:
            assert self._cls._class_service_function
            self._instance_service_function = self._cls._class_service_function._bind_parameters(
                self, self._options, self._args, self._kwargs
            )
        return self._instance_service_function

    def _get_parameter_values(self) -> dict[str, Any]:
        # binds args and kwargs according to the class constructor signature
        # (implicit by parameters or explicit)
        sig = _get_class_constructor_signature(self._user_cls)
        bound_vars = sig.bind(*self._args, **self._kwargs)
        bound_vars.apply_defaults()
        return bound_vars.arguments

    def _new_user_cls_instance(self):
        if not _use_annotation_parameters(self._user_cls):
            # TODO(elias): deprecate this code path eventually
            user_cls_instance = self._user_cls(*self._args, **self._kwargs)
        else:
            # ignore constructor (assumes there is no custom constructor,
            # which is guaranteed by _use_annotation_parameters)
            # set the attributes on the class corresponding to annotations
            # with = parameter() specifications
            param_values = self._get_parameter_values()
            user_cls_instance = self._user_cls.__new__(self._user_cls)  # new instance without running __init__
            user_cls_instance.__dict__.update(param_values)

        # TODO: always use Obj instances instead of making modifications to user cls
        # TODO: OR (if simpler for now) replace all the PartialFunctions on the user cls
        #   with getattr(self, method_name)

        # user cls instances are only created locally, so we have all partial functions available
        instance_methods = {}
        for method_name in _find_partial_methods_for_user_cls(self._user_cls, _PartialFunctionFlags.FUNCTION):
            instance_methods[method_name] = getattr(self, method_name)

        user_cls_instance._modal_functions = instance_methods
        return user_cls_instance

    async def keep_warm(self, warm_pool_size: int) -> None:
        """Set the warm pool size for the class containers

        Please exercise care when using this advanced feature!
        Setting and forgetting a warm pool on functions can lead to increased costs.

        Note that all Modal methods and web endpoints of a class share the same set
        of containers and the warm_pool_size affects that common container pool.

        ```python notest
        # Usage on a parametrized function.
        Model = modal.Cls.from_name("my-app", "Model")
        Model("fine-tuned-model").keep_warm(2)
        ```
        """
        if not self._uses_common_service_function():
            raise VersionError(
                "Class instance `.keep_warm(...)` can't be used on classes deployed using client version <v0.63"
            )
        await self._cached_service_function().keep_warm(warm_pool_size)

    def _cached_user_cls_instance(self):
        """Get or construct the local object

        Used for .local() calls and getting attributes of classes"""
        if not self._user_cls_instance:
            self._user_cls_instance = self._new_user_cls_instance()  # Instantiate object

        return self._user_cls_instance

    def _enter(self):
        assert self._user_cls
        if not self._has_entered:
            user_cls_instance = self._cached_user_cls_instance()
            if hasattr(user_cls_instance, "__enter__"):
                user_cls_instance.__enter__()

            for method_flag in (
                _PartialFunctionFlags.ENTER_PRE_SNAPSHOT,
                _PartialFunctionFlags.ENTER_POST_SNAPSHOT,
            ):
                for enter_method in _find_callables_for_obj(user_cls_instance, method_flag).values():
                    enter_method()

            self._has_entered = True

    @property
    def _entered(self) -> bool:
        # needed because _aenter is nowrap
        return self._has_entered

    @_entered.setter
    def _entered(self, val: bool):
        self._has_entered = val

    @synchronizer.nowrap
    async def _aenter(self):
        if not self._entered:  # use the property to get at the impl class
            user_cls_instance = self._cached_user_cls_instance()
            if hasattr(user_cls_instance, "__aenter__"):
                await user_cls_instance.__aenter__()
            elif hasattr(user_cls_instance, "__enter__"):
                user_cls_instance.__enter__()
        self._has_entered = True

    def __getattr__(self, k):
        # This is a bit messy and branchy because:
        # * Support for pre-0.63 lookups *and* newer classes
        # * Support .remote() on both hydrated (local or remote classes) or unhydrated classes (remote classes only)
        # * Support .local() on both hydrated and unhydrated classes (assuming local access to code)
        # * Support attribute access (when local cls is available)

        def _get_method_bound_function() -> Optional["_Function"]:
            """Gets _Function object for method - either for a local or a hydrated remote class

            * If class is neither local or hydrated - raise exception (should never happen)
            * If attribute isn't a method - return None
            """
            if self._cls._method_functions is None:
                raise ExecutionError("Method is not local and not hydrated")

            if class_bound_method := self._cls._method_functions.get(k, None):
                # If we know the user is accessing a *method* and not another attribute,
                # we don't have to create an instance of the user class yet.
                # This is because it might just be a call to `.remote()` on it which
                # doesn't require a local instance.
                # As long as we have the service function or params, we can do remote calls
                # without calling the constructor of the class in the calling context.
                if self._cls._class_service_function is None:
                    # a <v0.63 lookup
                    return class_bound_method._bind_parameters(self, self._options, self._args, self._kwargs)
                else:
                    return _bind_instance_method(self._cached_service_function(), class_bound_method)

            return None  # The attribute isn't a method

        if self._cls._method_functions is not None:
            # We get here with either a hydrated Cls or an unhydrated one with local definition
            if method := _get_method_bound_function():
                return method
            elif self._user_cls:
                # We have the local definition, and the attribute isn't a method
                # so we instantiate if we don't have an instance, and try to get the attribute
                user_cls_instance = self._cached_user_cls_instance()
                return getattr(user_cls_instance, k)
            else:
                # This is the case for a *hydrated* class without the local definition, i.e. a lookup
                # where the attribute isn't a registered method of the class
                raise NotFoundError(
                    f"Class has no method `{k}` and attributes (or undecorated methods) can't be accessed for"
                    f" remote classes (`Cls.from_name` instances)"
                )

        # Not hydrated Cls, and we don't have the class - typically a Cls.from_name that
        # has not yet been loaded. So use a special loader that loads it lazily:

        async def method_loader(fun, resolver: Resolver, existing_object_id):
            await resolver.load(self._cls)  # load class so we get info about methods
            method_function = _get_method_bound_function()
            if method_function is None:
                raise NotFoundError(
                    f"Class has no method {k}, and attributes can't be accessed for `Cls.from_name` instances"
                )
            await resolver.load(method_function)  # get the appropriate method handle (lazy)
            fun._hydrate_from_other(method_function)

        # The reason we don't *always* use this lazy loader is because it precludes attribute access
        # on local classes.
        return _Function._from_loader(
            method_loader,
            repr,
            deps=lambda: [],  # TODO: use cls as dep instead of loading inside method_loader?
            hydrate_lazily=True,
        )


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
    _method_functions: Optional[dict[str, _Function]] = None  # Placeholder _Functions for each method
    _options: Optional[api_pb2.FunctionOptions]
    _callables: dict[str, Callable[..., Any]]
    _app: Optional["modal.app._App"] = None  # not set for lookups
    _name: Optional[str]

    def _initialize_from_empty(self):
        self._user_cls = None
        self._class_service_function = None
        self._options = None
        self._callables = {}
        self._name = None

    def _initialize_from_other(self, other: "_Cls"):
        super()._initialize_from_other(other)
        self._user_cls = other._user_cls
        self._class_service_function = other._class_service_function
        self._method_functions = other._method_functions
        self._options = other._options
        self._callables = other._callables
        self._name = other._name

    def _get_partial_functions(self) -> dict[str, _PartialFunction]:
        if not self._user_cls:
            raise AttributeError("You can only get the partial functions of a local Cls instance")
        return _find_partial_methods_for_user_cls(self._user_cls, _PartialFunctionFlags.all())

    def _get_app(self) -> "modal.app._App":
        return self._app

    def _get_user_cls(self) -> type:
        return self._user_cls

    def _get_name(self) -> str:
        return self._name

    def _get_class_service_function(self) -> "modal.functions._Function":
        return self._class_service_function

    def _get_method_names(self) -> Collection[str]:
        # returns method names for a *local* class only for now (used by cli)
        return self._method_functions.keys()

    def _hydrate_metadata(self, metadata: Message):
        assert isinstance(metadata, api_pb2.ClassHandleMetadata)
        if (
            self._class_service_function
            and self._class_service_function._method_handle_metadata
            and len(self._class_service_function._method_handle_metadata)
        ):
            # The class only has a class service function and no method placeholders (v0.67+)
            if self._method_functions:
                # We're here when the Cls is loaded locally (e.g. _Cls.from_local) so the _method_functions mapping is
                # populated with (un-hydrated) _Function objects
                for (
                    method_name,
                    method_handle_metadata,
                ) in self._class_service_function._method_handle_metadata.items():
                    self._method_functions[method_name]._hydrate(
                        self._class_service_function.object_id, self._client, method_handle_metadata
                    )

            else:
                # We're here when the function is loaded remotely (e.g. _Cls.from_name)
                self._method_functions = {}
                for (
                    method_name,
                    method_handle_metadata,
                ) in self._class_service_function._method_handle_metadata.items():
                    self._method_functions[method_name] = _Function._new_hydrated(
                        self._class_service_function.object_id, self._client, method_handle_metadata
                    )
        elif self._class_service_function and self._class_service_function.object_id:
            # A class with a class service function and method placeholder functions
            self._method_functions = {}
            for method in metadata.methods:
                self._method_functions[method.function_name] = _Function._new_hydrated(
                    self._class_service_function.object_id, self._client, method.function_handle_metadata
                )
        else:
            # pre 0.63 class that does not have a class service function and only method functions
            self._method_functions = {}
            for method in metadata.methods:
                self._method_functions[method.function_name] = _Function._new_hydrated(
                    method.function_id, self._client, method.function_handle_metadata
                )

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

        method_functions: dict[str, _Function] = {}
        partial_functions: dict[str, _PartialFunction] = _find_partial_methods_for_user_cls(
            user_cls, _PartialFunctionFlags.FUNCTION
        )

        for method_name, partial_function in partial_functions.items():
            method_function = class_service_function._bind_method(user_cls, method_name, partial_function)
            if partial_function.webhook_config is not None:
                app._web_endpoints.append(method_function.tag)
            partial_function.wrapped = True
            method_functions[method_name] = method_function

        # Disable the warning that these are not wrapped
        for partial_function in _find_partial_methods_for_user_cls(user_cls, ~_PartialFunctionFlags.FUNCTION).values():
            partial_function.wrapped = True

        # Get all callables
        callables: dict[str, Callable] = {
            k: pf.raw_f for k, pf in _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.all()).items()
        }

        def _deps() -> list[_Function]:
            return [class_service_function]

        async def _load(self: "_Cls", resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.ClassCreateRequest(
                app_id=resolver.app_id, existing_class_id=existing_object_id, only_class_function=True
            )
            resp = await resolver.client.stub.ClassCreate(req)
            self._hydrate(resp.class_id, resolver.client, resp.handle_metadata)

        rep = f"Cls({user_cls.__name__})"
        cls: _Cls = _Cls._from_loader(_load, rep, deps=_deps)
        cls._app = app
        cls._user_cls = user_cls
        cls._class_service_function = class_service_function
        cls._method_functions = method_functions
        cls._callables = callables
        cls._name = user_cls.__name__
        return cls

    def _uses_common_service_function(self):
        # Used for backwards compatibility with version < 0.63
        # where methods had individual top level functions
        return self._class_service_function is not None

    @classmethod
    @renamed_parameter((2024, 12, 18), "tag", "name")
    def from_name(
        cls: type["_Cls"],
        app_name: str,
        name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> "_Cls":
        """Reference a Cls from a deployed App by its name.

        In contrast to `modal.Cls.lookup`, this is a lazy method
        that defers hydrating the local object with metadata from
        Modal servers until the first time it is actually used.

        ```python
        Model = modal.Cls.from_name("other-app", "Model")
        ```
        """

        async def _load_remote(obj: _Object, resolver: Resolver, existing_object_id: Optional[str]):
            _environment_name = _get_environment_name(environment_name, resolver)
            request = api_pb2.ClassGetRequest(
                app_name=app_name,
                object_tag=name,
                namespace=namespace,
                environment_name=_environment_name,
                lookup_published=workspace is not None,
                workspace_name=workspace,
                only_class_function=True,
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

            print_server_warnings(response.server_warnings)

            class_service_name = f"{name}.*"  # special name of the base service function for the class

            class_service_function = _Function.from_name(
                app_name,
                class_service_name,
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
        cls = cls._from_loader(_load_remote, rep, is_another_app=True, hydrate_lazily=True)
        # TODO: when pre 0.63 is phased out, we can set class_service_function here instead
        cls._name = name
        return cls

    def with_options(
        self: "_Cls",
        cpu: Optional[Union[float, tuple[float, float]]] = None,
        memory: Optional[Union[int, tuple[int, int]]] = None,
        gpu: GPU_T = None,
        secrets: Collection[_Secret] = (),
        volumes: dict[Union[str, os.PathLike], _Volume] = {},
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
        Model = modal.Cls.from_name("my_app", "Model")
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
    @renamed_parameter((2024, 12, 18), "tag", "name")
    async def lookup(
        app_name: str,
        name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> "_Cls":
        """Lookup a Cls from a deployed App by its name.

        DEPRECATED: This method is deprecated in favor of `modal.Cls.from_name`.

        In contrast to `modal.Cls.from_name`, this is an eager method
        that will hydrate the local object with metadata from Modal servers.

        ```python notest
        Model = modal.Cls.from_name("other-app", "Model")
        model = Model()
        model.inference(...)
        ```
        """
        deprecation_warning(
            (2025, 1, 27),
            "`modal.Cls.lookup` is deprecated and will be removed in a future release."
            " It can be replaced with `modal.Cls.from_name`."
            "\n\nSee https://modal.com/docs/guide/modal-1-0-migration for more information.",
        )
        obj = _Cls.from_name(
            app_name, name, namespace=namespace, environment_name=environment_name, workspace=workspace
        )
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

    @synchronizer.no_input_translation
    def __call__(self, *args, **kwargs) -> _Obj:
        """This acts as the class constructor."""
        return _Obj(
            self,
            self._user_cls,
            self._options,
            args,
            kwargs,
        )

    def __getattr__(self, k):
        # Used by CLI and container entrypoint
        # TODO: remove this method - access to attributes on classes should be discouraged
        if k in self._method_functions:
            deprecation_warning(
                (2025, 1, 13),
                "Usage of methods directly on the class will soon be deprecated, "
                "instantiate classes before using methods, e.g.:\n"
                f"{self._name}().{k} instead of {self._name}.{k}",
                pending=True,
            )
            return self._method_functions[k]
        return getattr(self._user_cls, k)

    def _is_local(self) -> bool:
        return self._user_cls is not None


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
