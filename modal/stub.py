# Copyright Modal Labs 2022
import inspect
import os
import typing
import warnings
from datetime import date
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

from synchronicity.async_wrap import asynccontextmanager

from modal._types import typechecked
from modal_utils.async_utils import synchronize_api, synchronizer

from ._function_utils import FunctionInfo
from ._ipython import is_notebook
from ._output import OutputManager
from ._resolver import Resolver
from .app import _container_app, _ContainerApp, _LocalApp, is_local
from .client import _Client
from .cls import _Cls
from .config import config, logger
from .exception import InvalidError, deprecation_error
from .functions import PartialFunction, _Function, _PartialFunction
from .gpu import GPU_T
from .image import _Image
from .mount import _Mount
from .network_file_system import _NetworkFileSystem
from .object import _Object
from .proxy import _Proxy
from .queue import _Queue
from .retries import Retries
from .runner import _run_stub
from .sandbox import _Sandbox
from .schedule import Schedule
from .secret import _Secret
from .volume import _Volume

_default_image: _Image = _Image.debian_slim()


class _LocalEntrypoint:
    _info: FunctionInfo
    _stub: "_Stub"

    def __init__(self, info, stub):
        self._info = info  # type: ignore
        self._stub = stub

    def __call__(self, *args, **kwargs):
        return self._info.raw_f(*args, **kwargs)

    @property
    def info(self) -> FunctionInfo:
        return self._info

    @property
    def stub(self) -> "_Stub":
        return self._stub


LocalEntrypoint = synchronize_api(_LocalEntrypoint)


def check_sequence(items: typing.Sequence[typing.Any], item_type: typing.Type[typing.Any], error_msg: str):
    if not isinstance(items, (list, tuple)):
        raise InvalidError(error_msg)
    if not all(isinstance(v, item_type) for v in items):
        raise InvalidError(error_msg)


CLS_T = typing.TypeVar("CLS_T", bound=typing.Type)


class _Stub:
    """A `Stub` is a description of how to create a Modal application.

    The stub object principally describes Modal objects (`Function`, `Image`,
    `Secret`, etc.) associated with the application. It has three responsibilities:

    * Syncing of identities across processes (your local Python interpreter and
      every Modal worker active in your application).
    * Making Objects stay alive and not be garbage collected for as long as the
      app lives (see App lifetime below).
    * Manage log collection for everything that happens inside your code.

    **Registering functions with an app**

    The most common way to explicitly register an Object with an app is through the
    `@stub.function()` decorator. It both registers the annotated function itself and
    other passed objects, like schedules and secrets, with the app:

    ```python
    import modal

    stub = modal.Stub()

    @stub.function(
        secret=modal.Secret.from_name("some_secret"),
        schedule=modal.Period(days=1),
    )
    def foo():
        pass
    ```

    In this example, the secret and schedule are registered with the app.
    """

    _name: Optional[str]
    _description: Optional[str]
    _blueprint: Dict[str, _Object]
    _function_mounts: Dict[str, _Mount]
    _mounts: Sequence[_Mount]
    _secrets: Sequence[_Secret]
    _web_endpoints: List[str]  # Used by the CLI
    _local_entrypoints: Dict[str, _LocalEntrypoint]
    _container_app: Optional[_ContainerApp]
    _local_app: Optional[_LocalApp]
    _all_stubs: ClassVar[Dict[str, List["_Stub"]]] = {}

    @typechecked
    def __init__(
        self,
        name: Optional[str] = None,
        *,
        image: Optional[_Image] = None,  # default image for all functions (default is `modal.Image.debian_slim()`)
        mounts: Sequence[_Mount] = [],  # default mounts for all functions
        secrets: Sequence[_Secret] = [],  # default secrets for all functions
        **blueprint: _Object,  # any Modal Object dependencies (Dict, Queue, etc.)
    ) -> None:
        """Construct a new app stub, optionally with default image, mounts, secrets

        Any "blueprint" objects are loaded as part of running or deploying the app,
        and are accessible by name on the running container app, e.g.:
        ```python
        stub = modal.Stub(key_value_store=modal.Dict.new())

        @stub.function()
        def store_something(key: str, value: str):
            stub.app.key_value_store.put(key, value)
        ```
        """

        self._name = name
        self._description = name

        check_sequence(mounts, _Mount, "mounts has to be a list or tuple of Mount/AioMount objects")
        check_sequence(secrets, _Secret, "secrets has to be a list or tuple of Secret/AioSecret objects")
        if image is not None and not isinstance(image, _Image):
            raise InvalidError("image has to be a modal Image or AioImage object")

        for k, v in blueprint.items():
            self._validate_blueprint_value(k, v)

        self._blueprint = blueprint
        if image is not None:
            self._blueprint["image"] = image  # backward compatibility since "image" used to be on the blueprint

        self._function_mounts = {}
        self._mounts = mounts
        self._secrets = secrets
        self._local_entrypoints = {}
        self._web_endpoints = []
        self._local_app = None  # when this is the launcher process
        self._container_app = None  # when this is inside a container

        string_name = self._name or ""

        if not is_local() and _container_app._stub_name == string_name:
            _container_app._associate_stub_container(self)
            # note that all stubs with the correct name will get the container app assigned
            self._container_app = _container_app

        _Stub._all_stubs.setdefault(string_name, []).append(self)

    @property
    def name(self) -> Optional[str]:
        """The user-provided name of the Stub."""
        return self._name

    @property
    def app(self):
        """`stub.app` is deprecated: use e.g. `stub.obj` instead of `stub.app.obj`
        if you need to access objects on the running app.
        """
        deprecation_error(date(2023, 9, 11), _Stub.app.__doc__)

    @property
    def app_id(self) -> Optional[str]:
        """Return the app_id, if the stub is running."""
        if self._container_app:
            return self._container_app._app_id
        elif self._local_app:
            return self._local_app._app_id
        else:
            return None

    @property
    def description(self) -> Optional[str]:
        """The Stub's `name`, if available, or a fallback descriptive identifier."""
        return self._description

    def set_description(self, description: str):
        self._description = description

    def _validate_blueprint_value(self, key: str, value: Any):
        if not isinstance(value, _Object):
            raise InvalidError(f"Stub attribute {key} with value {value} is not a valid Modal object")

    def _add_object(self, tag, obj):
        if self._container_app:
            # If this is inside a container, then objects can be defined after app initialization.
            # So we may have to initialize objects once they get bound to the stub.
            if self._container_app._has_object(tag):
                self._container_app._hydrate_object(obj, tag)

        self._blueprint[tag] = obj

    def __getitem__(self, tag: str):
        # Deprecated? Note: this is currently the only way to refer to lifecycled methods on the stub, since they have . in the tag
        return self._blueprint[tag]

    def __setitem__(self, tag: str, obj: _Object):
        self._validate_blueprint_value(tag, obj)
        # Deprecated ?
        self._add_object(tag, obj)

    def __getattr__(self, tag: str) -> _Object:
        assert isinstance(tag, str)
        if tag.startswith("__"):
            # Hacky way to avoid certain issues, e.g. pickle will try to look this up
            raise AttributeError(f"Stub has no member {tag}")
        # Return a reference to an object that will be created in the future
        return self._blueprint[tag]

    def __setattr__(self, tag: str, obj: _Object):
        # Note that only attributes defined in __annotations__ are set on the object itself,
        # everything else is registered on the blueprint
        if tag in self.__annotations__:
            object.__setattr__(self, tag, obj)
        else:
            self._validate_blueprint_value(tag, obj)
            self._add_object(tag, obj)

    def get_objects(self) -> List[Tuple[str, _Object]]:
        """Used by the container app to initialize objects."""
        return list(self._blueprint.items())

    def _uncreate_all_objects(self):
        # TODO(erikbern): this doesn't unhydrate objects that aren't tagged
        for obj in self._blueprint.values():
            obj._unhydrate()

    @typechecked
    def is_inside(self, image: Optional[_Image] = None) -> bool:
        """Returns if the program is currently running inside a container for this app."""
        if self._container_app is None:
            return False
        elif self._container_app != _container_app:
            return False
        elif image is None:
            # stub.app is set, which means we're inside this stub (no specific image)
            return True

        # We need to look up the image handle from the image provider
        assert isinstance(image, _Image)
        for tag, provider in self._blueprint.items():
            if provider == image:
                break
        else:
            raise InvalidError(
                inspect.cleandoc(
                    """`is_inside` only works for an image associated with an App. For instance:
                    stub.image = Image.debian_slim()
                    if stub.is_inside(stub.image):
                        print("I'm inside!")
                    """
                )
            )

        assert isinstance(image, _Image)
        return image._is_inside()

    @asynccontextmanager
    async def _set_local_app(self, app: _LocalApp) -> AsyncGenerator[None, None]:
        self._local_app = app
        try:
            yield
        finally:
            self._local_app = None

    @asynccontextmanager
    async def run(
        self,
        client: Optional[_Client] = None,
        stdout=None,
        show_progress: bool = True,
        detach: bool = False,
        output_mgr: Optional[OutputManager] = None,
    ) -> AsyncGenerator["_Stub", None]:
        """Context manager that runs an app on Modal.

        Use this as the main entry point for your Modal application. All calls
        to Modal functions should be made within the scope of this context
        manager, and they will correspond to the current app.

        Note that this method used to return a separate "App" object. This is
        no longer useful since you can use the stub itself for access to all
        objects. For backwards compatibility reasons, it returns the same stub.
        """
        # TODO(erikbern): deprecate this one too?
        async with _run_stub(self, client, stdout, show_progress, detach, output_mgr):
            yield self

    def _get_default_image(self):
        if "image" in self._blueprint:
            return self._blueprint["image"]
        else:
            return _default_image

    @property
    def _pty_input_stream(self):
        return self._blueprint.get("_pty_input_stream", None)

    def _add_pty_input_stream(self):
        if self._pty_input_stream:
            warnings.warn(
                "Running multiple interactive functions at the same time is not fully supported, and could lead to unexpected behavior."
            )
        else:
            self._blueprint["_pty_input_stream"] = _Queue.new()

    def _get_watch_mounts(self):
        all_mounts = [
            *self._mounts,
        ]
        for function in self.registered_functions.values():
            all_mounts.extend(function._all_mounts)

        return [m for m in all_mounts if m.is_local()]

    def _add_function(self, function: _Function):
        if function.tag in self._blueprint:
            old_function = self._blueprint[function.tag]
            if isinstance(old_function, _Function):
                if not is_notebook():
                    logger.warning(
                        f"Warning: Tag '{function.tag}' collision!"
                        f" Overriding existing function [{old_function._info.module_name}].{old_function._info.function_name}"
                        f" with new function [{function._info.module_name}].{function._info.function_name}"
                    )
            else:
                logger.warning(f"Warning: tag {function.tag} exists but is overridden by function")

        self._add_object(function.tag, function)

    @property
    def registered_functions(self) -> Dict[str, _Function]:
        """All modal.Function objects registered on the stub."""
        return {tag: obj for tag, obj in self._blueprint.items() if isinstance(obj, _Function)}

    @property
    def registered_classes(self) -> Dict[str, _Function]:
        """All modal.Cls objects registered on the stub."""
        return {tag: obj for tag, obj in self._blueprint.items() if isinstance(obj, _Cls)}

    @property
    def registered_entrypoints(self) -> Dict[str, _LocalEntrypoint]:
        """All local CLI entrypoints registered on the stub."""
        return self._local_entrypoints

    @property
    def registered_web_endpoints(self) -> List[str]:
        """Names of web endpoint (ie. webhook) functions registered on the stub."""
        return self._web_endpoints

    def local_entrypoint(
        self, _warn_parentheses_missing=None, *, name: Optional[str] = None
    ) -> Callable[[Callable[..., Any]], None]:
        """Decorate a function to be used as a CLI entrypoint for a Modal App.

        These functions can be used to define code that runs locally to set up the app,
        and act as an entrypoint to start Modal functions from. Note that regular
        Modal functions can also be used as CLI entrypoints, but unlike `local_entrypoint`,
        those functions are executed remotely directly.

        **Example**

        ```python
        @stub.local_entrypoint()
        def main():
            some_modal_function.call()
        ```

        You can call the function using `modal run` directly from the CLI:

        ```shell
        modal run stub_module.py
        ```

        Note that an explicit [`stub.run()`](/docs/reference/modal.Stub#run) is not needed, as an
        [app](/docs/guide/apps) is automatically created for you.

        **Multiple Entrypoints**

        If you have multiple `local_entrypoint` functions, you can qualify the name of your stub and function:

        ```shell
        modal run stub_module.py::stub.some_other_function
        ```

        **Parsing Arguments**

        If your entrypoint function take arguments with primitive types, `modal run` automatically parses them as
        CLI options. For example, the following function can be called with `modal run stub_module.py --foo 1 --bar "hello"`:

        ```python
        @stub.local_entrypoint()
        def main(foo: int, bar: str):
            some_modal_function.call(foo, bar)
        ```

        Currently, `str`, `int`, `float`, `bool`, and `datetime.datetime` are supported. Use `modal run stub_module.py --help` for more
        information on usage.

        """
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@stub.local_entrypoint()`.")
        if name is not None and not isinstance(name, str):
            raise InvalidError("Invalid value for `name`: Must be string.")

        def wrapped(raw_f: Callable[..., Any]) -> None:
            info = FunctionInfo(raw_f)
            tag = name if name is not None else raw_f.__qualname__
            entrypoint = self._local_entrypoints[tag] = _LocalEntrypoint(info, self)
            return entrypoint

        return wrapped

    @typechecked
    def function(
        self,
        _warn_parentheses_missing=None,
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[
            Union[str, os.PathLike], _NetworkFileSystem
        ] = {},  # Deprecated, use `network_file_systems` instead
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        allow_cross_region_volumes: bool = False,  # Whether using network file systems from other regions is allowed.
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},  # Experimental. Do not use!
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[
            int
        ] = None,  # An optional maximum number of concurrent containers running the function (use keep_warm for minimum).
        allow_concurrent_inputs: Optional[int] = None,  # Number of inputs the container may fetch to run concurrently.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode./
        keep_warm: Optional[
            int
        ] = None,  # An optional minimum number of containers to always keep warm (use concurrency_limit for maximum).
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[
            bool
        ] = None,  # Set this to True if it's a non-generator function returning a [sync/async] generator object
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    ) -> Callable[..., _Function]:
        """Decorator to register a new Modal function with this stub."""
        if isinstance(_warn_parentheses_missing, _Image):
            # Handle edge case where maybe (?) some users passed image as a positional arg
            raise InvalidError("`image` needs to be a positional argument: `@stub.function(image=image)`.")
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@stub.function()`.")

        if image is None:
            image = self._get_default_image()

        secrets = [*self._secrets, *secrets]

        if shared_volumes:
            deprecation_error(
                date(2023, 7, 5),
                "`shared_volumes` is deprecated. Use the argument `network_file_systems` instead.",
            )

        def wrapped(
            f: Union[_PartialFunction, Callable[..., Any]],
            _cls: Optional[type] = None,  # Used for methods only
            _auto_snapshot_enabled: Optional[bool] = None,  # Used for methods only
        ) -> _Function:
            is_generator_override: Optional[bool] = is_generator

            if isinstance(f, _PartialFunction):
                f.wrapped = True
                info = FunctionInfo(f.raw_f, serialized=serialized, name_override=name, cls=_cls)
                raw_f = f.raw_f
                webhook_config = f.webhook_config
                is_generator_override = f.is_generator
                if webhook_config:
                    self._web_endpoints.append(info.get_tag())
            else:
                info = FunctionInfo(f, serialized=serialized, name_override=name, cls=_cls)
                webhook_config = None
                raw_f = f

            if not _cls and not info.is_serialized() and "." in info.function_name:  # This is a method
                raise InvalidError(
                    "`stub.function` on methods is not allowed. See https://modal.com/docs/guide/lifecycle-functions instead"
                )

            info.get_tag()

            if is_generator_override is None:
                is_generator_override = inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)

            if interactive:
                self._add_pty_input_stream()

            function = _Function.from_args(
                info,
                stub=self,
                image=image,
                secret=secret,
                secrets=secrets,
                schedule=schedule,
                is_generator=is_generator_override,
                gpu=gpu,
                mounts=[*self._mounts, *mounts],
                network_file_systems=network_file_systems,
                allow_cross_region_volumes=allow_cross_region_volumes,
                volumes=volumes,
                memory=memory,
                proxy=proxy,
                retries=retries,
                concurrency_limit=concurrency_limit,
                allow_concurrent_inputs=allow_concurrent_inputs,
                container_idle_timeout=container_idle_timeout,
                timeout=timeout,
                cpu=cpu,
                interactive=interactive,
                keep_warm=keep_warm,
                name=name,
                cloud=cloud,
                webhook_config=webhook_config,
                cls=_cls,
                auto_snapshot_enabled=_auto_snapshot_enabled,
            )

            self._add_function(function)
            return function

        return wrapped

    def cls(
        self,
        _warn_parentheses_missing=None,
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[
            Union[str, os.PathLike], _NetworkFileSystem
        ] = {},  # Deprecated, use `network_file_systems` instead
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        allow_cross_region_volumes: bool = False,  # Whether using network file systems from other regions is allowed.
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},  # Experimental. Do not use!
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        allow_concurrent_inputs: Optional[int] = None,  # Number of inputs the container may fetch to run concurrently.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
        auto_snapshot_enabled: Optional[bool] = None,  # Whether to run and snapshot __enter__ as part of image build.
    ) -> Callable[[CLS_T], _Cls]:
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@stub.cls()`.")

        if auto_snapshot_enabled is None:
            auto_snapshot_enabled = config.get("auto_snapshot")

        decorator: Callable[[PartialFunction, type], _Function] = self.function(
            image=image,
            secret=secret,
            secrets=secrets,
            gpu=gpu,
            serialized=serialized,
            mounts=mounts,
            shared_volumes=shared_volumes,
            network_file_systems=network_file_systems,
            allow_cross_region_volumes=allow_cross_region_volumes,
            volumes=volumes,
            cpu=cpu,
            memory=memory,
            proxy=proxy,
            retries=retries,
            concurrency_limit=concurrency_limit,
            allow_concurrent_inputs=allow_concurrent_inputs,
            container_idle_timeout=container_idle_timeout,
            timeout=timeout,
            interactive=interactive,
            keep_warm=keep_warm,
            cloud=cloud,
        )

        def wrapper(user_cls: CLS_T) -> _Cls:
            partial_functions: Dict[str, PartialFunction] = {}
            functions: Dict[str, _Function] = {}

            for parent_cls in user_cls.mro():
                if parent_cls is object:
                    continue
                for k, v in parent_cls.__dict__.items():
                    if isinstance(v, PartialFunction):
                        partial_functions[k] = v
                        partial_function = synchronizer._translate_in(v)  # TODO: remove need for?
                        functions[k] = decorator(
                            partial_function,
                            user_cls,
                            auto_snapshot_enabled,
                        )

            tag: str = user_cls.__name__
            cls: _Cls = _Cls.from_local(user_cls, functions)
            self._add_object(tag, cls)
            return cls

        return wrapper

    def _get_deduplicated_function_mounts(self, mounts: Dict[str, _Mount]):
        cached_mounts = []
        for root_path, mount in mounts.items():
            if root_path not in self._function_mounts:
                self._function_mounts[root_path] = mount
            cached_mounts.append(self._function_mounts[root_path])
        return cached_mounts

    async def spawn_sandbox(
        self,
        *entrypoint_args: str,
        image: Optional[_Image] = None,  # The image to run as the container for the sandbox.
        mounts: Sequence[_Mount] = (),  # Mounts to attach to the sandbox.
        secrets: Sequence[_Secret] = (),  # Environment variables to inject into the sandbox.
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        timeout: Optional[int] = None,  # Maximum execution time of the sandbox in seconds.
        workdir: Optional[str] = None,  # Working directory of the sandbox.
        gpu: GPU_T = None,
        cloud: Optional[str] = None,
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
    ) -> _Sandbox:
        """Sandboxes are a way to run arbitrary commands in dynamically defined environments.

        This function returns a [SandboxHandle](/docs/reference/modal.Sandbox#modalsandboxsandbox), which can be used to interact with the running sandbox.

        Refer to the [docs](/docs/guide/sandbox) on how to spawn and use sandboxes.
        """
        from .sandbox import _Sandbox
        from .stub import _default_image

        if self._local_app:
            app_id = self._local_app.app_id
            environment_name = self._local_app._environment_name
            client = self._local_app.client
        elif self._container_app:
            app_id = self._container_app.app_id
            environment_name = self._container_app._environment_name
            client = self._container_app.client
        else:
            raise InvalidError("`stub.spawn_sandbox` requires a running app.")

        # TODO(erikbern): pulling a lot of app internals here, let's clean up shortly
        resolver = Resolver(client, environment_name=environment_name, app_id=app_id)
        obj = _Sandbox._new(
            entrypoint_args,
            image=image or _default_image,
            mounts=mounts,
            secrets=secrets,
            timeout=timeout,
            workdir=workdir,
            gpu=gpu,
            cloud=cloud,
            cpu=cpu,
            memory=memory,
            network_file_systems=network_file_systems,
        )
        await resolver.load(obj)
        return obj


Stub = synchronize_api(_Stub)
