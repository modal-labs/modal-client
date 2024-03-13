# Copyright Modal Labs 2022
import inspect
import os
import typing
from pathlib import PurePosixPath
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

from synchronicity.async_wrap import asynccontextmanager

from modal._types import typechecked

from ._ipython import is_notebook
from ._output import OutputManager
from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from ._utils.function_utils import FunctionInfo
from ._utils.mount_utils import validate_volumes
from .app import _container_app, _ContainerApp, _LocalApp, is_local
from .client import _Client
from .cls import _Cls
from .config import logger
from .exception import InvalidError, deprecation_error, deprecation_warning
from .functions import _Function
from .gpu import GPU_T
from .image import _Image
from .mount import _Mount
from .network_file_system import _NetworkFileSystem
from .object import _Object
from .partial_function import PartialFunction, _PartialFunction
from .proxy import _Proxy
from .retries import Retries
from .runner import _run_stub
from .sandbox import _Sandbox
from .schedule import Schedule
from .scheduler_placement import SchedulerPlacement
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
        secrets=[modal.Secret.from_name("some_secret")],
        schedule=modal.Period(days=1),
    )
    def foo():
        pass
    ```

    In this example, the secret and schedule are registered with the app.
    """

    _name: Optional[str]
    _description: Optional[str]
    _indexed_objects: Dict[str, _Object]
    _function_mounts: Dict[str, _Mount]
    _mounts: Sequence[_Mount]
    _secrets: Sequence[_Secret]
    _volumes: Dict[Union[str, PurePosixPath], _Volume]
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
        volumes: Dict[Union[str, PurePosixPath], _Volume] = {},  # default volumes for all functions
        **indexed_objects: _Object,  # any Modal Object dependencies (Dict, Queue, etc.)
    ) -> None:
        """Construct a new app stub, optionally with default image, mounts, secrets

        Any "indexed_objects" objects are loaded as part of running or deploying the app,
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

        check_sequence(mounts, _Mount, "mounts has to be a list or tuple of Mount objects")
        check_sequence(secrets, _Secret, "secrets has to be a list or tuple of Secret objects")
        validate_volumes(volumes)

        if image is not None and not isinstance(image, _Image):
            raise InvalidError("image has to be a modal Image or AioImage object")

        if indexed_objects:
            deprecation_warning(
                (2023, 12, 13),
                "Passing **kwargs to a stub is deprecated. In most cases, you can just define the objects in global scope.",
            )

        for k, v in indexed_objects.items():
            self._validate_blueprint_value(k, v)

        self._indexed_objects = indexed_objects
        if image is not None:
            self._indexed_objects["image"] = image  # backward compatibility since "image" used to be on the blueprint

        self._mounts = mounts

        self._secrets = secrets
        self._volumes = volumes
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
    def is_interactive(self) -> bool:
        """Whether the current app for the stub is running in interactive mode."""
        # return self._name
        if self._local_app:
            return self._local_app.is_interactive
        else:
            return False

    @property
    def app(self):
        """`stub.app` is deprecated: use e.g. `stub.obj` instead of `stub.app.obj`
        if you need to access objects on the running app.
        """
        deprecation_error((2023, 9, 11), _Stub.app.__doc__)

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

        self._indexed_objects[tag] = obj

    def __getitem__(self, tag: str):
        # Deprecated? Note: this is currently the only way to refer to lifecycled methods on the stub, since they have . in the tag
        return self._indexed_objects[tag]

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
        return self._indexed_objects[tag]

    def __setattr__(self, tag: str, obj: _Object):
        # Note that only attributes defined in __annotations__ are set on the object itself,
        # everything else is registered on the indexed_objects
        if tag in self.__annotations__:
            object.__setattr__(self, tag, obj)
        else:
            self._validate_blueprint_value(tag, obj)
            self._add_object(tag, obj)

    @property
    def image(self) -> _Image:
        # Exists to get the type inference working for `stub.image`
        # Will also keep this one after we remove [get/set][item/attr]
        return self._indexed_objects["image"]

    def get_objects(self) -> List[Tuple[str, _Object]]:
        """Used by the container app to initialize objects."""
        return list(self._indexed_objects.items())

    def _uncreate_all_objects(self):
        # TODO(erikbern): this doesn't unhydrate objects that aren't tagged
        for obj in self._indexed_objects.values():
            obj._unhydrate()

    @typechecked
    def is_inside(self, image: Optional[_Image] = None):
        """Deprecated: use `Image.imports()` instead! Usage:
        ```
        my_image = modal.Image.debian_slim().pip_install("torch")
        with my_image.imports():
            import torch
        ```
        """
        deprecation_error((2023, 11, 8), _Stub.is_inside.__doc__)

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
        if "image" in self._indexed_objects:
            return self._indexed_objects["image"]
        else:
            return _default_image

    def _get_watch_mounts(self):
        all_mounts = [
            *self._mounts,
        ]
        for function in self.registered_functions.values():
            all_mounts.extend(function._all_mounts)

        return [m for m in all_mounts if m.is_local()]

    def _add_function(self, function: _Function):
        if function.tag in self._indexed_objects:
            old_function = self._indexed_objects[function.tag]
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
        return {tag: obj for tag, obj in self._indexed_objects.items() if isinstance(obj, _Function)}

    @property
    def registered_classes(self) -> Dict[str, _Function]:
        """All modal.Cls objects registered on the stub."""
        return {tag: obj for tag, obj in self._indexed_objects.items() if isinstance(obj, _Cls)}

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
            some_modal_function.remote()
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
            if tag in self._local_entrypoints:
                # TODO: get rid of this limitation.
                raise InvalidError(f"Duplicate local entrypoint name: {tag}. Local entrypoint names must be unique.")
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
        secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),  # Modal Mounts added to the container
        network_file_systems: Dict[
            Union[str, PurePosixPath], _NetworkFileSystem
        ] = {},  # Mountpoints for Modal NetworkFileSystems
        volumes: Dict[Union[str, PurePosixPath], _Volume] = {},  # Mountpoints for Modal Volumes
        allow_cross_region_volumes: bool = False,  # Whether using network file systems from other regions is allowed.
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
        keep_warm: Optional[
            int
        ] = None,  # An optional minimum number of containers to always keep warm (use concurrency_limit for maximum).
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[
            bool
        ] = None,  # Set this to True if it's a non-generator function returning a [sync/async] generator object
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
        enable_memory_snapshot: bool = False,  # Enable memory checkpointing for faster cold starts.
        checkpointing_enabled: Optional[bool] = None,  # Deprecated
        block_network: bool = False,  # Whether to block network access
        max_inputs: Optional[
            int
        ] = None,  # Maximum number of inputs a container should handle before shutting down. With `max_inputs = 1`, containers will be single-use.
        # The next group of parameters are deprecated; do not use in any new code
        interactive: bool = False,  # Deprecated: use the `modal.interact()` hook instead
        secret: Optional[_Secret] = None,  # Deprecated: use `secrets`
        shared_volumes: Dict[
            Union[str, PurePosixPath], _NetworkFileSystem
        ] = {},  # Deprecated, use `network_file_systems` instead
        # Parameters below here are experimental. Use with caution!
        _allow_background_volume_commits: bool = False,  # Experimental flag
        _experimental_boost: bool = False,  # Experimental flag for lower latency function execution (alpha).
        _experimental_scheduler: bool = False,  # Experimental flag for more fine-grained scheduling (alpha).
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    ) -> Callable[..., _Function]:
        """Decorator to register a new Modal function with this stub."""
        if isinstance(_warn_parentheses_missing, _Image):
            # Handle edge case where maybe (?) some users passed image as a positional arg
            raise InvalidError("`image` needs to be a keyword argument: `@stub.function(image=image)`.")
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@stub.function()`.")

        if interactive:
            deprecation_error(
                (2024, 2, 29), "interactive=True has been deprecated. Set MODAL_INTERACTIVE_FUNCTIONS=1 instead."
            )

        if image is None:
            image = self._get_default_image()

        secrets = [*self._secrets, *secrets]

        if shared_volumes:
            deprecation_error(
                (2023, 7, 5),
                "`shared_volumes` is deprecated. Use the argument `network_file_systems` instead.",
            )

        def wrapped(
            f: Union[_PartialFunction, Callable[..., Any]],
            _cls: Optional[type] = None,  # Used for methods only
        ) -> _Function:
            nonlocal keep_warm, is_generator

            if isinstance(f, _PartialFunction):
                f.wrapped = True
                info = FunctionInfo(f.raw_f, serialized=serialized, name_override=name, cls=_cls)
                raw_f = f.raw_f
                webhook_config = f.webhook_config
                is_generator = f.is_generator
                keep_warm = f.keep_warm or keep_warm

                if webhook_config:
                    if interactive:
                        raise InvalidError("interactive=True is not supported with web endpoint functions")
                    self._web_endpoints.append(info.get_tag())
            else:
                info = FunctionInfo(f, serialized=serialized, name_override=name, cls=_cls)
                webhook_config = None
                raw_f = f

            if not _cls and not info.is_serialized() and "." in info.function_name:  # This is a method
                raise InvalidError(
                    "`stub.function` on methods is not allowed. See https://modal.com/docs/guide/lifecycle-functions instead"
                )

            if is_generator is None:
                is_generator = inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)

            function = _Function.from_args(
                info,
                stub=self,
                image=image,
                secret=secret,
                secrets=secrets,
                schedule=schedule,
                is_generator=is_generator,
                gpu=gpu,
                mounts=[*self._mounts, *mounts],
                network_file_systems=network_file_systems,
                allow_cross_region_volumes=allow_cross_region_volumes,
                volumes={**self._volumes, **volumes},
                memory=memory,
                proxy=proxy,
                retries=retries,
                concurrency_limit=concurrency_limit,
                allow_concurrent_inputs=allow_concurrent_inputs,
                container_idle_timeout=container_idle_timeout,
                timeout=timeout,
                cpu=cpu,
                keep_warm=keep_warm,
                cloud=cloud,
                webhook_config=webhook_config,
                enable_memory_snapshot=enable_memory_snapshot,
                checkpointing_enabled=checkpointing_enabled,
                allow_background_volume_commits=_allow_background_volume_commits,
                block_network=block_network,
                max_inputs=max_inputs,
                _experimental_boost=_experimental_boost,
                _experimental_scheduler=_experimental_scheduler,
                _experimental_scheduler_placement=_experimental_scheduler_placement,
            )

            self._add_function(function)
            return function

        return wrapped

    def cls(
        self,
        _warn_parentheses_missing=None,
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        network_file_systems: Dict[
            Union[str, PurePosixPath], _NetworkFileSystem
        ] = {},  # Mountpoints for Modal NetworkFileSystems
        volumes: Dict[Union[str, PurePosixPath], _Volume] = {},  # Mountpoints for Modal Volumes
        allow_cross_region_volumes: bool = False,  # Whether using network file systems from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        allow_concurrent_inputs: Optional[int] = None,  # Number of inputs the container may fetch to run concurrently.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
        enable_memory_snapshot: bool = False,  # Enable memory checkpointing for faster cold starts.
        checkpointing_enabled: Optional[bool] = None,  # Deprecated
        block_network: bool = False,  # Whether to block network access
        _allow_background_volume_commits: bool = False,
        max_inputs: Optional[
            int
        ] = None,  # Limits the number of inputs a container handles before shutting down. Use `max_inputs = 1` for single-use containers.
        # The next group of parameters are deprecated; do not use in any new code
        interactive: bool = False,  # Deprecated: use the `modal.interact()` hook instead
        secret: Optional[_Secret] = None,  # Deprecated: use `secrets`
        shared_volumes: Dict[
            Union[str, PurePosixPath], _NetworkFileSystem
        ] = {},  # Deprecated, use `network_file_systems` instead
        # Parameters below here are experimental. Use with caution!
        _experimental_boost: bool = False,  # Experimental flag for lower latency function execution (alpha).
        _experimental_scheduler: bool = False,  # Experimental flag for more fine-grained scheduling (alpha).
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    ) -> Callable[[CLS_T], _Cls]:
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@stub.cls()`.")

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
            enable_memory_snapshot=enable_memory_snapshot,
            checkpointing_enabled=checkpointing_enabled,
            block_network=block_network,
            _allow_background_volume_commits=_allow_background_volume_commits,
            max_inputs=max_inputs,
            _experimental_boost=_experimental_boost,
            _experimental_scheduler=_experimental_scheduler,
            _experimental_scheduler_placement=_experimental_scheduler_placement,
        )

        def wrapper(user_cls: CLS_T) -> _Cls:
            cls: _Cls = _Cls.from_local(user_cls, self, decorator)

            if len(cls._functions) > 1 and keep_warm is not None:
                deprecation_warning(
                    (2023, 10, 20),
                    "`@stub.cls(keep_warm=...)` is deprecated when there is more than 1 method."
                    " Use `@method(keep_warm=...)` on each method instead!",
                )

            tag: str = user_cls.__name__
            self._add_object(tag, cls)
            return cls

        return wrapper

    async def spawn_sandbox(
        self,
        *entrypoint_args: str,
        image: Optional[_Image] = None,  # The image to run as the container for the sandbox.
        mounts: Sequence[_Mount] = (),  # Mounts to attach to the sandbox.
        secrets: Sequence[_Secret] = (),  # Environment variables to inject into the sandbox.
        network_file_systems: Dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},
        timeout: Optional[int] = None,  # Maximum execution time of the sandbox in seconds.
        workdir: Optional[str] = None,  # Working directory of the sandbox.
        gpu: GPU_T = None,
        cloud: Optional[str] = None,
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        block_network: bool = False,  # Whether to block network access
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},  # Volumes to mount in the sandbox.
        _allow_background_volume_commits: bool = False,
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
            block_network=block_network,
            volumes=volumes,
            allow_background_volume_commits=_allow_background_volume_commits,
        )
        await resolver.load(obj)
        return obj

    def include(self, /, other_stub: "_Stub"):
        """Include another stub's objects in this one.

        Useful splitting up Modal apps across different self-contained files

        ```python
        stub_a = modal.Stub("a")
        @stub.function()
        def foo():
            ...

        stub_b = modal.Stub("b")
        @stub.function()
        def bar():
            ...

        stub_a.include(stub_b)

        @stub_a.local_entrypoint()
        def main():
            # use function declared on the included stub
            bar.remote()
        ```
        """
        for tag, object in other_stub._indexed_objects.items():
            existing_object = self._indexed_objects.get(tag)
            if existing_object and existing_object != object:
                logger.warning(
                    f"Named app object {tag} with existing value {existing_object} is being overwritten by a different object {object}"
                )

            self._add_object(tag, object)


Stub = synchronize_api(_Stub)
