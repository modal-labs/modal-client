# Copyright Modal Labs 2022
import inspect
import typing
import warnings
from io import TextIOWrapper
from pathlib import PurePosixPath
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

from google.protobuf.message import Message
from synchronicity.async_wrap import asynccontextmanager

from modal_proto import api_pb2

from ._ipython import is_notebook
from ._output import OutputManager
from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from ._utils.function_utils import FunctionInfo
from ._utils.mount_utils import validate_volumes
from .app_utils import (  # noqa: F401
    _list_apps,
    list_apps,
)
from .client import _Client
from .cloud_bucket_mount import _CloudBucketMount
from .cls import _Cls
from .config import logger
from .exception import InvalidError, deprecation_error, deprecation_warning
from .functions import _Function
from .gpu import GPU_T
from .image import _Image
from .mount import _Mount
from .network_file_system import _NetworkFileSystem
from .object import _Object
from .partial_function import _find_callables_for_cls, _PartialFunction, _PartialFunctionFlags
from .proxy import _Proxy
from .retries import Retries
from .runner import _run_app
from .running_app import RunningApp
from .sandbox import _Sandbox
from .schedule import Schedule
from .scheduler_placement import SchedulerPlacement
from .secret import _Secret
from .volume import _Volume

_default_image: _Image = _Image.debian_slim()


class _LocalEntrypoint:
    _info: FunctionInfo
    _app: "_App"

    def __init__(self, info: FunctionInfo, app: "_App") -> None:
        self._info = info
        self._app = app

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._info.raw_f(*args, **kwargs)

    @property
    def info(self) -> FunctionInfo:
        return self._info

    @property
    def app(self) -> "_App":
        return self._app

    @property
    def stub(self) -> "_App":
        # Deprecated soon, only for backwards compatibility
        return self._app


LocalEntrypoint = synchronize_api(_LocalEntrypoint)


def check_sequence(items: typing.Sequence[typing.Any], item_type: typing.Type[typing.Any], error_msg: str) -> None:
    if not isinstance(items, (list, tuple)):
        raise InvalidError(error_msg)
    if not all(isinstance(v, item_type) for v in items):
        raise InvalidError(error_msg)


CLS_T = typing.TypeVar("CLS_T", bound=typing.Type)


class _App:
    """A Modal app (prior to April 2024 a "stub") is a group of functions and classes
    deployed together.

    The app serves at least three purposes:

    * A unit of deployment for functions and classes.
    * Syncing of identities of (primarily) functions and classes across processes
      (your local Python interpreter and every Modal containerr active in your application).
    * Manage log collection for everything that happens inside your code.

    **Registering functions with an app**

    The most common way to explicitly register an Object with an app is through the
    `@app.function()` decorator. It both registers the annotated function itself and
    other passed objects, like schedules and secrets, with the app:

    ```python
    import modal

    app = modal.App()  # Note: app were called "stub" up until April 2024

    @app.function(
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
    _image: Optional[_Image]
    _mounts: Sequence[_Mount]
    _secrets: Sequence[_Secret]
    _volumes: Dict[Union[str, PurePosixPath], _Volume]
    _web_endpoints: List[str]  # Used by the CLI
    _local_entrypoints: Dict[str, _LocalEntrypoint]
    _running_app: Optional[RunningApp]
    _client: Optional[_Client]
    _all_apps: ClassVar[Dict[Optional[str], List["_App"]]] = {}

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        image: Optional[_Image] = None,  # default image for all functions (default is `modal.Image.debian_slim()`)
        mounts: Sequence[_Mount] = [],  # default mounts for all functions
        secrets: Sequence[_Secret] = [],  # default secrets for all functions
        volumes: Dict[Union[str, PurePosixPath], _Volume] = {},  # default volumes for all functions
        **kwargs: _Object,  # DEPRECATED: passing additional objects to the stub as kwargs is no longer supported
    ) -> None:
        """Construct a new app, optionally with default image, mounts, secrets, or volumes.

        ```python notest
        image = modal.Image.debian_slim().pip_install(...)
        mount = modal.Mount.from_local_dir("./config")
        secret = modal.Secret.from_name("my-secret")
        volume = modal.Volume.from_name("my-data")
        app = modal.App(image=image, mounts=[mount], secrets=[secret], volumes={"/mnt/data": volume})
        ```
        """
        if name is not None and not isinstance(name, str):
            raise InvalidError("Invalid value for `name`: Must be string.")

        self._name = name
        self._description = name

        check_sequence(mounts, _Mount, "`mounts=` has to be a list or tuple of Mount objects")
        check_sequence(secrets, _Secret, "`secrets=` has to be a list or tuple of Secret objects")
        validate_volumes(volumes)

        if image is not None and not isinstance(image, _Image):
            raise InvalidError("image has to be a modal Image or AioImage object")

        if kwargs:
            deprecation_error(
                (2023, 12, 13),
                "Passing additional objects to the app constructor is deprecated."
                f" Please remove the following parameters from your app definition: {', '.join(kwargs)}."
                " In most cases, persistent (named) objects can just be defined in the global scope.",
            )

        for k, v in kwargs.items():
            self._validate_blueprint_value(k, v)

        self._indexed_objects = kwargs
        self._image = image
        self._mounts = mounts
        self._secrets = secrets
        self._volumes = volumes
        self._local_entrypoints = {}
        self._web_endpoints = []
        self._running_app = None  # Set inside container, OR during the time an app is running locally
        self._client = None

        # Register this app. This is used to look up the app in the container, when we can't get it from the function
        _App._all_apps.setdefault(self._name, []).append(self)

    @property
    def name(self) -> Optional[str]:
        """The user-provided name of the App."""
        return self._name

    @property
    def is_interactive(self) -> bool:
        """Whether the current app for the app is running in interactive mode."""
        # return self._name
        if self._running_app:
            return self._running_app.interactive
        else:
            return False

    @property
    def app_id(self) -> Optional[str]:
        """Return the app_id, if the app is running."""
        if self._running_app:
            return self._running_app.app_id
        else:
            return None

    @property
    def description(self) -> Optional[str]:
        """The App's `name`, if available, or a fallback descriptive identifier."""
        return self._description

    def set_description(self, description: str):
        self._description = description

    def _validate_blueprint_value(self, key: str, value: Any):
        if not isinstance(value, _Object):
            raise InvalidError(f"App attribute `{key}` with value {value!r} is not a valid Modal object")

    def _add_object(self, tag, obj):
        if self._running_app:
            # If this is inside a container, then objects can be defined after app initialization.
            # So we may have to initialize objects once they get bound to the app.
            if tag in self._running_app.tag_to_object_id:
                object_id: str = self._running_app.tag_to_object_id[tag]
                metadata: Message = self._running_app.object_handle_metadata[object_id]
                obj._hydrate(object_id, self._client, metadata)

        self._indexed_objects[tag] = obj

    def __getitem__(self, tag: str):
        """App assignments of the form `app.x` or `app["x"]` are deprecated!

        The only use cases for these assignments is in conjunction with `.new()`, which is now
        in itself deprecated. If you are constructing objects with `.from_name(...)`, there is no
        need to assign those objects to the app. Example:

        ```python
        d = modal.Dict.from_name("my-dict", create_if_missing=True)

        @app.function()
        def f(x, y):
            d[x] = y  # Refer to d in global scope
        ```
        """
        deprecation_error((2024, 3, 25), _App.__getitem__.__doc__)

    def __setitem__(self, tag: str, obj: _Object):
        deprecation_error((2024, 3, 25), _App.__getitem__.__doc__)

    def __getattr__(self, tag: str):
        # TODO(erikbern): remove this method later
        assert isinstance(tag, str)
        if tag.startswith("__"):
            # Hacky way to avoid certain issues, e.g. pickle will try to look this up
            raise AttributeError(f"App has no member {tag}")
        if tag not in self._indexed_objects:
            # Primarily to make hasattr work
            raise AttributeError(f"App has no member {tag}")
        deprecation_error((2024, 3, 25), _App.__getitem__.__doc__)

    def __setattr__(self, tag: str, obj: _Object):
        # TODO(erikbern): remove this method later
        # Note that only attributes defined in __annotations__ are set on the object itself,
        # everything else is registered on the indexed_objects
        if tag in self.__annotations__:
            object.__setattr__(self, tag, obj)
        elif tag == "image":
            self._image = obj
        else:
            deprecation_error((2024, 3, 25), _App.__getitem__.__doc__)

    @property
    def image(self) -> _Image:
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

    def _uncreate_all_objects(self):
        # TODO(erikbern): this doesn't unhydrate objects that aren't tagged
        for obj in self._indexed_objects.values():
            obj._unhydrate()

    def is_inside(self, image: Optional[_Image] = None):
        """Deprecated: use `Image.imports()` instead! Usage:
        ```
        my_image = modal.Image.debian_slim().pip_install("torch")
        with my_image.imports():
            import torch
        ```
        """
        deprecation_error((2023, 11, 8), _App.is_inside.__doc__)

    @asynccontextmanager
    async def _set_local_app(self, client: _Client, app: RunningApp) -> AsyncGenerator[None, None]:
        self._client = client
        self._running_app = app
        try:
            yield
        finally:
            self._client = None
            self._running_app = None

    @asynccontextmanager
    async def run(
        self,
        client: Optional[_Client] = None,
        stdout: Optional[TextIOWrapper] = None,
        show_progress: bool = True,
        detach: bool = False,
        output_mgr: Optional[OutputManager] = None,
    ) -> AsyncGenerator["_App", None]:
        """Context manager that runs an app on Modal.

        Use this as the main entry point for your Modal application. All calls
        to Modal functions should be made within the scope of this context
        manager, and they will correspond to the current app.

        Note that this method used to return a separate "App" object. This is
        no longer useful since you can use the app itself for access to all
        objects. For backwards compatibility reasons, it returns the same app.
        """
        # TODO(erikbern): deprecate this one too?
        async with _run_app(self, client, stdout, show_progress, detach, output_mgr):
            yield self

    def _get_default_image(self):
        if self._image:
            return self._image
        else:
            return _default_image

    def _get_watch_mounts(self):
        all_mounts = [
            *self._mounts,
        ]
        for function in self.registered_functions.values():
            all_mounts.extend(function._all_mounts)

        return [m for m in all_mounts if m.is_local()]

    def _add_function(self, function: _Function, is_web_endpoint: bool):
        if function.tag in self._indexed_objects:
            old_function = self._indexed_objects[function.tag]
            if isinstance(old_function, _Function):
                if not is_notebook():
                    logger.warning(
                        f"Warning: Tag '{function.tag}' collision!"
                        " Overriding existing function "
                        f"[{old_function._info.module_name}].{old_function._info.function_name}"
                        f" with new function [{function._info.module_name}].{function._info.function_name}"
                    )
            else:
                logger.warning(f"Warning: tag {function.tag} exists but is overridden by function")

        self._add_object(function.tag, function)
        if is_web_endpoint:
            self._web_endpoints.append(function.tag)

    def _init_container(self, client: _Client, running_app: RunningApp):
        self._client = client
        self._running_app = running_app

        # Hydrate objects on app
        for tag, object_id in running_app.tag_to_object_id.items():
            if tag in self._indexed_objects:
                obj = self._indexed_objects[tag]
                handle_metadata = running_app.object_handle_metadata[object_id]
                obj._hydrate(object_id, client, handle_metadata)

    @property
    def registered_functions(self) -> Dict[str, _Function]:
        """All modal.Function objects registered on the app."""
        return {tag: obj for tag, obj in self._indexed_objects.items() if isinstance(obj, _Function)}

    @property
    def registered_classes(self) -> Dict[str, _Function]:
        """All modal.Cls objects registered on the app."""
        return {tag: obj for tag, obj in self._indexed_objects.items() if isinstance(obj, _Cls)}

    @property
    def registered_entrypoints(self) -> Dict[str, _LocalEntrypoint]:
        """All local CLI entrypoints registered on the app."""
        return self._local_entrypoints

    @property
    def indexed_objects(self) -> Dict[str, _Object]:
        return self._indexed_objects

    @property
    def registered_web_endpoints(self) -> List[str]:
        """Names of web endpoint (ie. webhook) functions registered on the app."""
        return self._web_endpoints

    def local_entrypoint(
        self, _warn_parentheses_missing: Any = None, *, name: Optional[str] = None
    ) -> Callable[[Callable[..., Any]], None]:
        """Decorate a function to be used as a CLI entrypoint for a Modal App.

        These functions can be used to define code that runs locally to set up the app,
        and act as an entrypoint to start Modal functions from. Note that regular
        Modal functions can also be used as CLI entrypoints, but unlike `local_entrypoint`,
        those functions are executed remotely directly.

        **Example**

        ```python
        @app.local_entrypoint()
        def main():
            some_modal_function.remote()
        ```

        You can call the function using `modal run` directly from the CLI:

        ```shell
        modal run app_module.py
        ```

        Note that an explicit [`app.run()`](/docs/reference/modal.App#run) is not needed, as an
        [app](/docs/guide/apps) is automatically created for you.

        **Multiple Entrypoints**

        If you have multiple `local_entrypoint` functions, you can qualify the name of your app and function:

        ```shell
        modal run app_module.py::app.some_other_function
        ```

        **Parsing Arguments**

        If your entrypoint function take arguments with primitive types, `modal run` automatically parses them as
        CLI options.
        For example, the following function can be called with `modal run app_module.py --foo 1 --bar "hello"`:

        ```python
        @app.local_entrypoint()
        def main(foo: int, bar: str):
            some_modal_function.call(foo, bar)
        ```

        Currently, `str`, `int`, `float`, `bool`, and `datetime.datetime` are supported.
        Use `modal run app_module.py --help` for more information on usage.

        """
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@app.local_entrypoint()`.")
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

    def function(
        self,
        _warn_parentheses_missing: Any = None,
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
        volumes: Dict[
            Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes & CloudBucketMounts
        allow_cross_region_volumes: bool = False,  # Whether using network file systems from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, Tuple[int, int]]] = None,
        ephemeral_disk: Optional[int] = None,  # Specify, in MiB, the ephemeral disk size for the Function.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[
            int
        ] = None,  # An optional maximum number of concurrent containers running the function (keep_warm sets minimum).
        allow_concurrent_inputs: Optional[int] = None,  # Number of inputs the container may fetch to run concurrently.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        keep_warm: Optional[
            int
        ] = None,  # An optional minimum number of containers to always keep warm (use concurrency_limit for maximum).
        name: Optional[str] = None,  # Sets the Modal name of the function within the app
        is_generator: Optional[
            bool
        ] = None,  # Set this to True if it's a non-generator function returning a [sync/async] generator object
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
        region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
        enable_memory_snapshot: bool = False,  # Enable memory checkpointing for faster cold starts.
        checkpointing_enabled: Optional[bool] = None,  # Deprecated
        block_network: bool = False,  # Whether to block network access
        # Maximum number of inputs a container should handle before shutting down.
        # With `max_inputs = 1`, containers will be single-use.
        max_inputs: Optional[int] = None,
        # The next group of parameters are deprecated; do not use in any new code
        interactive: bool = False,  # Deprecated: use the `modal.interact()` hook instead
        secret: Optional[_Secret] = None,  # Deprecated: use `secrets`
        # Parameters below here are experimental. Use with caution!
        _allow_background_volume_commits: Optional[bool] = None,
        _experimental_boost: bool = False,  # Experimental flag for lower latency function execution (alpha).
        _experimental_scheduler: bool = False,  # Experimental flag for more fine-grained scheduling (alpha).
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    ) -> Callable[..., _Function]:
        """Decorator to register a new Modal function with this app."""
        if isinstance(_warn_parentheses_missing, _Image):
            # Handle edge case where maybe (?) some users passed image as a positional arg
            raise InvalidError("`image` needs to be a keyword argument: `@app.function(image=image)`.")
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@app.function()`.")

        if interactive:
            deprecation_error(
                (2024, 5, 1), "interactive=True has been deprecated. Set MODAL_INTERACTIVE_FUNCTIONS=1 instead."
            )

        if image is None:
            image = self._get_default_image()

        secrets = [*self._secrets, *secrets]

        def wrapped(
            f: Union[_PartialFunction, Callable[..., Any], None],
        ) -> _Function:
            nonlocal keep_warm, is_generator

            # Check if the decorated object is a class
            if inspect.isclass(f):
                raise TypeError("The @app.function decorator cannot be used on a class. Please use @app.cls instead.")

            if isinstance(f, _PartialFunction):
                # typically for @function-wrapped @web_endpoint and @asgi_app
                f.wrapped = True
                info = FunctionInfo(f.raw_f, serialized=serialized, name_override=name)
                raw_f = f.raw_f
                webhook_config = f.webhook_config
                is_generator = f.is_generator
                keep_warm = f.keep_warm or keep_warm

                if webhook_config and interactive:
                    raise InvalidError("interactive=True is not supported with web endpoint functions")
            else:
                info = FunctionInfo(f, serialized=serialized, name_override=name)
                webhook_config = None
                raw_f = f

            if info.function_name.endswith(".app"):
                warnings.warn(
                    "Beware: the function name is `app`. Modal will soon rename `Stub` to `App`, "
                    "so you might run into issues if you have code like `app = modal.App()` in the same scope"
                )

            if is_generator is None:
                is_generator = inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)

            scheduler_placement: Optional[SchedulerPlacement] = _experimental_scheduler_placement
            if region:
                if scheduler_placement:
                    raise InvalidError("`region` and `_experimental_scheduler_placement` cannot be used together")
                scheduler_placement = SchedulerPlacement(region=region)

            function = _Function.from_args(
                info,
                app=self,
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
                cpu=cpu,
                memory=memory,
                ephemeral_disk=ephemeral_disk,
                proxy=proxy,
                retries=retries,
                concurrency_limit=concurrency_limit,
                allow_concurrent_inputs=allow_concurrent_inputs,
                container_idle_timeout=container_idle_timeout,
                timeout=timeout,
                keep_warm=keep_warm,
                cloud=cloud,
                webhook_config=webhook_config,
                enable_memory_snapshot=enable_memory_snapshot,
                checkpointing_enabled=checkpointing_enabled,
                allow_background_volume_commits=_allow_background_volume_commits,
                block_network=block_network,
                max_inputs=max_inputs,
                scheduler_placement=scheduler_placement,
                _experimental_boost=_experimental_boost,
                _experimental_scheduler=_experimental_scheduler,
            )

            self._add_function(function, webhook_config is not None)
            return function

        return wrapped

    def cls(
        self,
        _warn_parentheses_missing: Optional[bool] = None,
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        network_file_systems: Dict[
            Union[str, PurePosixPath], _NetworkFileSystem
        ] = {},  # Mountpoints for Modal NetworkFileSystems
        volumes: Dict[
            Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes & CloudBucketMounts
        allow_cross_region_volumes: bool = False,  # Whether using network file systems from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, Tuple[int, int]]] = None,
        ephemeral_disk: Optional[int] = None,  # Specify, in MiB, the ephemeral disk size for the Function.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        allow_concurrent_inputs: Optional[int] = None,  # Number of inputs the container may fetch to run concurrently.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
        region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
        enable_memory_snapshot: bool = False,  # Enable memory checkpointing for faster cold starts.
        checkpointing_enabled: Optional[bool] = None,  # Deprecated
        block_network: bool = False,  # Whether to block network access
        _allow_background_volume_commits: Optional[bool] = None,
        # Limits the number of inputs a container handles before shutting down.
        # Use `max_inputs = 1` for single-use containers.
        max_inputs: Optional[int] = None,
        # The next group of parameters are deprecated; do not use in any new code
        interactive: bool = False,  # Deprecated: use the `modal.interact()` hook instead
        secret: Optional[_Secret] = None,  # Deprecated: use `secrets`
        # Parameters below here are experimental. Use with caution!
        _experimental_boost: bool = False,  # Experimental flag for lower latency function execution (alpha).
        _experimental_scheduler: bool = False,  # Experimental flag for more fine-grained scheduling (alpha).
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    ) -> Callable[[CLS_T], _Cls]:
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@app.cls()`.")

        if interactive:
            deprecation_error(
                (2024, 5, 1), "interactive=True has been deprecated. Set MODAL_INTERACTIVE_FUNCTIONS=1 instead."
            )

        if image is None:
            image = self._get_default_image()

        secrets = [*self._secrets, *secrets]

        def wrapper(user_cls: CLS_T) -> _Cls:
            nonlocal keep_warm

            # Check if the decorated object is a class
            if not inspect.isclass(user_cls):
                raise TypeError("The @app.cls decorator must be used on a class.")

            info = FunctionInfo(None, serialized=serialized, cls=user_cls)

            scheduler_placement: Optional[SchedulerPlacement] = _experimental_scheduler_placement
            if region:
                if scheduler_placement:
                    raise InvalidError("`region` and `_experimental_scheduler_placement` cannot be used together")
                scheduler_placement = SchedulerPlacement(region=region)

            cls_func = _Function.from_args(
                info,
                app=self,
                image=image,
                secret=secret,
                secrets=secrets,
                gpu=gpu,
                mounts=[*self._mounts, *mounts],
                network_file_systems=network_file_systems,
                allow_cross_region_volumes=allow_cross_region_volumes,
                volumes={**self._volumes, **volumes},
                memory=memory,
                ephemeral_disk=ephemeral_disk,
                proxy=proxy,
                retries=retries,
                concurrency_limit=concurrency_limit,
                allow_concurrent_inputs=allow_concurrent_inputs,
                container_idle_timeout=container_idle_timeout,
                timeout=timeout,
                cpu=cpu,
                keep_warm=keep_warm,
                cloud=cloud,
                enable_memory_snapshot=enable_memory_snapshot,
                checkpointing_enabled=checkpointing_enabled,
                allow_background_volume_commits=_allow_background_volume_commits,
                block_network=block_network,
                max_inputs=max_inputs,
                scheduler_placement=scheduler_placement,
                _experimental_boost=_experimental_boost,
                _experimental_scheduler=_experimental_scheduler,
                # class service function, so the following attributes which relate to
                # the callable itself are invalid and set to defaults:
                webhook_config=None,
                is_generator=False,
            )

            self._add_function(cls_func, is_web_endpoint=False)

            cls: _Cls = _Cls.from_local(user_cls, self, cls_func)

            if (
                _find_callables_for_cls(user_cls, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
                and not enable_memory_snapshot
            ):
                raise InvalidError("A class must have `enable_memory_snapshot=True` to use `snap=True` on its methods.")

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
        region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the sandbox on.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, Tuple[int, int]]] = None,
        block_network: bool = False,  # Whether to block network access
        volumes: Dict[
            Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes and CloudBucketMounts
        _allow_background_volume_commits: Optional[bool] = None,
        pty_info: Optional[api_pb2.PTYInfo] = None,
        _experimental_scheduler: bool = False,  # Experimental flag for more fine-grained scheduling (alpha).
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    ) -> _Sandbox:
        """Sandboxes are a way to run arbitrary commands in dynamically defined environments.

        This function returns a [SandboxHandle](/docs/reference/modal.Sandbox#modalsandboxsandbox),
        which can be used to interact with the running sandbox.

        Refer to the [docs](/docs/guide/sandbox) on how to spawn and use sandboxes.
        """
        if self._running_app:
            app_id = self._running_app.app_id
            environment_name = self._running_app.environment_name
            client = self._client
        else:
            raise InvalidError("`app.spawn_sandbox` requires a running app.")

        if _allow_background_volume_commits is False:
            deprecation_warning(
                (2024, 5, 13),
                "Disabling volume background commits is now deprecated. Set _allow_background_volume_commits=True.",
            )
        elif _allow_background_volume_commits is None:
            _allow_background_volume_commits = True

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
            region=region,
            cpu=cpu,
            memory=memory,
            network_file_systems=network_file_systems,
            block_network=block_network,
            volumes=volumes,
            allow_background_volume_commits=_allow_background_volume_commits,
            pty_info=pty_info,
            _experimental_scheduler=_experimental_scheduler,
            _experimental_scheduler_placement=_experimental_scheduler_placement,
        )
        await resolver.load(obj)
        return obj

    def include(self, /, other_app: "_App"):
        """Include another app's objects in this one.

        Useful splitting up Modal apps across different self-contained files

        ```python
        app_a = modal.App("a")
        @app.function()
        def foo():
            ...

        app_b = modal.App("b")
        @app.function()
        def bar():
            ...

        app_a.include(app_b)

        @app_a.local_entrypoint()
        def main():
            # use function declared on the included app
            bar.remote()
        ```
        """
        for tag, object in other_app._indexed_objects.items():
            existing_object = self._indexed_objects.get(tag)
            if existing_object and existing_object != object:
                logger.warning(
                    f"Named app object {tag} with existing value {existing_object} is being "
                    f"overwritten by a different object {object}"
                )

            self._add_object(tag, object)


App = synchronize_api(_App)


class _Stub(_App):
    """This enables using an "Stub" class instead of "App".

    For most of Modal's history, the app class was called "Stub", so this exists for
    backwards compatibility, in order to facilitate moving from "Stub" to "App".
    """

    def __new__(cls, *args, **kwargs):
        deprecation_warning(
            (2024, 4, 29),
            'The use of "Stub" has been deprecated in favor of "App".'
            " This is a pure name change with no other implications.",
        )
        return _App(*args, **kwargs)


Stub = synchronize_api(_Stub)
