# Copyright Modal Labs 2022
import inspect
import typing
import warnings
from collections.abc import AsyncGenerator, Coroutine, Sequence
from pathlib import PurePosixPath
from textwrap import dedent
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
    Union,
    overload,
)

import typing_extensions
from google.protobuf.message import Message
from synchronicity.async_wrap import asynccontextmanager

from modal_proto import api_pb2

from ._functions import _Function
from ._ipython import is_notebook
from ._object import _get_environment_name, _Object
from ._utils.async_utils import synchronize_api
from ._utils.deprecation import deprecation_error, deprecation_warning, renamed_parameter
from ._utils.function_utils import FunctionInfo, is_global_object, is_method_fn
from ._utils.grpc_utils import retry_transient_errors
from ._utils.mount_utils import validate_volumes
from .client import _Client
from .cloud_bucket_mount import _CloudBucketMount
from .cls import _Cls, parameter
from .config import logger
from .exception import ExecutionError, InvalidError
from .functions import Function
from .gpu import GPU_T
from .image import _Image
from .mount import _Mount
from .network_file_system import _NetworkFileSystem
from .partial_function import (
    PartialFunction,
    _find_partial_methods_for_user_cls,
    _PartialFunction,
    _PartialFunctionFlags,
)
from .proxy import _Proxy
from .retries import Retries
from .app_layout import AppLayout
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


def check_sequence(items: typing.Sequence[typing.Any], item_type: type[typing.Any], error_msg: str) -> None:
    if not isinstance(items, (list, tuple)):
        raise InvalidError(error_msg)
    if not all(isinstance(v, item_type) for v in items):
        raise InvalidError(error_msg)


CLS_T = typing.TypeVar("CLS_T", bound=type[Any])


P = typing_extensions.ParamSpec("P")
ReturnType = typing.TypeVar("ReturnType")
OriginalReturnType = typing.TypeVar("OriginalReturnType")


class _FunctionDecoratorType:
    @overload
    def __call__(
        self, func: PartialFunction[P, ReturnType, OriginalReturnType]
    ) -> Function[P, ReturnType, OriginalReturnType]:
        ...  # already wrapped by a modal decorator, e.g. web_endpoint

    @overload
    def __call__(
        self, func: Callable[P, Coroutine[Any, Any, ReturnType]]
    ) -> Function[P, ReturnType, Coroutine[Any, Any, ReturnType]]:
        ...  # decorated async function

    @overload
    def __call__(self, func: Callable[P, ReturnType]) -> Function[P, ReturnType, ReturnType]:
        ...  # decorated non-async function

    def __call__(self, func):
        ...


class _App:
    """A Modal App is a group of functions and classes that are deployed together.

    The app serves at least three purposes:

    * A unit of deployment for functions and classes.
    * Syncing of identities of (primarily) functions and classes across processes
      (your local Python interpreter and every Modal container active in your application).
    * Manage log collection for everything that happens inside your code.

    **Registering functions with an app**

    The most common way to explicitly register an Object with an app is through the
    `@app.function()` decorator. It both registers the annotated function itself and
    other passed objects, like schedules and secrets, with the app:

    ```python
    import modal

    app = modal.App()

    @app.function(
        secrets=[modal.Secret.from_name("some_secret")],
        schedule=modal.Period(days=1),
    )
    def foo():
        pass
    ```

    In this example, the secret and schedule are registered with the app.
    """

    _all_apps: ClassVar[dict[Optional[str], list["_App"]]] = {}
    _container_app: ClassVar[Optional["_App"]] = None

    _name: Optional[str]
    _description: Optional[str]
    _functions: dict[str, _Function]
    _classes: dict[str, _Cls]

    _image: Optional[_Image]
    _mounts: Sequence[_Mount]
    _secrets: Sequence[_Secret]
    _volumes: dict[Union[str, PurePosixPath], _Volume]
    _web_endpoints: list[str]  # Used by the CLI
    _local_entrypoints: dict[str, _LocalEntrypoint]

    # Running apps only (container apps or running local)
    _app_id: Optional[str]  # Kept after app finishes
    _app_layout: Optional[AppLayout]  # Various app info
    _client: Optional[_Client]
    _interactive: Optional[bool]

    _include_source_default: Optional[bool] = None

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        image: Optional[_Image] = None,  # default image for all functions (default is `modal.Image.debian_slim()`)
        mounts: Sequence[_Mount] = [],  # default mounts for all functions
        secrets: Sequence[_Secret] = [],  # default secrets for all functions
        volumes: dict[Union[str, PurePosixPath], _Volume] = {},  # default volumes for all functions
        include_source: Optional[bool] = None,
    ) -> None:
        """Construct a new app, optionally with default image, mounts, secrets, or volumes.

        ```python notest
        image = modal.Image.debian_slim().pip_install(...)
        secret = modal.Secret.from_name("my-secret")
        volume = modal.Volume.from_name("my-data")
        app = modal.App(image=image, secrets=[secret], volumes={"/mnt/data": volume})
        ```
        """
        if name is not None and not isinstance(name, str):
            raise InvalidError("Invalid value for `name`: Must be string.")

        self._name = name
        self._description = name
        self._include_source_default = include_source

        check_sequence(mounts, _Mount, "`mounts=` has to be a list or tuple of `modal.Mount` objects")
        check_sequence(secrets, _Secret, "`secrets=` has to be a list or tuple of `modal.Secret` objects")
        validate_volumes(volumes)

        if image is not None and not isinstance(image, _Image):
            raise InvalidError("`image=` has to be a `modal.Image` object")

        self._functions = {}
        self._classes = {}
        self._image = image
        self._mounts = mounts
        self._secrets = secrets
        self._volumes = volumes
        self._local_entrypoints = {}
        self._web_endpoints = []

        self._app_id = None
        self._app_layout = None  # Set inside container, OR during the time an app is running locally
        self._client = None
        self._interactive = None

        # Register this app. This is used to look up the app in the container, when we can't get it from the function
        _App._all_apps.setdefault(self._name, []).append(self)

    @property
    def name(self) -> Optional[str]:
        """The user-provided name of the App."""
        return self._name

    @property
    def is_interactive(self) -> Optional[bool]:
        """Whether the current app for the app is running in interactive mode."""
        return self._interactive

    @property
    def app_id(self) -> Optional[str]:
        """Return the app_id of a running or stopped app."""
        return self._app_id

    @property
    def description(self) -> Optional[str]:
        """The App's `name`, if available, or a fallback descriptive identifier."""
        return self._description

    @staticmethod
    @renamed_parameter((2024, 12, 18), "label", "name")
    async def lookup(
        name: str,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_App":
        """Look up an App with a given name, creating a new App if necessary.

        Note that Apps created through this method will be in a deployed state,
        but they will not have any associated Functions or Classes. This method
        is mainly useful for creating an App to associate with a Sandbox:

        ```python
        app = modal.App.lookup("my-app", create_if_missing=True)
        modal.Sandbox.create("echo", "hi", app=app)
        ```
        """
        if client is None:
            client = await _Client.from_env()

        environment_name = _get_environment_name(environment_name)

        request = api_pb2.AppGetOrCreateRequest(
            app_name=name,
            environment_name=environment_name,
            object_creation_type=(api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING if create_if_missing else None),
        )

        response = await retry_transient_errors(client.stub.AppGetOrCreate, request)

        app = _App(name)
        app._app_id = response.app_id
        app._client = client
        app._app_layout = AppLayout()
        return app

    def set_description(self, description: str):
        self._description = description

    def _validate_blueprint_value(self, key: str, value: Any):
        if not isinstance(value, _Object):
            raise InvalidError(f"App attribute `{key}` with value {value!r} is not a valid Modal object")

    @property
    def image(self) -> _Image:
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

    def _uncreate_all_objects(self):
        # TODO(erikbern): this doesn't unhydrate objects that aren't tagged
        for obj in self._functions.values():
            obj._unhydrate()
        for obj in self._classes.values():
            obj._unhydrate()

    @asynccontextmanager
    async def _set_local_app(
        self, client: _Client, app_layout: AppLayout, app_id: str, interactive: bool
    ) -> AsyncGenerator[None, None]:
        self._client = client
        self._app_layout = app_layout
        self._app_id = app_id
        self._interactive = interactive
        try:
            yield
        finally:
            self._client = None
            self._app_layout = None
            self._interactive = None
            self._uncreate_all_objects()

    @asynccontextmanager
    async def run(
        self,
        client: Optional[_Client] = None,
        show_progress: Optional[bool] = None,
        detach: bool = False,
        interactive: bool = False,
        environment_name: Optional[str] = None,
    ) -> AsyncGenerator["_App", None]:
        """Context manager that runs an app on Modal.

        Use this as the main entry point for your Modal application. All calls
        to Modal functions should be made within the scope of this context
        manager, and they will correspond to the current app.

        **Example**

        ```python notest
        with app.run():
            some_modal_function.remote()
        ```

        To enable output printing, use `modal.enable_output()`:

        ```python notest
        with modal.enable_output():
            with app.run():
                some_modal_function.remote()
        ```

        Note that you cannot invoke this in global scope of a file where you have
        Modal functions or Classes, since that would run the block when the function
        or class is imported in your containers as well. If you want to run it as
        your entrypoint, consider wrapping it:

        ```python
        if __name__ == "__main__":
            with app.run():
                some_modal_function.remote()
        ```

        You can then run your script with:

        ```shell
        python app_module.py
        ```

        Note that this method used to return a separate "App" object. This is
        no longer useful since you can use the app itself for access to all
        objects. For backwards compatibility reasons, it returns the same app.
        """
        from .runner import _run_app  # Defer import of runner.py, which imports a lot from Rich

        # See Github discussion here: https://github.com/modal-labs/modal-client/pull/2030#issuecomment-2237266186

        if show_progress is True:
            deprecation_error(
                (2024, 11, 20),
                "`show_progress=True` is no longer supported. Use `with modal.enable_output():` instead.",
            )
        elif show_progress is False:
            deprecation_warning((2024, 11, 20), "`show_progress=False` is deprecated (and has no effect)")

        async with _run_app(
            self, client=client, detach=detach, interactive=interactive, environment_name=environment_name
        ):
            yield self

    def _get_default_image(self):
        if self._image:
            return self._image
        else:
            return _default_image

    def _get_watch_mounts(self):
        if not self._app_layout:
            raise ExecutionError("`_get_watch_mounts` requires a running app.")

        all_mounts = [
            *self._mounts,
        ]
        for function in self.registered_functions.values():
            all_mounts.extend(function._serve_mounts)

        return [m for m in all_mounts if m.is_local()]

    def _add_function(self, function: _Function, is_web_endpoint: bool):
        if old_function := self._functions.get(function.tag, None):
            if old_function is function:
                return  # already added the same exact instance, ignore

            if not is_notebook():
                logger.warning(
                    f"Warning: function name '{function.tag}' collision!"
                    " Overriding existing function "
                    f"[{old_function._info.module_name}].{old_function._info.function_name}"
                    f" with new function [{function._info.module_name}].{function._info.function_name}"
                )
        if function.tag in self._classes:
            logger.warning(f"Warning: tag {function.tag} exists but is overridden by function")

        if self._app_layout:
            # If this is inside a container, then objects can be defined after app initialization.
            # So we may have to initialize objects once they get bound to the app.
            if function.tag in self._app_layout.function_ids:
                object_id: str = self._app_layout.function_ids[function.tag]
                metadata: Message = self._app_layout.object_handle_metadata[object_id]
                function._hydrate(object_id, self._client, metadata)

        self._functions[function.tag] = function
        if is_web_endpoint:
            self._web_endpoints.append(function.tag)

    def _add_class(self, tag: str, cls: _Cls):
        if self._app_layout:
            # If this is inside a container, then objects can be defined after app initialization.
            # So we may have to initialize objects once they get bound to the app.
            if tag in self._app_layout.class_ids:
                object_id: str = self._app_layout.class_ids[tag]
                metadata: Message = self._app_layout.object_handle_metadata[object_id]
                cls._hydrate(object_id, self._client, metadata)

        self._classes[tag] = cls

    def _init_container(self, client: _Client, app_id: str, app_layout: AppLayout):
        self._app_id = app_id
        self._app_layout = app_layout
        self._client = client

        _App._container_app = self

        # Hydrate function objects
        for tag, object_id in app_layout.function_ids.items():
            if tag in self._functions:
                obj = self._functions[tag]
                handle_metadata = app_layout.object_handle_metadata[object_id]
                obj._hydrate(object_id, client, handle_metadata)

        # Hydrate class objects
        for tag, object_id in app_layout.class_ids.items():
            if tag in self._classes:
                obj = self._classes[tag]
                handle_metadata = app_layout.object_handle_metadata[object_id]
                obj._hydrate(object_id, client, handle_metadata)

    @property
    def registered_functions(self) -> dict[str, _Function]:
        """All modal.Function objects registered on the app."""
        return self._functions

    @property
    def registered_classes(self) -> dict[str, _Cls]:
        """All modal.Cls objects registered on the app."""
        return self._classes

    @property
    def registered_entrypoints(self) -> dict[str, _LocalEntrypoint]:
        """All local CLI entrypoints registered on the app."""
        return self._local_entrypoints

    @property
    def indexed_objects(self) -> dict[str, _Object]:
        deprecation_warning(
            (2024, 11, 25),
            "`app.indexed_objects` is deprecated! Use `app.registered_functions` or `app.registered_classes` instead.",
        )
        return dict(**self._functions, **self._classes)

    @property
    def registered_web_endpoints(self) -> list[str]:
        """Names of web endpoint (ie. webhook) functions registered on the app."""
        return self._web_endpoints

    def local_entrypoint(
        self, _warn_parentheses_missing: Any = None, *, name: Optional[str] = None
    ) -> Callable[[Callable[..., Any]], _LocalEntrypoint]:
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

        def wrapped(raw_f: Callable[..., Any]) -> _LocalEntrypoint:
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
        gpu: Union[
            GPU_T, list[GPU_T]
        ] = None,  # GPU request as string ("any", "T4", ...), object (`modal.GPU.A100()`, ...), or a list of either
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),  # Modal Mounts added to the container
        network_file_systems: dict[
            Union[str, PurePosixPath], _NetworkFileSystem
        ] = {},  # Mountpoints for Modal NetworkFileSystems
        volumes: dict[
            Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes & CloudBucketMounts
        allow_cross_region_volumes: bool = False,  # Whether using network file systems from other regions is allowed.
        # Specify, in fractional CPU cores, how many CPU cores to request.
        # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
        # CPU throttling will prevent a container from exceeding its specified limit.
        cpu: Optional[Union[float, tuple[float, float]]] = None,
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, tuple[int, int]]] = None,
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
        block_network: bool = False,  # Whether to block network access
        # Maximum number of inputs a container should handle before shutting down.
        # With `max_inputs = 1`, containers will be single-use.
        max_inputs: Optional[int] = None,
        i6pn: Optional[bool] = None,  # Whether to enable IPv6 container networking within the region.
        # Whether the function's home package should be included in the image - defaults to True
        include_source: Optional[bool] = None,
        # Parameters below here are experimental. Use with caution!
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
        _experimental_buffer_containers: Optional[int] = None,  # Number of additional, idle containers to keep around.
        _experimental_proxy_ip: Optional[str] = None,  # IP address of proxy
        _experimental_custom_scaling_factor: Optional[float] = None,  # Custom scaling factor
        _experimental_enable_gpu_snapshot: bool = False,  # Experimentally enable GPU memory snapshots.
    ) -> _FunctionDecoratorType:
        """Decorator to register a new Modal [Function](/docs/reference/modal.Function) with this App."""
        if isinstance(_warn_parentheses_missing, _Image):
            # Handle edge case where maybe (?) some users passed image as a positional arg
            raise InvalidError("`image` needs to be a keyword argument: `@app.function(image=image)`.")
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@app.function()`.")

        if image is None:
            image = self._get_default_image()

        secrets = [*self._secrets, *secrets]

        def wrapped(
            f: Union[_PartialFunction, Callable[..., Any], None],
        ) -> _Function:
            nonlocal keep_warm, is_generator, cloud, serialized

            # Check if the decorated object is a class
            if inspect.isclass(f):
                raise TypeError(
                    "The `@app.function` decorator cannot be used on a class. Please use `@app.cls` instead."
                )

            if isinstance(f, _PartialFunction):
                # typically for @function-wrapped @web_endpoint, @asgi_app, or @batched
                f.wrapped = True

                # but we don't support @app.function wrapping a method.
                if is_method_fn(f.raw_f.__qualname__):
                    raise InvalidError(
                        "The `@app.function` decorator cannot be used on class methods. "
                        "Swap with `@modal.method` or `@modal.web_endpoint`, or drop the `@app.function` decorator. "
                        "Example: "
                        "\n\n"
                        "```python\n"
                        "@app.cls()\n"
                        "class MyClass:\n"
                        "    @modal.web_endpoint()\n"
                        "    def f(self, x):\n"
                        "        ...\n"
                        "```\n"
                    )
                i6pn_enabled = i6pn or (f.flags & _PartialFunctionFlags.CLUSTERED)
                cluster_size = f.cluster_size  # Experimental: Clustered functions

                info = FunctionInfo(f.raw_f, serialized=serialized, name_override=name)
                raw_f = f.raw_f
                webhook_config = f.webhook_config
                is_generator = f.is_generator
                keep_warm = f.keep_warm or keep_warm
                batch_max_size = f.batch_max_size
                batch_wait_ms = f.batch_wait_ms
            else:
                if not is_global_object(f.__qualname__) and not serialized:
                    raise InvalidError(
                        dedent(
                            """
                            The `@app.function` decorator must apply to functions in global scope,
                            unless `serialize=True` is set.
                            If trying to apply additional decorators, they may need to use `functools.wraps`.
                            """
                        )
                    )

                if is_method_fn(f.__qualname__):
                    raise InvalidError(
                        dedent(
                            """
                            The `@app.function` decorator cannot be used on class methods.
                            Please use `@app.cls` with `@modal.method` instead. Example:

                            ```python
                            @app.cls()
                            class MyClass:
                                @modal.method()
                                def f(self, x):
                                    ...
                            ```
                            """
                        )
                    )

                info = FunctionInfo(f, serialized=serialized, name_override=name)
                webhook_config = None
                batch_max_size = None
                batch_wait_ms = None
                raw_f = f

                cluster_size = None  # Experimental: Clustered functions
                i6pn_enabled = i6pn

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

            function = _Function.from_local(
                info,
                app=self,
                image=image,
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
                batch_max_size=batch_max_size,
                batch_wait_ms=batch_wait_ms,
                container_idle_timeout=container_idle_timeout,
                timeout=timeout,
                keep_warm=keep_warm,
                cloud=cloud,
                webhook_config=webhook_config,
                enable_memory_snapshot=enable_memory_snapshot,
                block_network=block_network,
                max_inputs=max_inputs,
                scheduler_placement=scheduler_placement,
                _experimental_buffer_containers=_experimental_buffer_containers,
                _experimental_proxy_ip=_experimental_proxy_ip,
                i6pn_enabled=i6pn_enabled,
                cluster_size=cluster_size,  # Experimental: Clustered functions
                include_source=include_source if include_source is not None else self._include_source_default,
                _experimental_enable_gpu_snapshot=_experimental_enable_gpu_snapshot,
            )

            self._add_function(function, webhook_config is not None)

            return function

        return wrapped

    @typing_extensions.dataclass_transform(field_specifiers=(parameter,), kw_only_default=True)
    def cls(
        self,
        _warn_parentheses_missing: Optional[bool] = None,
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
        gpu: Union[
            GPU_T, list[GPU_T]
        ] = None,  # GPU request as string ("any", "T4", ...), object (`modal.GPU.A100()`, ...), or a list of either
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        network_file_systems: dict[
            Union[str, PurePosixPath], _NetworkFileSystem
        ] = {},  # Mountpoints for Modal NetworkFileSystems
        volumes: dict[
            Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes & CloudBucketMounts
        allow_cross_region_volumes: bool = False,  # Whether using network file systems from other regions is allowed.
        # Specify, in fractional CPU cores, how many CPU cores to request.
        # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
        # CPU throttling will prevent a container from exceeding its specified limit.
        cpu: Optional[Union[float, tuple[float, float]]] = None,
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, tuple[int, int]]] = None,
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
        block_network: bool = False,  # Whether to block network access
        # Limits the number of inputs a container handles before shutting down.
        # Use `max_inputs = 1` for single-use containers.
        max_inputs: Optional[int] = None,
        include_source: Optional[bool] = None,
        # Parameters below here are experimental. Use with caution!
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
        _experimental_buffer_containers: Optional[int] = None,  # Number of additional, idle containers to keep around.
        _experimental_proxy_ip: Optional[str] = None,  # IP address of proxy
        _experimental_custom_scaling_factor: Optional[float] = None,  # Custom scaling factor
        _experimental_enable_gpu_snapshot: bool = False,  # Experimentally enable GPU memory snapshots.
    ) -> Callable[[CLS_T], CLS_T]:
        """
        Decorator to register a new Modal [Cls](/docs/reference/modal.Cls) with this App.
        """
        if _warn_parentheses_missing:
            raise InvalidError("Did you forget parentheses? Suggestion: `@app.cls()`.")

        scheduler_placement = _experimental_scheduler_placement
        if region:
            if scheduler_placement:
                raise InvalidError("`region` and `_experimental_scheduler_placement` cannot be used together")
            scheduler_placement = SchedulerPlacement(region=region)

        def wrapper(user_cls: CLS_T) -> CLS_T:
            nonlocal keep_warm

            # Check if the decorated object is a class
            if not inspect.isclass(user_cls):
                raise TypeError("The @app.cls decorator must be used on a class.")

            batch_functions = _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.BATCHED)
            if batch_functions:
                if len(batch_functions) > 1:
                    raise InvalidError(f"Modal class {user_cls.__name__} can only have one batched function.")
                if len(_find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.FUNCTION)) > 1:
                    raise InvalidError(
                        f"Modal class {user_cls.__name__} with a modal batched function cannot have other modal methods."  # noqa
                    )
                batch_function = next(iter(batch_functions.values()))
                batch_max_size = batch_function.batch_max_size
                batch_wait_ms = batch_function.batch_wait_ms
            else:
                batch_max_size = None
                batch_wait_ms = None

            if (
                _find_partial_methods_for_user_cls(user_cls, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
                and not enable_memory_snapshot
            ):
                raise InvalidError("A class must have `enable_memory_snapshot=True` to use `snap=True` on its methods.")

            info = FunctionInfo(None, serialized=serialized, user_cls=user_cls)

            cls_func = _Function.from_local(
                info,
                app=self,
                image=image or self._get_default_image(),
                secrets=[*self._secrets, *secrets],
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
                batch_max_size=batch_max_size,
                batch_wait_ms=batch_wait_ms,
                container_idle_timeout=container_idle_timeout,
                timeout=timeout,
                cpu=cpu,
                keep_warm=keep_warm,
                cloud=cloud,
                enable_memory_snapshot=enable_memory_snapshot,
                block_network=block_network,
                max_inputs=max_inputs,
                scheduler_placement=scheduler_placement,
                include_source=include_source if include_source is not None else self._include_source_default,
                _experimental_buffer_containers=_experimental_buffer_containers,
                _experimental_proxy_ip=_experimental_proxy_ip,
                _experimental_custom_scaling_factor=_experimental_custom_scaling_factor,
                _experimental_enable_gpu_snapshot=_experimental_enable_gpu_snapshot,
            )

            self._add_function(cls_func, is_web_endpoint=False)

            cls: _Cls = _Cls.from_local(user_cls, self, cls_func)

            tag: str = user_cls.__name__
            self._add_class(tag, cls)
            return cls  # type: ignore  # a _Cls instance "simulates" being the user provided class

        return wrapper

    async def spawn_sandbox(
        self,
        *entrypoint_args: str,
        image: Optional[_Image] = None,  # The image to run as the container for the sandbox.
        mounts: Sequence[_Mount] = (),  # Mounts to attach to the sandbox.
        secrets: Sequence[_Secret] = (),  # Environment variables to inject into the sandbox.
        network_file_systems: dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},
        timeout: Optional[int] = None,  # Maximum execution time of the sandbox in seconds.
        workdir: Optional[str] = None,  # Working directory of the sandbox.
        gpu: GPU_T = None,
        cloud: Optional[str] = None,
        region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the sandbox on.
        # Specify, in fractional CPU cores, how many CPU cores to request.
        # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
        # CPU throttling will prevent a container from exceeding its specified limit.
        cpu: Optional[Union[float, tuple[float, float]]] = None,
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, tuple[int, int]]] = None,
        block_network: bool = False,  # Whether to block network access
        volumes: dict[
            Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes and CloudBucketMounts
        pty_info: Optional[api_pb2.PTYInfo] = None,
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
    ) -> None:
        """mdmd:hidden"""
        arglist = ", ".join(repr(s) for s in entrypoint_args)
        message = (
            "`App.spawn_sandbox` is deprecated.\n\n"
            "Sandboxes can be created using the `Sandbox` object:\n\n"
            f"```\nsb = Sandbox.create({arglist}, app=app)\n```\n\n"
            "See https://modal.com/docs/guide/sandbox for more info on working with sandboxes."
        )
        deprecation_error((2024, 7, 5), message)

    def include(self, /, other_app: "_App"):
        """Include another App's objects in this one.

        Useful for splitting up Modal Apps across different self-contained files.

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
        for tag, function in other_app._functions.items():
            self._add_function(function, False)  # TODO(erikbern): webhook config?

        for tag, cls in other_app._classes.items():
            existing_cls = self._classes.get(tag)
            if existing_cls and existing_cls != cls:
                logger.warning(
                    f"Named app class {tag} with existing value {existing_cls} is being "
                    f"overwritten by a different class {cls}"
                )

            self._add_class(tag, cls)

    async def _logs(self, client: Optional[_Client] = None) -> AsyncGenerator[str, None]:
        """Stream logs from the app.

        This method is considered private and its interface may change - use at your own risk!
        """
        if not self._app_id:
            raise InvalidError("`app._logs` requires a running/stopped app.")

        client = client or self._client or await _Client.from_env()

        last_log_batch_entry_id: Optional[str] = None
        while True:
            request = api_pb2.AppGetLogsRequest(
                app_id=self._app_id,
                timeout=55,
                last_entry_id=last_log_batch_entry_id,
            )
            async for log_batch in client.stub.AppGetLogs.unary_stream(request):
                if log_batch.entry_id:
                    # log_batch entry_id is empty for fd="server" messages from AppGetLogs
                    last_log_batch_entry_id = log_batch.entry_id
                if log_batch.app_done:
                    return
                for log in log_batch.items:
                    if log.data:
                        yield log.data

    @classmethod
    def _get_container_app(cls) -> Optional["_App"]:
        """Returns the `App` running inside a container.

        This will return `None` outside of a Modal container."""
        return cls._container_app

    @classmethod
    def _reset_container_app(cls):
        """Only used for tests."""
        cls._container_app = None


App = synchronize_api(_App)


class _Stub(_App):
    """mdmd:hidden
    This enables using a "Stub" class instead of "App".

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
