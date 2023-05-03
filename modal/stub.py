# Copyright Modal Labs 2022
import asyncio
import typing
from datetime import date
import inspect
from multiprocessing.synchronize import Event
import os
import sys
import warnings
from typing import AsyncGenerator, Callable, Dict, List, Optional, Union, Any, Sequence

from synchronicity.async_wrap import asynccontextmanager
from modal._types import typechecked

from modal_proto import api_pb2

from modal_utils.async_utils import synchronize_apis, synchronizer
from modal_utils.decorator_utils import decorator_with_options_unsupported, decorator_with_options
from .retries import Retries

from ._function_utils import FunctionInfo
from ._ipython import is_notebook
from ._output import OutputManager
from ._pty import exec_cmd
from .app import _App, _container_app, is_local
from .client import _Client
from .config import logger
from .exception import InvalidError, deprecation_warning
from .functions import _Function, _FunctionHandle, PartialFunction, AioPartialFunction, _PartialFunction
from .functions import _asgi_app, _web_endpoint, _wsgi_app
from .gpu import GPU_T
from .image import _Image, _ImageHandle
from .mount import _get_client_mount, _Mount
from .object import _Provider
from .proxy import _Proxy
from .queue import _Queue
from .runner import run_stub, deploy_stub, serve_update
from .schedule import Schedule
from .secret import _Secret
from .shared_volume import _SharedVolume

_default_image: _Image = _Image.debian_slim()


class LocalEntrypoint:
    raw_f: Callable[..., Any]
    _stub: "_Stub"

    def __init__(self, raw_f, stub):
        self.raw_f = raw_f  # type: ignore
        self._stub = stub

    def __call__(self, *args, **kwargs):
        return self.raw_f(*args, **kwargs)


def check_sequence(items: typing.Sequence[typing.Any], item_type: typing.Type[typing.Any], error_msg: str):
    if not isinstance(items, (list, tuple)):
        raise InvalidError(error_msg)
    if not all(isinstance(v, item_type) for v in items):
        raise InvalidError(error_msg)


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
    _description: str
    _app_id: str
    _blueprint: Dict[str, _Provider]
    _client_mount: Optional[_Mount]
    _function_mounts: Dict[str, _Mount]
    _mounts: Sequence[_Mount]
    _secrets: Sequence[_Secret]
    _function_handles: Dict[str, _FunctionHandle]
    _web_endpoints: List[str]  # Used by the CLI
    _local_entrypoints: Dict[str, LocalEntrypoint]
    _local_mounts: List[_Mount]
    _app: Optional[_App]
    _all_stubs: typing.ClassVar[Dict[str, List["_Stub"]]] = {}

    @typechecked
    def __init__(
        self,
        name: Optional[str] = None,
        *,
        image: Optional[_Image] = None,  # default image for all functions (default is `modal.Image.debian_slim()`)
        mounts: Sequence[_Mount] = [],  # default mounts for all functions
        secrets: Sequence[_Secret] = [],  # default secrets for all functions
        **blueprint: _Provider,  # any Modal Object dependencies (Dict, Queue, etc.)
    ) -> None:
        """Construct a new app stub, optionally with default image, mounts, secrets

        Any "blueprint" objects are loaded as part of running or deploying the app,
        and are accessible by name on the running container app, e.g.:
        ```python
        stub = modal.Stub(key_value_store=modal.Dict())

        @stub.function()
        def store_something(key: str, value: str):
            stub.app.key_value_store.put(key, value)
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

        self._client_mount = None
        self._function_mounts = {}
        self._mounts = mounts
        self._secrets = secrets
        self._function_handles: Dict[str, _FunctionHandle] = {}
        self._local_entrypoints = {}
        self._local_mounts = []
        self._web_endpoints = []

        self._app = None

        string_name = self._name or ""
        existing_stubs = _Stub._all_stubs.setdefault(string_name, [])

        if not is_local() and _container_app._stub_name == string_name:
            if len(existing_stubs) == 1:  # warn the first time we reach a duplicate stub name for the active stub
                if self._name is None:
                    warning_sub_message = "unnamed stub"
                else:
                    warning_sub_message = f"stub with the same name ('{self._name}')"
                logger.warning(
                    f"You have more than one {warning_sub_message}. It's recommended to name all your Stubs uniquely when using multiple stubs"
                )
            # note that all stubs with the correct name will get the container app assigned
            self._app = _container_app

        _Stub._all_stubs[string_name].append(self)

    @property
    def name(self) -> Optional[str]:
        """The user-provided name of the Stub."""
        return self._name

    @property
    def app(self) -> Optional[_App]:
        """Reference to the currently running app, if any."""
        return self._app

    @property
    def description(self) -> str:
        """The Stub's `name`, if available, or a fallback descriptive identifier."""
        return self._description or self._infer_app_desc()

    def _validate_blueprint_value(self, key: str, value: Any):
        if not isinstance(value, _Provider):
            raise InvalidError(f"Stub attribute {key} with value {value} is not a valid Modal object")

    def _infer_app_desc(self):
        if is_notebook():
            # when running from a notebook the sys.argv for the kernel will
            # be really long and not very helpful
            return "Notebook"  # TODO: use actual name of notebook

        if is_local():
            script_filename = os.path.split(sys.argv[0])[-1]
            args = [script_filename] + sys.argv[1:]
            return " ".join(args)
        else:
            # in a container we rarely use the description, but nice to have a fallback
            # instead of displaying "_container_entrypoint.py [base64 garbage]"
            return "[unnamed app]"

    def __getitem__(self, tag: str):
        # Deprecated? Note: this is currently the only way to refer to lifecycled methods on the stub, since they have . in the tag
        return self._blueprint[tag]

    def __setitem__(self, tag: str, obj: _Provider):
        self._validate_blueprint_value(tag, obj)
        # Deprecated ?
        self._blueprint[tag] = obj

    def __getattr__(self, tag: str) -> _Provider:
        assert isinstance(tag, str)
        if tag.startswith("__"):
            # Hacky way to avoid certain issues, e.g. pickle will try to look this up
            raise AttributeError(f"Stub has no member {tag}")
        # Return a reference to an object that will be created in the future
        return self._blueprint[tag]

    def __setattr__(self, tag: str, obj: _Provider):
        # Note that only attributes defined in __annotations__ are set on the object itself,
        # everything else is registered on the blueprint
        if tag in self.__annotations__:
            object.__setattr__(self, tag, obj)
        else:
            self._validate_blueprint_value(tag, obj)
            self._blueprint[tag] = obj

    @typechecked
    def is_inside(self, image: Optional[_Image] = None) -> bool:
        """Returns if the program is currently running inside a container for this app."""
        if self._app is None:
            return False
        elif self._app != _container_app:
            return False
        elif image is None:
            # stub.app is set, which means we're inside this stub (no specific image)
            return True

        # We need to look up the image handle from the image provider
        assert isinstance(image, _Image)
        for tag, provider in self._blueprint.items():
            if provider == image:
                image_handle = self._app[tag]
                break
        else:
            raise InvalidError(
                inspect.cleandoc(
                    """`is_inside` only works for an image associated with an App. For instance:
                    stub.image = DebianSlim()
                    if stub.is_inside(stub.image):
                        print("I'm inside!")
                    """
                )
            )

        assert isinstance(image_handle, _ImageHandle)
        return image_handle._is_inside()

    @asynccontextmanager
    async def _set_app(self, app: _App) -> AsyncGenerator[None, None]:
        self._app = app
        try:
            yield
        finally:
            self._app = None

    @asynccontextmanager
    async def run(
        self,
        client: Optional[_Client] = None,
        stdout=None,
        show_progress: Optional[bool] = None,
        detach: bool = False,
        output_mgr: Optional[OutputManager] = None,
    ) -> AsyncGenerator[_App, None]:
        """Context manager that runs an app on Modal.

        Use this as the main entry point for your Modal application. All calls
        to Modal functions should be made within the scope of this context
        manager, and they will correspond to the current app.

        See the documentation for the [`App`](modal.App) class for more details.
        """
        async with run_stub(self, client, stdout, show_progress, detach, output_mgr) as app:
            yield app

    async def serve(
        self,
        client: Optional[_Client] = None,
        stdout=None,
        show_progress: Optional[bool] = None,
        timeout: float = 1e10,
    ) -> None:
        """Deprecated. Use the `modal serve` CLI command instead."""
        deprecation_warning(
            date(2023, 2, 28),
            self.serve.__doc__,
        )
        if self._app is not None:
            raise InvalidError(
                "The stub already has an app running."
                " Are you calling stub.serve() directly?"
                " Consider using the `modal serve` shell command."
            )
        async with run_stub(self, client=client, stdout=stdout, show_progress=show_progress):
            await asyncio.sleep(timeout)

    async def serve_update(
        self,
        existing_app_id: str,
        is_ready: Event,
    ) -> None:
        await serve_update(self, existing_app_id, is_ready)

    @typechecked
    async def deploy(
        self,
        name: Optional[
            str
        ] = None,  # Unique name of the deployment. Subsequent deploys with the same name overwrites previous ones. Falls back to the app name
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client=None,
        stdout=None,
        show_progress=None,
        object_entity: str = "ap",
    ) -> _App:
        """Deploy an app and export its objects persistently.

        Typically, using the command-line tool `modal deploy <module or script>`
        should be used, instead of this method.

        **Usage:**

        ```python
        if __name__ == "__main__":
            stub.deploy()
        ```

        Deployment has two primary purposes:

        * Persists all of the objects in the app, allowing them to live past the
          current app run. For schedules this enables headless "cron"-like
          functionality where scheduled functions continue to be invoked after
          the client has disconnected.
        * Allows for certain kinds of these objects, _deployment objects_, to be
          referred to and used by other apps.
        """
        return await deploy_stub(self, name, namespace, client, stdout, show_progress, object_entity)

    def _get_default_image(self):
        if "image" in self._blueprint:
            return self._blueprint["image"]
        else:
            return _default_image

    @property
    def _pty_input_stream(self):
        return self._blueprint.get("_pty_input_stream", None)

    def _get_function_mounts(self, function_info: FunctionInfo):
        # Get the common mounts for the stub.
        mounts = list(self._mounts)

        # Create client mount
        if self._client_mount is None:
            self._client_mount = _get_client_mount()
        mounts.append(self._client_mount)

        # Create function mounts
        for key, mount in function_info.get_mounts().items():
            if key not in self._function_mounts:
                self._function_mounts[key] = mount
            mounts.append(self._function_mounts[key])

        return mounts

    def _get_function_handle(self, info: FunctionInfo) -> _FunctionHandle:
        """This can either return a hydrated or an unhydrated _FunctionHandle

        If called from within a container_app that has this function handle,
        it will return a Hydrated funciton handle, but in all other contexts
        it will be unhydrated.
        """
        tag = info.get_tag()
        if tag in self._function_handles:
            return self._function_handles[tag]

        function_handle = _FunctionHandle._new()
        function_handle._initialize_from_local(self, info)
        self._function_handles[tag] = function_handle
        return function_handle  # note that the function handle is not yet hydrated at this point:

    def _add_function(self, function: _Function, mounts: List[_Mount]):
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
        self._blueprint[function.tag] = function

        # Track all mounts. This is needed for file watching
        for mount in mounts:
            if mount.is_local():
                self._local_mounts.append(mount)

    @property
    def registered_functions(self) -> Dict[str, _FunctionHandle]:
        """All modal.Function objects registered on the stub."""
        return self._function_handles

    @property
    def registered_entrypoints(self) -> Dict[str, LocalEntrypoint]:
        """All local CLI entrypoints registered on the stub."""
        return self._local_entrypoints

    @property
    def registered_web_endpoints(self) -> List[str]:
        """Names of web endpoint (ie. webhook) functions registered on the stub."""
        return self._web_endpoints

    @decorator_with_options
    def local_entrypoint(self, raw_f=None, name: Optional[str] = None):
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
        tag = name if name is not None else raw_f.__qualname__
        entrypoint = self._local_entrypoints[tag] = LocalEntrypoint(raw_f, self)
        return entrypoint

    @typing.overload
    def function(
        self,
        f: None = None,  # The decorated function
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[Union[str, os.PathLike], _SharedVolume] = {},
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[
            bool
        ] = None,  # Set this to True if it's a non-generator function returning a [sync/async] generator object
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, auto.
    ) -> Callable[[Union[_PartialFunction, Callable[..., Any]]], _FunctionHandle]:
        ...

    @typing.overload
    def function(
        self,
        f: Union[_PartialFunction, Callable[..., Any]],  # The decorated function
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[Union[str, os.PathLike], _SharedVolume] = {},
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[
            bool
        ] = None,  # Set this to True if it's a non-generator function returning a [sync/async] generator object
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, auto.
    ) -> _FunctionHandle:
        ...

    @decorator_with_options
    @typechecked
    def function(
        self,
        f: Optional[Union[_PartialFunction, Callable[..., Any]]] = None,  # The decorated function
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[Union[str, os.PathLike], _SharedVolume] = {},
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[
            bool
        ] = None,  # Set this to True if it's a non-generator function returning a [sync/async] generator object
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, auto.
        _cls: Optional[type] = None,  # Used for methods only
    ) -> Union[
        _FunctionHandle, Callable[[Union[_PartialFunction, Callable[..., Any]]], _FunctionHandle]
    ]:  # Function object - callable as a regular function within a Modal app
        """Decorator to register a new Modal function with this stub."""
        if image is None:
            image = self._get_default_image()

        if isinstance(f, _PartialFunction):
            f.wrapped = True
            info = FunctionInfo(f.raw_f, serialized=serialized, name_override=name)
            raw_f = f.raw_f
            webhook_config = f.webhook_config
            is_generator = f.is_generator
            if webhook_config:
                self._web_endpoints.append(info.get_tag())
        else:
            info = FunctionInfo(f, serialized=serialized, name_override=name)
            webhook_config = None
            raw_f = f

        if not _cls and not info.is_serialized() and "." in info.function_name:  # This is a method
            deprecation_warning(
                date(2023, 4, 20),
                inspect.cleandoc(
                    """@stub.function on methods is deprecated. Use the @stub.cls and @method decorators. Usage:

                    ```
                    @stub.cls(cpu=8)
                    class MyCls:
                        @method()
                        def f(self):
                            ...
                    ```
                    """
                ),
            )

        function_handle = self._get_function_handle(info)
        base_mounts = self._get_function_mounts(info)
        secrets = [*self._secrets, *secrets]

        if is_generator is None:
            is_generator = inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)

        if is_generator and webhook_config:
            if webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
                raise InvalidError(
                    inspect.cleandoc(
                        """Webhooks cannot be generators. If you want to streaming response, use `fastapi.responses.StreamingResponse`. Usage:

                        def my_iter():
                            for x in range(10):
                                time.sleep(1.0)
                                yield str(i)

                        @stub.function()
                        @web_endpoint()
                        def web():
                            return StreamingResponse(my_iter())
                        """
                    )
                )
            else:
                raise InvalidError("Webhooks cannot be generators")

        if interactive:
            if self._pty_input_stream:
                warnings.warn(
                    "Running multiple interactive functions at the same time is not fully supported, and could lead to unexpected behavior."
                )
            else:
                self._blueprint["_pty_input_stream"] = _Queue()

        function = _Function(
            function_handle,
            info,
            _stub=self,
            image=image,
            secret=secret,
            secrets=secrets,
            schedule=schedule,
            is_generator=is_generator,
            gpu=gpu,
            base_mounts=base_mounts,
            mounts=mounts,
            shared_volumes=shared_volumes,
            allow_cross_region_volumes=allow_cross_region_volumes,
            memory=memory,
            proxy=proxy,
            retries=retries,
            concurrency_limit=concurrency_limit,
            container_idle_timeout=container_idle_timeout,
            timeout=timeout,
            cpu=cpu,
            interactive=interactive,
            keep_warm=keep_warm,
            name=name,
            cloud=cloud,
            webhook_config=webhook_config,
            _cls=_cls,
        )

        self._add_function(function, [*base_mounts, *mounts])
        return function_handle

    @typechecked
    def web_endpoint(
        self,
        method: str = "GET",  # REST method for the created endpoint.
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        """`stub.web_endpoint` is deprecated. Use `modal.web_endpoint` instead. Usage:

        ```python
        from modal import Stub, web_endpoint

        stub = Stub()
        @stub.function(cpu=42)
        @web_endpoint(method="POST")
        def my_function():
            ...
        ```"""
        deprecation_warning(
            date(2023, 4, 18),
            self.web_endpoint.__doc__,
        )
        return _web_endpoint(method, label, wait_for_response)

    @typechecked
    def asgi_app(
        self,
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        """`stub.asgi_app` is deprecated. Use `modal.asgi_app` instead."""
        deprecation_warning(date(2023, 4, 18), self.asgi_app.__doc__)
        return _asgi_app(label, wait_for_response)

    @typechecked
    def wsgi_app(
        self,
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        """`stub.wsgi_app` is deprecated. Use `modal.wsgi_app` instead."""
        deprecation_warning(date(2023, 4, 18), self.wsgi_app.__doc__)
        return _wsgi_app(label, wait_for_response)

    @decorator_with_options
    @typechecked
    def webhook(
        self,
        raw_f=None,
        *,
        method: str = "GET",
        label: Optional[str] = None,
        wait_for_response: bool = True,
        **function_args,
    ) -> _FunctionHandle:
        """`stub.webhook` is deprecated. Use `stub.function` in combination with `modal.web_endpoint` instead. Usage:

        ```python
        @stub.function(cpu=42)
        @web_endpoint(method="POST")
        def my_function():
           ...
        ```"""
        deprecation_warning(
            date(2023, 4, 3),
            self.webhook.__doc__,
        )
        web_endpoint = _web_endpoint(method=method, label=label, wait_for_response=wait_for_response)(raw_f)
        return self.function(web_endpoint, **function_args)

    @decorator_with_options
    @typechecked
    def asgi(
        self,
        raw_f,
        *,
        label: Optional[str] = None,
        wait_for_response: bool = True,
        **function_args,
    ) -> _FunctionHandle:
        """`stub.asgi` is deprecated. Use `stub.function` in combination with `modal.asgi_app` instead. Usage:

        ```python
        @stub.function(cpu=42)
        @asgi_app()
        def my_asgi_app():
            ...
        ```"""
        deprecation_warning(
            date(2023, 4, 3),
            self.asgi.__doc__,
        )
        web_endpoint = _asgi_app(label=label, wait_for_response=wait_for_response)(raw_f)
        return self.function(web_endpoint, **function_args)

    @decorator_with_options
    def wsgi(
        self,
        raw_f,
        label: Optional[str] = None,
        wait_for_response: bool = True,
        **function_args,
    ) -> _FunctionHandle:
        """`stub.wsgi` is deprecated. Use stub.function in combination with `modal.wsgi_app` instead. Usage:

        ```
        @stub.function(cpu=42)
        @wsgi_app()
        def my_wsgi_app():
            ...
        ```"""
        deprecation_warning(
            date(2023, 4, 3),
            self.wsgi.__doc__,
        )
        web_endpoint = _wsgi_app(label=label, wait_for_response=wait_for_response)(raw_f)
        return self.function(web_endpoint, **function_args)

    async def interactive_shell(self, cmd=None, image=None, **kwargs):
        """Run an interactive shell (like `bash`) within the image for this app.

        This is useful for online debugging and interactive exploration of the
        contents of this image. If `cmd` is optionally provided, it will be run
        instead of the default shell inside this image.

        **Example**

        ```python
        import modal

        stub = modal.Stub(image=modal.Image.debian_slim().apt_install("vim"))

        if __name__ == "__main__":
            stub.interactive_shell("/bin/bash")
        ```

        Or alternatively:

        ```python
        import modal

        stub = modal.Stub()
        app_image = modal.Image.debian_slim().apt_install("vim")

        if __name__ == "__main__":
            stub.interactive_shell(cmd="/bin/bash", image=app_image)
        ```
        """
        # TODO(erikbern): rewrite the docstring above to point the user towards `modal shell`
        wrapped_fn = self.function(interactive=True, timeout=86400, image=image or self._get_default_image(), **kwargs)(
            exec_cmd
        )

        async with self.run():
            await wrapped_fn.call(cmd)

    @decorator_with_options_unsupported
    def cls(
        self,
        user_cls: Optional[type] = None,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[Union[str, os.PathLike], _SharedVolume] = {},
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, auto.
    ) -> type:
        function_handles: Dict[str, _FunctionHandle] = {}
        for k, v in user_cls.__dict__.items():
            if isinstance(v, (PartialFunction, AioPartialFunction)):
                partial_function = synchronizer._translate_in(v)  # TODO: remove need for?
                function_handles[k] = self.function(
                    _cls=user_cls,
                    image=image,
                    secret=secret,
                    secrets=secrets,
                    gpu=gpu,
                    serialized=serialized,
                    mounts=mounts,
                    shared_volumes=shared_volumes,
                    allow_cross_region_volumes=allow_cross_region_volumes,
                    cpu=cpu,
                    memory=memory,
                    proxy=proxy,
                    retries=retries,
                    concurrency_limit=concurrency_limit,
                    container_idle_timeout=container_idle_timeout,
                    timeout=timeout,
                    interactive=interactive,
                    keep_warm=keep_warm,
                    cloud=cloud,
                )(partial_function)

        _PartialFunction.initialize_cls(user_cls, function_handles)
        return user_cls

    def _hydrate_function_handles(self, client: _Client, container_app: _App):
        for tag, obj in container_app._tag_to_object.items():
            if isinstance(obj, _FunctionHandle):
                function_id = obj.object_id
                handle_metadata = obj._get_handle_metadata()
                if tag not in self._function_handles:
                    # this could happen if a sibling function decoration is lazy loaded at a later than function import
                    # assigning the app's hydrated function handle ensures it will be used for the later decoration return value
                    self._function_handles[tag] = obj
                else:
                    self._function_handles[tag]._hydrate(client, function_id, handle_metadata)


Stub, AioStub = synchronize_apis(_Stub)
