# Copyright Modal Labs 2022
import asyncio
import typing
from datetime import date
import inspect
from multiprocessing.synchronize import Event
import os
import sys
import warnings
from typing import AsyncGenerator, Callable, Dict, List, Optional, Union, Any, Sequence, Set

from synchronicity.async_wrap import asynccontextmanager
from modal._types import typechecked

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.decorator_utils import decorator_with_options, decorator_with_options_deprecated
from .retries import Retries

from ._function_utils import FunctionInfo
from ._ipython import is_notebook
from ._output import OutputManager
from ._pty import exec_cmd
from .app import _App, _container_app, is_local
from .client import _Client
from .config import logger
from .exception import InvalidError, deprecation_warning
from .functions import _Function, _FunctionHandle
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


class WebhookConfig:
    def __init__(self, raw_f: Callable[..., Any], webhook_config: api_pb2.WebhookConfig):
        self.raw_f = raw_f
        self.webhook_config = webhook_config


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
    `@stub.function` decorator. It both registers the annotated function itself and
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

    _name: str
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
    _loose_webhook_configs: Set[Callable[..., Any]]  # Used to warn users if they forget to decorate a webhook

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
        self._function_handles = {}
        self._local_entrypoints = {}
        self._local_mounts = []
        self._web_endpoints = []
        self._loose_webhook_configs = set()

        self._app = None
        if not is_local():
            # TODO(erikbern): in theory there could be multiple stubs defined.
            # We should try to determine whether this is in fact the "right" one.
            # We could probably do this by looking at the app's name.
            self._app = _container_app

    @property
    def name(self) -> str:
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
                    print("I'm inside!")"""
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
        """Run an app until the program is interrupted."""
        deprecation_warning(
            date(2023, 2, 28),
            "stub.serve() is deprecated. Use the `modal serve` CLI command instead",
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
        tag = info.get_tag()
        function_handle: Optional[_FunctionHandle] = None
        if self._app:
            # Grab the existing function handle from the running app
            # TODO: handle missing items, or wrong types
            try:
                handle = self._app[tag]
                if isinstance(handle, _FunctionHandle):
                    function_handle = handle
                else:
                    logger.warning(f"Object {tag} has wrong type {type(handle)}")
            except KeyError:
                logger.warning(
                    f"Could not find Modal function '{tag}' in app '{self.description}'. '{tag}' may still be invoked as local function: {tag}()"
                )

        if function_handle is None:
            function_handle = _FunctionHandle._new()

        function_handle._initialize_from_local(self, info)
        self._function_handles[tag] = function_handle
        return function_handle

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
        @stub.local_entrypoint
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
        @stub.local_entrypoint
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
        shared_volumes: Dict[str, _SharedVolume] = {},
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Union[bool, int, None] = None,  # An optional number of containers to always keep warm.
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[bool] = None,  # If not set, it's inferred from the function signature
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, auto.
    ) -> Callable[[Callable[..., Any]], _FunctionHandle]:
        ...

    @typing.overload
    def function(
        self,
        f: Union[WebhookConfig, Callable[..., Any]],  # The decorated function
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[str, _SharedVolume] = {},
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Union[bool, int, None] = None,  # An optional number of containers to always keep warm.
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[bool] = None,  # If not set, it's inferred from the function signature
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, auto.
    ) -> _FunctionHandle:
        ...

    @decorator_with_options
    @typechecked
    def function(
        self,
        f: Optional[Union[WebhookConfig, Callable[..., Any]]] = None,  # The decorated function
        *,
        image: Optional[_Image] = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[str, _SharedVolume] = {},
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Union[bool, int, None] = None,  # An optional number of containers to always keep warm.
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[bool] = None,  # If not set, it's inferred from the function signature
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, auto.
    ) -> Union[
        _FunctionHandle, Callable[[Callable[..., Any]], _FunctionHandle]
    ]:  # Function object - callable as a regular function within a Modal app
        """Decorator to register a new Modal function with this stub."""
        if image is None:
            image = self._get_default_image()

        if isinstance(f, WebhookConfig):
            info = FunctionInfo(f.raw_f, serialized=serialized, name_override=name)
            webhook_config = f.webhook_config
            self._web_endpoints.append(info.get_tag())
            raw_f = f.raw_f
            self._loose_webhook_configs.remove(raw_f)

            if is_generator or (inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)):
                if webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
                    raise InvalidError(
                        "Webhooks cannot be generators. If you want to streaming response, use fastapi.responses.StreamingResponse. Example:\n\n"
                        "def my_iter():\n    for x in range(10):\n        time.sleep(1.0)\n        yield str(i)\n\n"
                        "@stub.function()\n@stub.web_endpoint()\ndef web():\n    return StreamingResponse(my_iter())\n"
                    )
                else:
                    raise InvalidError("Webhooks cannot be generators")
        else:
            info = FunctionInfo(f, serialized=serialized, name_override=name)
            webhook_config = None
            raw_f = f

        function_handle = self._get_function_handle(info)
        base_mounts = self._get_function_mounts(info)
        secrets = [*self._secrets, *secrets]

        if is_generator is None:
            is_generator = inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)

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
            serialized=serialized,
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
        )

        self._add_function(function, [*base_mounts, *mounts])
        return function_handle

    @decorator_with_options_deprecated
    @typechecked
    def web_endpoint(
        self,
        raw_f: Optional[Callable[..., Any]] = None,
        method: str = "GET",  # REST method for the created endpoint.
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        """Register a basic web endpoint with this application.

        This is the simple way to create a web endpoint on Modal. The function
        behaves as a [FastAPI](https://fastapi.tiangolo.com/) handler and should
        return a response object to the caller.

        Endpoints created with `@stub.web_endpoint` are meant to be simple, single
        request handlers and automatically have
        [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled.
        For more flexibility, use `@stub.asgi_app`.

        To learn how to use Modal with popular web frameworks, see the
        [guide on web endpoints](https://modal.com/docs/guide/webhooks).

        All webhook requests have a 150s maximum request time for the HTTP request itself. However, the underlying functions can
        run for longer and return results to the caller on completion.

        The two `wait_for_response` modes for webhooks are as follows:
        * `wait_for_response=True` - tries to fulfill the request on the original URL, but returns a 302 redirect after ~150s to a result URL (original URL with an added `__modal_function_id=...` query parameter)
        * `wait_for_response=False` - immediately returns a 202 ACCEPTED response with a JSON payload: `{"result_url": "..."}` containing the result "redirect" URL from above (which in turn redirects to itself every ~150s)
        """
        if isinstance(raw_f, _FunctionHandle):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@stub.function()\n@stub.web_endpoint()\ndef my_webhook():\n    ..."
            )
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        self._loose_webhook_configs.add(raw_f)

        return WebhookConfig(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION,
                method=method,
                requested_suffix=label,
                async_mode=_response_mode,
            ),
        )

    @decorator_with_options_deprecated
    @typechecked
    def asgi_app(
        self,
        raw_f=None,
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        """Register an ASGI app with this application.

        Asynchronous Server Gateway Interface (ASGI) is a standard for Python
        synchronous and asynchronous apps, supported by all popular Python web
        libraries. This is an advanced decorator that gives full flexibility in
        defining one or more web endpoints on Modal.

        To learn how to use Modal with popular web frameworks, see the
        [guide on web endpoints](https://modal.com/docs/guide/webhooks).

        The two `wait_for_response` modes for webhooks are as follows:
        * wait_for_response=True - tries to fulfill the request on the original URL, but returns a 302 redirect after ~150s to a result URL (original URL with an added `__modal_function_id=fc-1234abcd` query parameter)
        * wait_for_response=False - immediately returns a 202 ACCEPTED response with a json payload: `{"result_url": "..."}` containing the result "redirect" url from above (which in turn redirects to itself every 150s)
        """
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        self._loose_webhook_configs.add(raw_f)

        return WebhookConfig(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
                requested_suffix=label,
                async_mode=_response_mode,
            ),
        )

    @decorator_with_options_deprecated
    @typechecked
    def wsgi_app(
        self,
        raw_f=None,
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default
        self._loose_webhook_configs.add(raw_f)
        return WebhookConfig(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_WSGI_APP,
                requested_suffix=label,
                async_mode=_response_mode,
            ),
        )

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
        deprecation_warning(
            date(2023, 4, 3),
            "stub.webhook() is deprecated. Use stub.function in combination with stub.web_endpoint instead. Usage:\n\n"
            '@stub.function(cpu=42)\n@stub.web_endpoint(method="POST")\ndef my_function():\n    ...',
        )
        web_endpoint = self.web_endpoint(method=method, label=label, wait_for_response=wait_for_response)(raw_f)
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
        deprecation_warning(
            date(2023, 4, 3),
            "stub.asgi() is deprecated. Use stub.function in combination with stub.asgi_app instead. Usage:\n\n"
            "@stub.function(cpu=42)\n@stub.asgi_app()\ndef my_asgi_app():\n    ...",
        )
        web_endpoint = self.asgi_app(label=label, wait_for_response=wait_for_response)(raw_f)
        return self.function(web_endpoint, **function_args)

    @decorator_with_options
    def wsgi(
        self,
        raw_f,
        label: Optional[str] = None,
        wait_for_response: bool = True,
        **function_args,
    ) -> _FunctionHandle:
        deprecation_warning(
            date(2023, 4, 3),
            "stub.wsgi() is deprecated. Use stub.function in combination with stub.wsgi_app instead. Usage:\n\n"
            "@stub.function(cpu=42)\n@stub.wsgi_app()\ndef my_wsgi_app():\n    ...",
        )
        web_endpoint = self.wsgi_app(label=label, wait_for_response=wait_for_response)(raw_f)
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


Stub, AioStub = synchronize_apis(_Stub)
