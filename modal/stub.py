# Copyright Modal Labs 2022
import inspect
import os
import sys
import typing
import warnings
from datetime import date
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Union

from synchronicity.async_wrap import asynccontextmanager

from modal._types import typechecked
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api, synchronizer

from ._function_utils import FunctionInfo
from ._ipython import is_notebook
from ._output import OutputManager
from .app import _App, _container_app, is_local
from .client import _Client
from .cls import make_remote_cls_constructors
from .config import logger
from .exception import InvalidError, deprecation_error, deprecation_warning
from .functions import PartialFunction, _Function, _FunctionHandle, _PartialFunction
from .gpu import GPU_T
from .image import _Image, _ImageHandle
from .mount import _Mount
from .network_file_system import _NetworkFileSystem
from .object import _Provider
from .proxy import _Proxy
from .queue import _Queue
from .retries import Retries
from .runner import _run_stub
from .schedule import Schedule
from .secret import _Secret
from .volume import _Volume

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
    _description: str
    _app_id: str
    _blueprint: Dict[str, _Provider]
    _function_mounts: Dict[str, _Mount]
    _mounts: Sequence[_Mount]
    _secrets: Sequence[_Secret]
    _web_endpoints: List[str]  # Used by the CLI
    _local_entrypoints: Dict[str, LocalEntrypoint]
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
        show_progress: bool = True,
        detach: bool = False,
        output_mgr: Optional[OutputManager] = None,
    ) -> AsyncGenerator[_App, None]:
        """Context manager that runs an app on Modal.

        Use this as the main entry point for your Modal application. All calls
        to Modal functions should be made within the scope of this context
        manager, and they will correspond to the current app.

        See the documentation for the [`App`](modal.App) class for more details.
        """
        # TODO(erikbern): deprecate this one too?
        async with _run_stub(self, client, stdout, show_progress, detach, output_mgr) as app:
            yield app

    @typechecked
    async def deploy(
        self,
        name: Optional[
            str
        ] = None,  # Unique name of the deployment. Subsequent deploys with the same name overwrites previous ones. Falls back to the app name
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client=None,
        stdout=None,
        show_progress=True,
        object_entity: str = "ap",
    ) -> _App:
        """`stub.deploy` is deprecated and no longer supported. Use the `modal deploy` command instead.

        For programmatic usage, use `modal.runner.deploy_stub`
        """
        deprecation_error(
            date(2023, 5, 9),
            self.deploy.__doc__,
        )

    def _get_default_image(self):
        if "image" in self._blueprint:
            return self._blueprint["image"]
        else:
            return _default_image

    @property
    def _pty_input_stream(self):
        return self._blueprint.get("_pty_input_stream", None)

    def _get_watch_mounts(self):
        all_mounts = [
            *self._mounts,
        ]
        for function_handle in self.registered_functions.values():
            # TODO(elias): This is quite ugly and should be refactored once we merge Function/FunctionHandle
            function = self[function_handle._info.get_tag()]
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
        self._blueprint[function.tag] = function

    @property
    def registered_functions(self) -> Dict[str, _Function]:
        """All modal.Function objects registered on the stub."""
        return {tag: obj for tag, obj in self._blueprint.items() if isinstance(obj, _Function)}

    @property
    def registered_entrypoints(self) -> Dict[str, LocalEntrypoint]:
        """All local CLI entrypoints registered on the stub."""
        return self._local_entrypoints

    @property
    def registered_web_endpoints(self) -> List[str]:
        """Names of web endpoint (ie. webhook) functions registered on the stub."""
        return self._web_endpoints

    def local_entrypoint(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], None]:
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

        def wrapped(raw_f: Callable[..., Any]) -> None:
            tag = name if name is not None else raw_f.__qualname__
            entrypoint = self._local_entrypoints[tag] = LocalEntrypoint(raw_f, self)
            return entrypoint

        return wrapped

    @typechecked
    def function(
        self,
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
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},  # Experimental. Do not use!
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
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
        _cls: Optional[type] = None,  # Used for methods only
    ) -> Callable[[Union[_PartialFunction, Callable[..., Any]]], _FunctionHandle]:
        """Decorator to register a new Modal function with this stub."""
        if image is None:
            image = self._get_default_image()

        secrets = [*self._secrets, *secrets]

        if shared_volumes:
            deprecation_warning(
                date(2023, 7, 5),
                "`shared_volumes` is deprecated. Use the argument `network_file_systems` instead.",
            )
            network_file_systems = {**network_file_systems, **shared_volumes}

        def wrapped(f: Union[_PartialFunction, Callable[..., Any]]) -> _FunctionHandle:
            is_generator_override: Optional[bool] = is_generator

            if isinstance(f, _PartialFunction):
                f.wrapped = True
                info = FunctionInfo(f.raw_f, serialized=serialized, name_override=name)
                raw_f = f.raw_f
                webhook_config = f.webhook_config
                is_generator_override = f.is_generator
                if webhook_config:
                    self._web_endpoints.append(info.get_tag())
            else:
                info = FunctionInfo(f, serialized=serialized, name_override=name)
                webhook_config = None
                raw_f = f

            if not _cls and not info.is_serialized() and "." in info.function_name:  # This is a method
                deprecation_error(
                    date(2023, 4, 20),
                    inspect.cleandoc(
                        """@stub.function on methods is deprecated and no longer supported.

                        Use the @stub.cls and @method decorators. Usage:

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

            info.get_tag()

            if is_generator_override is None:
                is_generator_override = inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)

            if interactive:
                if self._pty_input_stream:
                    warnings.warn(
                        "Running multiple interactive functions at the same time is not fully supported, and could lead to unexpected behavior."
                    )
                else:
                    self._blueprint["_pty_input_stream"] = _Queue.new()

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
                container_idle_timeout=container_idle_timeout,
                timeout=timeout,
                cpu=cpu,
                interactive=interactive,
                keep_warm=keep_warm,
                name=name,
                cloud=cloud,
                webhook_config=webhook_config,
                cls=_cls,
            )

            self._add_function(function)
            return function._handle

        return wrapped

    @typechecked
    def web_endpoint(
        self,
        method: str = "GET",  # REST method for the created endpoint.
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        """`stub.web_endpoint` is deprecated and no longer supported. Use `modal.web_endpoint` instead. Usage:

        ```python
        from modal import Stub, web_endpoint

        stub = Stub()
        @stub.function(cpu=42)
        @web_endpoint(method="POST")
        def my_function():
            ...
        ```"""
        deprecation_error(
            date(2023, 4, 18),
            self.web_endpoint.__doc__,
        )

    @typechecked
    def asgi_app(
        self,
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        """`stub.asgi_app` is deprecated and no longer supported. Use `modal.asgi_app` instead."""
        deprecation_error(date(2023, 4, 18), self.asgi_app.__doc__)

    @typechecked
    def wsgi_app(
        self,
        label: Optional[
            str
        ] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
    ):
        """`stub.wsgi_app` is deprecated and no longer supported. Use `modal.wsgi_app` instead."""
        deprecation_error(date(2023, 4, 18), self.wsgi_app.__doc__)

    async def interactive_shell(self, cmd=None, image=None, **kwargs):
        """`stub.interactive_shell` is deprecated and no longer supported. Use the `modal shell` command instead.

        For programmatic usage, use `modal.runner.interactive_shell`
        """
        deprecation_error(
            date(2023, 5, 9),
            self.interactive_shell.__doc__,
        )

    def cls(
        self,
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
        allow_cross_region_volumes: bool = False,  # Whether using shared volumes from other regions is allowed.
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},  # Experimental. Do not use!
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        proxy: Optional[_Proxy] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[Union[int, Retries]] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        container_idle_timeout: Optional[int] = None,  # Timeout for idle containers waiting for inputs to shut down.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        keep_warm: Optional[int] = None,  # An optional number of containers to always keep warm.
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
    ) -> Callable[[CLS_T], CLS_T]:
        def wrapper(user_cls: CLS_T) -> CLS_T:
            partial_functions: Dict[str, PartialFunction] = {}
            function_handles: Dict[str, _FunctionHandle] = {}

            for k, v in user_cls.__dict__.items():
                if isinstance(v, PartialFunction):
                    partial_functions[k] = v
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
                        network_file_systems=network_file_systems,
                        allow_cross_region_volumes=allow_cross_region_volumes,
                        volumes=volumes,
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
            remote = make_remote_cls_constructors(user_cls, partial_functions, function_handles)
            user_cls.remote = remote
            return user_cls

        return wrapper

    def _get_deduplicated_function_mounts(self, mounts: Dict[str, _Mount]):
        cached_mounts = []
        for root_path, mount in mounts.items():
            if root_path not in self._function_mounts:
                self._function_mounts[root_path] = mount
            cached_mounts.append(self._function_mounts[root_path])
        return cached_mounts


Stub = synchronize_api(_Stub)
