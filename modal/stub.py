# Copyright Modal Labs 2022
import contextlib
import inspect
import os
import sys
import warnings
from datetime import date
from enum import Enum
from typing import AsyncGenerator, Collection, Dict, List, Optional, Union

from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.app_utils import is_valid_app_name
from modal_utils.async_utils import TaskContext, synchronize_apis
from modal_utils.decorator_utils import decorator_with_options

from ._function_utils import FunctionInfo
from ._ipython import is_notebook
from ._output import OutputManager, step_completed, step_progress
from ._pty import exec_cmd, write_stdin_to_pty_stream
from .app import _App, container_app, is_local
from .client import _Client
from .config import config, logger
from .exception import InvalidError, deprecation_error, deprecation_warning
from .functions import _Function, _FunctionHandle
from .gpu import _GPUConfig
from .image import _Image
from .mount import _create_client_mount, _Mount, client_mount_name
from .object import Provider, Ref
from .queue import _Queue
from .rate_limit import RateLimit
from .schedule import Schedule
from .secret import _Secret
from .shared_volume import _SharedVolume

_default_image = _Image.debian_slim()


class StubRunMode(Enum):
    RUN = "run"
    DEPLOY = "deploy"
    DETACH = "detach"
    SERVE = "serve"


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
    _blueprint: Dict[str, Provider]
    _client_mount: Optional[_Mount]
    _function_mounts: Dict[str, _Mount]
    _mounts: Collection[_Mount]
    _secrets: Collection[_Secret]
    _function_handles: Dict[str, _FunctionHandle]

    def __init__(
        self,
        name: str = None,
        *,
        mounts: Collection[_Mount] = [],
        secrets: Collection[_Secret] = [],
        **blueprint,
    ) -> None:
        """Construct a new app stub, optionally with default mounts."""

        self._name = name
        if name is not None:
            self._description = name
        else:
            self._description = self._infer_app_desc()
        self._app_id = None
        self._blueprint = blueprint
        self._client_mount = None
        self._function_mounts = {}
        self._mounts = mounts
        self._secrets = secrets
        self._function_handles = {}
        super().__init__()

    @property
    def name(self) -> str:
        """The user-provided name of the Stub."""
        return self._name

    @property
    def description(self) -> str:
        """The Stub's `name`, if available, or a fallback descriptive identifier."""
        return self._description

    def _infer_app_desc(self):
        if is_notebook():
            # when running from a notebook the sys.argv for the kernel will
            # be really long an not very helpful
            return "Notebook"  # TODO: use actual name of notebook

        script_filename = os.path.split(sys.argv[0])[-1]
        args = [script_filename] + sys.argv[1:]
        return " ".join(args)

    def __getitem__(self, tag: str):
        # Deprecated?
        return self._blueprint[tag]

    def __setitem__(self, tag: str, obj: Provider):
        # Deprecated ?
        self._blueprint[tag] = obj

    def __getattr__(self, tag: str) -> Provider:
        assert isinstance(tag, str)
        if tag.startswith("__"):
            # Hacky way to avoid certain issues, e.g. pickle will try to look this up
            raise AttributeError(f"Stub has no member {tag}")
        # Return a reference to an object that will be created in the future
        return self._blueprint[tag]

    def __setattr__(self, tag: str, obj: Provider):
        # Note that only attributes defined in __annotations__ are set on the object itself,
        # everything else is registered on the blueprint
        if tag in self.__annotations__:
            object.__setattr__(self, tag, obj)
        else:
            self._blueprint[tag] = obj

    def is_inside(self, image: Optional[_Image] = None) -> bool:
        """Returns if the program is currently running inside a container for this app."""
        # TODO(erikbern): Add a client test for this function.
        if is_local():
            return False

        if image is not None:
            assert isinstance(image, _Image)
            for tag, provider in self._blueprint.items():
                if provider == image:
                    image_handle = container_app[tag]
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
        else:
            if "image" in self._blueprint:
                image_handle = container_app["image"]
            else:
                # At this point in the code, we are sure that the app is running
                # remotely, so it needs be able to load the ID of the default image.
                # However, we cannot call `self.load(_default_image)` because it is
                # an async function.
                #
                # Instead we load the image in App.init_container(), and this allows
                # us to retrieve its object ID from cache here.
                image_handle = container_app._load_cached(_default_image)

                # Check to make sure internal invariants are upheld.
                assert image_handle is not None, "fatal: default image should be loaded in App.init_container()"

        return image_handle._is_inside()

    @contextlib.asynccontextmanager
    async def _run(
        self,
        client,
        output_mgr: OutputManager,
        existing_app_id: Optional[str],
        last_log_entry_id: Optional[str] = None,
        name: Optional[str] = None,
        mode: StubRunMode = StubRunMode.RUN,
    ) -> AsyncGenerator[_App, None]:
        app_name = name if name is not None else self.description
        detach = mode == StubRunMode.DETACH
        if existing_app_id is not None:
            app = await _App._init_existing(self, client, existing_app_id)
        else:
            app = await _App._init_new(self, client, app_name, deploying=(mode == StubRunMode.DEPLOY), detach=detach)

        self._app_id = app.app_id
        aborted = False
        # Start tracking logs and yield context
        async with TaskContext(grace=config["logs_timeout"]) as tc:
            status_spinner = step_progress("Running app...")
            with output_mgr.ctx_if_visible(output_mgr.make_live(step_progress("Initializing..."))):
                app_id = app.app_id
                logs_loop = tc.create_task(
                    output_mgr.get_logs_loop(app_id, client, status_spinner, last_log_entry_id or "")
                )
            initialized_msg = f"Initialized. [grey70]View app at [underline]{app._app_page_url}[/underline][/grey70]"
            output_mgr.print_if_visible(step_completed(initialized_msg))

            try:
                # Create all members
                create_progress = Tree(step_progress("Creating objects..."), guide_style="gray50")
                with output_mgr.ctx_if_visible(output_mgr.make_live(create_progress)):
                    await app._create_all_objects(create_progress)
                create_progress.label = step_completed("Created objects.")
                output_mgr.print_if_visible(create_progress)

                # Update all functions client-side to point to the running app
                for tag, obj in self._function_handles.items():
                    obj._set_local_app(app)
                    obj._set_output_mgr(output_mgr)

                # Cancel logs loop after creating objects for a deployment.
                # TODO: we can get rid of this once we have 1) a way to separate builder
                # logs from runner logs and 2) a termination signal that's sent after object
                # creation is complete, that is also triggered on exceptions (`app.disconnect()`)
                if mode == StubRunMode.DEPLOY:
                    logs_loop.cancel()

                if self._pty_input_stream:
                    output_mgr._visible_progress = False
                    async with write_stdin_to_pty_stream(app._pty_input_stream):
                        yield app
                    output_mgr._visible_progress = True
                else:
                    # Yield to context
                    with output_mgr.ctx_if_visible(output_mgr.make_live(status_spinner)):
                        yield app
            except KeyboardInterrupt:
                aborted = True
                # mute cancellation errors on all function handles to prevent exception spam
                for tag, obj in self._function_handles.items():
                    obj._set_mute_cancellation(True)
                    getattr(app, tag)._set_mute_cancellation(True)  # app has a separate function handle

                if detach:
                    logs_loop.cancel()
                else:
                    print("Disconnecting from Modal - This will terminate your Modal app in a few seconds.\n")
            finally:
                if mode == StubRunMode.SERVE:
                    # Cancel logs loop since we're going to start another one.
                    logs_loop.cancel()
                else:
                    await app.disconnect()

        if mode == StubRunMode.DEPLOY:
            output_mgr.print_if_visible(step_completed("App deployed! 🎉"))
        elif aborted:
            if detach:
                output_mgr.print_if_visible(step_completed("Shutting down Modal client."))
                output_mgr.print_if_visible(
                    f"""The detached app keeps running. You can track its progress at: [magenta]{app.log_url()}[/magenta]"""
                )
            else:
                output_mgr.print_if_visible(step_completed("App aborted."))
        else:
            output_mgr.print_if_visible(step_completed("App completed."))
        self._app_id = None

    @contextlib.asynccontextmanager
    async def run(self, client=None, stdout=None, show_progress=None, detach=False) -> AsyncGenerator[_App, None]:
        """Context manager that runs an app on Modal.

        Use this as the main entry point for your Modal application. All calls
        to Modal functions should be made within the scope of this context
        manager, and they will correspond to the current app.

        See the documentation for the [`App`](modal.App) class for more details.
        """
        if not is_local():
            raise InvalidError(
                "Can not run an app from within a container. You might need to do something like this: \n"
                'if __name__ == "__main__":\n'
                "    with stub.run():\n"
                "        ...\n"
            )
        if client is None:
            client = await _Client.from_env()
        output_mgr = OutputManager(stdout, show_progress)
        mode = StubRunMode.DETACH if detach else StubRunMode.RUN
        async with self._run(client, output_mgr, existing_app_id=None, mode=mode) as app:
            yield app

    async def run_forever(self, client=None, stdout=None, show_progress=None) -> None:
        """**Deprecated.** Use `.serve()` instead."""

        deprecation_error(None, "Stub.run_forever is deprecated, use .serve() instead")
        await self.serve(client, stdout, show_progress)

    async def serve(self, client=None, stdout=None, show_progress=None, timeout=None) -> None:
        """Run an app until the program is interrupted. Modal watches source files
        and mounts for the app, and live updates the app when any changes are detected.

        This function is useful for developing and testing cron schedules, job queues, and webhooks,
        since they will run until the program is interrupted with `Ctrl + C` or other means.
        Any changes made to webhook handlers will show up almost immediately the next time the route is hit.
        """
        from ._watcher import TIMEOUT, watch

        if not is_local():
            raise InvalidError(
                "Can not run an app from within a container. You might need to do something like this: \n"
                'if __name__ == "__main__":\n'
                "    stub.serve()\n"
            )

        if self._app_id is not None:
            raise InvalidError(
                f"Found existing app '{self._app_id}'. You may have nested stub.serve() inside a running app like this:\n"
                'if __name__ == "__main__":\n'
                "    with stub.run():\n"
                "        stub.serve() # ❌\n\n"
                "You might need to do something like this: \n"
                'if __name__ == "__main__":\n'
                "    stub.serve()\n"
            )

        if client is None:
            client = await _Client.from_env()

        output_mgr = OutputManager(stdout, show_progress)
        if timeout is None:
            timeout = config["serve_timeout"]
        event_agen = watch(self, output_mgr, timeout)
        event = await event_agen.__anext__()

        app = None
        existing_app_id = None

        try:
            while event != TIMEOUT:
                if existing_app_id:
                    output_mgr.print_if_visible(f"⚡️ Updating app {existing_app_id}...")

                async with self._run(client, output_mgr, existing_app_id, mode=StubRunMode.SERVE) as app:
                    client.set_pre_stop(app.disconnect)
                    existing_app_id = app.app_id
                    event = await event_agen.__anext__()
        finally:
            await event_agen.aclose()

    async def deploy(
        self,
        name: str = None,  # Unique name of the deployment. Subsequent deploys with the same name overwrites previous ones. Falls back to the app name
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
        client=None,
        stdout=None,
        show_progress=None,
    ):
        """Deploy an app and export its objects persistently.

        Typically, using the command-line tool `modal app deploy <module or script>`
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
        if not is_local():
            raise InvalidError("Cannot run a deploy from within a container.")
        if name is None:
            name = self.name
        if name is None:
            raise InvalidError(
                "You need to either supply an explicit deployment name to the deploy command, or have a name set on the app.\n"
                "\n"
                "Examples:\n"
                'stub.deploy("some_name")\n\n'
                "or\n"
                'stub = Stub("some-name")'
            )

        if not is_valid_app_name(name):
            raise InvalidError(
                f"Invalid app name {name}. App names may only contain alphanumeric characters, dashes, periods, and underscores, and must be less than 64 characters in length. "
            )

        if client is None:
            client = await _Client.from_env()

        # Look up any existing deployment
        app_req = api_pb2.AppGetByDeploymentNameRequest(name=name, namespace=namespace, client_id=client.client_id)
        app_resp = await client.stub.AppGetByDeploymentName(app_req)
        existing_app_id = app_resp.app_id or None
        last_log_entry_id = app_resp.last_log_entry_id

        # The `_run` method contains the logic for starting and running an app
        output_mgr = OutputManager(stdout, show_progress)
        async with self._run(
            client, output_mgr, existing_app_id, last_log_entry_id, name=name, mode=StubRunMode.DEPLOY
        ) as app:
            deploy_req = api_pb2.AppDeployRequest(
                app_id=app._app_id,
                name=name,
                namespace=namespace,
            )
            deploy_response = await client.stub.AppDeploy(deploy_req)
        output_mgr.print_if_visible(f"\nView Deployment: [magenta]{deploy_response.url}[/magenta]")
        return app

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
            if config["sync_entrypoint"]:
                self._client_mount = _create_client_mount()
            else:
                self._client_mount = _Mount.from_name(
                    client_mount_name(), namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL
                )
        mounts.append(self._client_mount)

        # Create function mounts
        for key, mount in function_info.get_mounts().items():
            if key not in self._function_mounts:
                self._function_mounts[key] = mount
            mounts.append(self._function_mounts[key])

        return mounts

    def _get_function_secrets(self, raw_f, secret: Optional[_Secret] = None, secrets: Collection[_Secret] = ()):
        if secret and secrets:
            raise InvalidError(f"Function {raw_f} has both singular `secret` and plural `secrets` attached")
        if secret:
            return [secret, *self._secrets]
        else:
            return [*secrets, *self._secrets]

    def _add_function(self, function: _Function) -> _FunctionHandle:
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

        # We now need to create an actual handle.
        # This is a bit weird since the object isn't actually created yet,
        # but functions are weird and live and the global scope
        # These will be set with the correct object id when the app starts.
        function_handle = _FunctionHandle(function)
        self._function_handles[function.tag] = function_handle
        return function_handle

    @property
    def registered_functions(self) -> List[str]:
        return list(self._function_handles.keys())

    @decorator_with_options
    def function(
        self,
        raw_f=None,  # The decorated function
        *,
        image: _Image = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: Union[bool, _GPUConfig] = False,  # Whether a GPU is required
        rate_limit: Optional[RateLimit] = None,  # Optional RateLimit for the function
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Collection[_Mount] = (),
        shared_volumes: Dict[str, _SharedVolume] = {},
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MB. This is a soft limit.
        proxy: Optional[Ref] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[int] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        interactive: bool = False,  # Whether to run the function in interactive mode.
        _is_build_step: bool = False,  # Whether function is a build step; reserved for internal use.
        keep_warm: bool = False,  # Toggles an adaptively-sized warm pool for latency-sensitive apps.
        name: Optional[str] = None,  # Sets the Modal name of the function within the stub
        is_generator: Optional[bool] = None,  # If not set, it's inferred from the function signature
    ) -> _FunctionHandle:  # Function object - callable as a regular function within a Modal app
        """Decorator to register a new Modal function with this stub."""
        if image is None:
            image = self._get_default_image()
        info = FunctionInfo(raw_f, serialized=serialized, name_override=name)
        mounts = [*self._get_function_mounts(info), *mounts]
        secrets = self._get_function_secrets(raw_f, secret, secrets)

        if interactive:
            if self._pty_input_stream:
                warnings.warn(
                    "Running multiple interactive functions at the same time is not fully supported, and could lead to unexpected behavior."
                )
            else:
                self._blueprint["_pty_input_stream"] = _Queue()

        if is_generator is None:
            is_generator = inspect.isgeneratorfunction(raw_f) or inspect.isasyncgenfunction(raw_f)

        function = _Function(
            info,
            image=image,
            secrets=secrets,
            schedule=schedule,
            is_generator=is_generator,
            gpu=gpu,
            rate_limit=rate_limit,
            serialized=serialized,
            mounts=mounts,
            shared_volumes=shared_volumes,
            memory=memory,
            proxy=proxy,
            retries=retries,
            concurrency_limit=concurrency_limit,
            timeout=timeout,
            cpu=cpu,
            interactive=interactive,
            keep_warm=keep_warm,
            name=name,
        )

        if _is_build_step:
            # Don't add function to stub if it's a build step.
            return _FunctionHandle(function)

        return self._add_function(function)

    @decorator_with_options
    def generator(self, raw_f=None, **kwargs) -> _FunctionHandle:
        deprecation_warning(date(2022, 12, 1), "Stub.generator is deprecated. Use .function() instead.")
        kwargs.update(dict(is_generator=True))
        return self.function(raw_f, **kwargs)

    @decorator_with_options
    def webhook(
        self,
        raw_f,
        *,
        method: str = "GET",  # REST method for the created endpoint.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
        image: _Image = None,  # The image to run as the container for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: Union[bool, _GPUConfig] = False,  # Whether a GPU is required
        mounts: Collection[_Mount] = (),
        shared_volumes: Dict[str, _SharedVolume] = {},
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MB. This is a soft limit.
        proxy: Optional[Ref] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[int] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        keep_warm: bool = False,  # Toggles an adaptively-sized warm pool for latency-sensitive apps.
    ):
        """Register a basic web endpoint with this application.

        This is the simple way to create a web endpoint on Modal. The function
        behaves as a [FastAPI](https://fastapi.tiangolo.com/) handler and should
        return a response object to the caller.

        Endpoints created with `@stub.webhook` are meant to be simple, single
        request handlers and automatically have
        [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled.
        For more flexibility, use `@stub.asgi`.

        To learn how to use Modal with popular web frameworks, see the
        [guide on web endpoints](https://modal.com/docs/guide/webhooks).
        """
        if image is None:
            image = self._get_default_image()
        info = FunctionInfo(raw_f)
        mounts = [*self._get_function_mounts(info), *mounts]
        secrets = self._get_function_secrets(raw_f, secret, secrets)
        function = _Function(
            info,
            image=image,
            secrets=secrets,
            is_generator=True,
            gpu=gpu,
            mounts=mounts,
            shared_volumes=shared_volumes,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION, method=method, wait_for_response=wait_for_response
            ),
            cpu=cpu,
            memory=memory,
            proxy=proxy,
            retries=retries,
            concurrency_limit=concurrency_limit,
            timeout=timeout,
            keep_warm=keep_warm,
        )
        return self._add_function(function)

    @decorator_with_options
    def asgi(
        self,
        asgi_app,  # The asgi app
        *,
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
        image: _Image = None,  # The image to run as the container for the function
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: Union[bool, _GPUConfig] = False,  # Whether a GPU is required
        mounts: Collection[_Mount] = (),
        shared_volumes: Dict[str, _SharedVolume] = {},
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MB. This is a soft limit.
        proxy: Optional[Ref] = None,  # Reference to a Modal Proxy to use in front of this function.
        retries: Optional[int] = None,  # Number of times to retry each input in case of failure.
        concurrency_limit: Optional[int] = None,  # Limit for max concurrent containers running the function.
        timeout: Optional[int] = None,  # Maximum execution time of the function in seconds.
        keep_warm: bool = False,  # Toggles an adaptively-sized warm pool for latency-sensitive apps.
        _webhook_type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
    ):
        """Register an ASGI app with this application.

        Asynchronous Server Gateway Interface (ASGI) is a standard for Python
        synchronous and asynchronous apps, supported by all popular Python web
        libraries. This is an advanced decorator that gives full flexibility in
        defining one or more web endpoints on Modal.

        To learn how to use Modal with popular web frameworks, see the
        [guide on web endpoints](https://modal.com/docs/guide/webhooks).
        """
        if image is None:
            image = self._get_default_image()
        info = FunctionInfo(asgi_app)
        mounts = [*self._get_function_mounts(info), *mounts]
        secrets = self._get_function_secrets(asgi_app, secret, secrets)
        function = _Function(
            info,
            image=image,
            secrets=secrets,
            is_generator=True,
            gpu=gpu,
            mounts=mounts,
            shared_volumes=shared_volumes,
            webhook_config=api_pb2.WebhookConfig(type=_webhook_type, wait_for_response=wait_for_response),
            cpu=cpu,
            memory=memory,
            proxy=proxy,
            retries=retries,
            concurrency_limit=concurrency_limit,
            timeout=timeout,
            keep_warm=keep_warm,
        )
        return self._add_function(function)

    @decorator_with_options
    def wsgi(
        self,
        wsgi_app,
        **kwargs,
    ):
        """Exposes a WSGI app. For a list of arguments, see the documentation for `asgi`."""
        asgi_decorator = self.asgi(_webhook_type=api_pb2.WEBHOOK_TYPE_WSGI_APP, **kwargs)
        return asgi_decorator(wsgi_app)

    async def interactive_shell(self, cmd=None, image=None, **kwargs):
        """Run an interactive shell (like `bash`) within the image for this app.

        This is useful for online debugging and interactive exploration of the
        contents of this image. If `cmd` is optionally provided, it will be run
        instead of the default shell inside this image.

        **Example**

        ```python
        import modal

        stub = modal.Stub(image=modal.Image.debian_slim().apt_install(["vim"]))

        if __name__ == "__main__":
            stub.interactive_shell("/bin/bash")
        ```

        Or alternatively:

        ```python
        import modal

        stub = modal.Stub()
        app_image = modal.Image.debian_slim().apt_install(["vim"])

        if __name__ == "__main__":
            stub.interactive_shell(cmd="/bin/bash", image=app_image)
        ```
        """
        wrapped_fn = self.function(interactive=True, timeout=86400, image=image or self._get_default_image(), **kwargs)(
            exec_cmd
        )

        async with self.run():
            await wrapped_fn(cmd)


Stub, AioStub = synchronize_apis(_Stub)
