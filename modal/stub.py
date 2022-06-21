import asyncio
import os
import sys
from typing import Collection, Dict, Optional, Union

from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext, synchronize_apis, synchronizer
from modal_utils.decorator_utils import decorator_with_options

from ._function_utils import FunctionInfo
from ._output import OutputManager, step_completed, step_progress
from .client import _Client
from .config import config
from .exception import InvalidError, VersionError
from .functions import _Function
from .image import _DebianSlim, _Image
from .mount import _create_client_mount, _Mount, client_mount_name
from .object import Object, Ref, ref
from .rate_limit import RateLimit
from .running_app import _RunningApp, container_app, is_local
from .schedule import Schedule
from .secret import _Secret
from .shared_volume import _SharedVolume


class _Stub:
    """An App manages Objects (Functions, Images, Secrets, Schedules etc.) associated with your applications

    A stub is a description of how to create an app.

    The App has three main responsibilities:
    * Syncing of identities across processes (your local Python interpreter and every Modal worker active in your application)
    * Making Objects stay alive and not be garbage collected for as long as the app lives (see App lifetime below)
    * Manage log collection for everything that happens inside your code

    **Registering Functions with an app**

    The most common way to explicitly register an Object with an app is through the `app.function()` decorator.
    It both registers the annotated function itself and other passed objects like Schedules and Secrets with the
    specified app:

    ```python
    import modal

    stub = modal.Stub()

    @stub.function(secret=modal.ref("some_secret"), schedule=modal.Period(days=1))
    def foo():
        ...
    ```
    In this example, both `foo`, the secret and the schedule are registered with the app.
    """

    _blueprint: Dict[str, Object]

    def __init__(self, name=None, **blueprint):
        if name is None:
            name = self._infer_app_name()
        self._name = name
        self._blueprint = blueprint
        self._default_image = _DebianSlim()
        self._client_mount = None
        self._function_mounts = {}
        super().__init__()

    @property
    def name(self):
        return self._name

    def _infer_app_name(self):
        script_filename = os.path.split(sys.argv[0])[-1]
        args = [script_filename] + sys.argv[1:]
        return " ".join(args)

    def __getitem__(self, tag: str):
        assert isinstance(tag, str)
        # Return a reference to an object that will be created in the future
        return ref(None, tag)

    def __setitem__(self, tag, obj):
        self._blueprint[tag] = obj

    def is_inside(self, image: Optional[Ref] = None):
        """Returns if the current code block is executed within the `image` container"""
        # TODO: this should just be a global function
        if is_local():
            return False
        else:
            return container_app.is_inside(image)

    @synchronizer.asynccontextmanager
    async def _run(self, client, output_mgr, existing_app_id, last_log_entry_id=None, name=None):
        if existing_app_id is not None:
            running_app = await _RunningApp.init_existing(self, client, existing_app_id)
        else:
            running_app = await _RunningApp.init_new(self, client, name if name is not None else self.name)

        # Start tracking logs and yield context
        async with TaskContext(grace=config["logs_timeout"]) as tc:
            with output_mgr.ctx_if_visible(output_mgr.make_live(step_progress("Initializing..."))):
                live_task_status = output_mgr.make_live(step_progress("Running app..."))
                app_id = running_app.app_id
                tc.create_task(output_mgr.get_logs_loop(app_id, client, live_task_status, last_log_entry_id or ""))
            output_mgr.print_if_visible(step_completed("Initialized."))

            try:
                # Create all members
                progress = Tree(step_progress("Creating objects..."), guide_style="gray50")
                with output_mgr.ctx_if_visible(output_mgr.make_live(progress)):
                    await running_app.create_all_objects(progress)
                progress.label = step_completed("Created objects.")
                output_mgr.print_if_visible(progress)

                # Yield to context
                with output_mgr.ctx_if_visible(live_task_status):
                    yield running_app

            finally:
                await running_app.disconnect()

        output_mgr.print_if_visible(step_completed("App completed."))

    @synchronizer.asynccontextmanager
    async def _get_client(self, client=None):
        if client is None:
            async with _Client.from_env() as client:
                yield client
        else:
            yield client

    @synchronizer.asynccontextmanager
    async def run(self, client=None, stdout=None, show_progress=None):
        async with self._get_client(client) as client:
            output_mgr = OutputManager(stdout, show_progress)
            async with self._run(client, output_mgr, None) as running_app:
                yield running_app

    async def run_forever(self, client=None, stdout=None, show_progress=None):
        async with self._get_client(client) as client:
            output_mgr = OutputManager(stdout, show_progress)
            async with self._run(client, output_mgr, None):
                timeout = config["run_forever_timeout"]
                if timeout:
                    output_mgr.print_if_visible(step_completed(f"Running for {timeout} seconds... hit Ctrl-C to stop!"))
                    await asyncio.sleep(timeout)
                else:
                    output_mgr.print_if_visible(step_completed("Running forever... hit Ctrl-C to stop!"))
                    while True:
                        await asyncio.sleep(1.0)

    async def deploy(
        self,
        name: str = None,  # Unique name of the deployment. Subsequent deploys with the same name overwrites previous ones. Falls back to the app name
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
        client=None,
        stdout=None,
        show_progress=None,
    ):
        """Deploys and exports objects in the app

        Usage:
        ```python
        if __name__ == "__main__":
            stub.deploy()
        ```

        Deployment has two primary purposes:
        * Persists all of the objects (Functions, Images, Schedules etc.) in the app, allowing them to live past the current app run
          Notably for Schedules this enables headless "cron"-like functionality where scheduled functions continue to be invoked after
          the client has closed.
        * Allows for certain of these objects, *deployment objects*, to be referred to and used by other apps
        """
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

        async with self._get_client(client) as client:
            # Look up any existing deployment
            app_req = api_pb2.AppGetByDeploymentNameRequest(name=name, namespace=namespace, client_id=client.client_id)
            app_resp = await client.stub.AppGetByDeploymentName(app_req)
            existing_app_id = app_resp.app_id or None
            last_log_entry_id = app_resp.last_log_entry_id

            # The `_run` method contains the logic for starting and running an app
            output_mgr = OutputManager(stdout, show_progress)
            async with self._run(client, output_mgr, existing_app_id, last_log_entry_id, name=name) as running_app:
                deploy_req = api_pb2.AppDeployRequest(
                    app_id=running_app._app_id,
                    name=name,
                    namespace=namespace,
                )
                await client.stub.AppDeploy(deploy_req)
                return running_app._app_id

    def _get_default_image(self):
        if "image" in self._blueprint:
            return self._blueprint["image"]
        else:
            return self._default_image

    def _get_function_mounts(self, raw_f):
        mounts = []

        # Create client mount
        if self._client_mount is None:
            if config["sync_entrypoint"]:
                self._client_mount = _create_client_mount()
            else:
                self._client_mount = ref(client_mount_name(), namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
        mounts.append(self._client_mount)

        # Create function mounts
        info = FunctionInfo(raw_f)
        for key, mount in info.get_mounts().items():
            if key not in self._function_mounts:
                self._function_mounts[key] = mount
            mounts.append(self._function_mounts[key])

        return mounts

    @decorator_with_options
    def function(
        self,
        raw_f=None,  # The decorated function
        *,
        image: Union[_Image, Ref] = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[
            Union[_Secret, Ref]
        ] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Union[_Secret, Ref]] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        rate_limit: Optional[RateLimit] = None,  # Optional RateLimit for the function
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Collection[Union[_Mount, Ref]] = (),
        shared_volumes: Dict[str, Union[_SharedVolume, Ref]] = {},
        memory: Optional[int] = None,  # How much memory to request, in MB. This is a soft limit.
    ) -> _Function:  # Function object - callable as a regular function within a Modal app
        """Decorator to create Modal functions"""
        if image is None:
            image = self._get_default_image()
        mounts = [*self._get_function_mounts(raw_f), *mounts]
        function = _Function(
            raw_f,
            image=image,
            secret=secret,
            secrets=secrets,
            schedule=schedule,
            is_generator=False,
            gpu=gpu,
            rate_limit=rate_limit,
            serialized=serialized,
            mounts=mounts,
            shared_volumes=shared_volumes,
            memory=memory,
        )
        self._blueprint[function.tag] = function
        return function

    @decorator_with_options
    def generator(
        self,
        raw_f=None,  # The decorated function
        *,
        image: Union[_Image, Ref] = None,  # The image to run as the container for the function
        secret: Optional[
            Union[_Secret, Ref]
        ] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Union[_Secret, Ref]] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        rate_limit: Optional[RateLimit] = None,  # Optional RateLimit for the function
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Collection[Union[_Mount, Ref]] = (),
        shared_volumes: Dict[str, Union[_SharedVolume, Ref]] = {},
        memory: Optional[int] = None,  # How much memory to request, in MB. This is a soft limit.
    ) -> _Function:
        """Decorator to create Modal generators"""
        if image is None:
            image = self._get_default_image()
        mounts = [*self._get_function_mounts(raw_f), *mounts]
        function = _Function(
            raw_f,
            image=image,
            secret=secret,
            secrets=secrets,
            is_generator=True,
            gpu=gpu,
            rate_limit=rate_limit,
            serialized=serialized,
            mounts=mounts,
            shared_volumes=shared_volumes,
            memory=memory,
        )
        self._blueprint[function.tag] = function
        return function

    @decorator_with_options
    def asgi(
        self,
        asgi_app,  # The asgi app
        *,
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
        image: Union[_Image, Ref] = None,  # The image to run as the container for the function
        secret: Optional[
            Union[_Secret, Ref]
        ] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Union[_Secret, Ref]] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        mounts: Collection[Union[_Mount, Ref]] = (),
        shared_volumes: Dict[str, Union[_SharedVolume, Ref]] = {},
        memory: Optional[int] = None,  # How much memory to request, in MB. This is a soft limit.
    ):
        if image is None:
            image = self._get_default_image()
        mounts = [*self._get_function_mounts(asgi_app), *mounts]
        function = _Function(
            asgi_app,
            image=image,
            secret=secret,
            secrets=secrets,
            is_generator=True,
            gpu=gpu,
            mounts=mounts,
            shared_volumes=shared_volumes,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_ASGI_APP, wait_for_response=wait_for_response
            ),
            memory=memory,
        )
        self._blueprint[function.tag] = function
        return function

    @decorator_with_options
    def webhook(
        self,
        raw_f,
        *,
        method: str = "GET",  # REST method for the created endpoint.
        wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
        image: Union[_Image, Ref] = None,  # The image to run as the container for the function
        secret: Optional[
            Union[_Secret, Ref]
        ] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Union[_Secret, Ref]] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        mounts: Collection[Union[_Mount, Ref]] = (),
        shared_volumes: Dict[str, Union[_SharedVolume, Ref]] = {},
        memory: Optional[int] = None,  # How much memory to request, in MB. This is a soft limit.
    ):
        if image is None:
            image = self._get_default_image()
        mounts = [*self._get_function_mounts(raw_f), *mounts]
        function = _Function(
            raw_f,
            image=image,
            secret=secret,
            secrets=secrets,
            is_generator=True,
            gpu=gpu,
            mounts=mounts,
            shared_volumes=shared_volumes,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION, method=method, wait_for_response=wait_for_response
            ),
            memory=memory,
        )
        self._blueprint[function.tag] = function
        return function

    async def interactive_shell(self, cmd=None, mounts=[], secrets=[], image_ref=None, shared_volumes={}):
        """Run `cmd` interactively within this image. Similar to `docker run -it --entrypoint={cmd}`.

        If `cmd` is `None`, this falls back to the default shell within the image.
        """
        from ._image_pty import image_pty

        await image_pty(image_ref or self._image, self, cmd, mounts, secrets, shared_volumes)


Stub, AioStub = synchronize_apis(_Stub)


class _App(_Stub):
    """Deprecated class, use Stub instead."""

    def __init__(self, name=None, *, image=None):
        raise VersionError("App is deprecated and will be removed in 0.0.18. Please use Stub instead")
        super().__init__(name, image=image)


App, AioApp = synchronize_apis(_App)
