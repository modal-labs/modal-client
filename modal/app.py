import asyncio
import functools
import os
import sys
import warnings
from typing import Collection, Dict, Optional, Union

from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext, synchronize_apis, synchronizer
from modal_utils.decorator_utils import decorator_with_options

from ._function_utils import FunctionInfo
from ._output import OutputManager, step_completed, step_progress
from .client import _Client
from .config import config, logger
from .exception import InvalidError, NotFoundError
from .functions import _Function
from .image import _DebianSlim, _Image
from .mount import MODAL_CLIENT_MOUNT_NAME, _create_client_mount, _Mount
from .object import Object, Ref, ref
from .rate_limit import RateLimit
from .schedule import Schedule
from .secret import _Secret


async def _lookup_to_id(app_name: str, tag: str, namespace, client: _Client) -> str:
    """Internal method to resolve to an object id."""
    request = api_pb2.AppLookupObjectRequest(
        app_name=app_name,
        object_tag=tag,
        namespace=namespace,
    )
    response = await client.stub.AppLookupObject(request)
    if not response.object_id:
        raise NotFoundError(response.error_message)
    return response.object_id


async def _lookup(
    app_name: str,
    tag: Optional[str] = None,
    namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
    client: Optional[_Client] = None,
) -> Object:
    if client is None:
        client = _Client.from_env()
    object_id = await _lookup_to_id(app_name, tag, namespace, client)
    return Object.from_id(object_id, client)


lookup, aio_lookup = synchronize_apis(_lookup)


class _RunningApp:
    _tag_to_object: Dict[str, Object]
    _tag_to_existing_id: Dict[str, str]
    _seed_to_object_id: Dict[str, str]
    _client: _Client
    _app_id: str

    def __init__(
        self,
        app: "_App",
        client: _Client,
        app_id: str,
        tag_to_object: Optional[Dict[str, Object]] = None,
        tag_to_existing_id: Optional[Dict[str, str]] = None,
    ):
        self._app = app
        self._app_id = app_id
        self._client = client
        self._tag_to_object = tag_to_object or {}
        self._tag_to_existing_id = tag_to_existing_id or {}
        self._seed_to_object_id = {}

    @property
    def client(self):
        return self._client

    @property
    def app_id(self):
        return self._app_id

    async def include(self, app_name, tag=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
        """Looks up an object and return a newly constructed one."""
        warnings.warn("RunningApp.include is deprecated. Use modal.lookup instead", DeprecationWarning)
        return await _lookup(app_name, tag, namespace, self.client)

    async def load(self, obj: Object, progress: Optional[Tree] = None, existing_object_id: Optional[str] = None) -> str:
        """Takes an object as input, create it, and return an object id."""
        if progress:
            creating_message = obj.get_creating_message()
            if creating_message is not None:
                step_node = progress.add(step_progress(creating_message))

        if isinstance(obj, Ref):
            # TODO: should we just move this code to the Ref class?
            if obj.app_name is not None:
                # A different app
                object_id = await _lookup_to_id(obj.app_name, obj.tag, obj.namespace, self._client)
            else:
                # Same app
                if obj.tag in self._tag_to_object:
                    object_id = self._tag_to_object[obj.tag].object_id
                else:
                    real_obj = self._app._blueprint[obj.tag]
                    existing_object_id = self._tag_to_existing_id.get(obj.tag)
                    object_id = await self.load(real_obj, progress, existing_object_id)
                    self._tag_to_object[obj.tag] = Object.from_id(object_id, self.client)
        else:
            # Create object
            if obj.seed in self._seed_to_object_id:
                object_id = self._seed_to_object_id[obj.seed]
            else:
                load = functools.partial(self.load, progress=progress)
                object_id = await obj.load(load, self.client, self.app_id, existing_object_id)
                if existing_object_id is not None and object_id != existing_object_id:
                    # TODO(erikbern): this is a very ugly fix to a problem that's on the server side.
                    # Unlike every other object, images are not assigned random ids, but rather an
                    # id given by the hash of its contents. This means we can't _force_ an image to
                    # have a particular id. The better solution is probably to separate "images"
                    # from "image definitions" or something like that, but that's a big project.
                    if not existing_object_id.startswith("im-"):
                        raise Exception(
                            f"Tried creating an object using existing id {existing_object_id} but it has id {object_id}"
                        )
                self._seed_to_object_id[obj.seed] = object_id

        if object_id is None:
            raise Exception(f"object_id for object of type {type(obj)} is None")

        if progress:
            if creating_message is not None:
                created_message = obj.get_created_message()
                assert created_message is not None
                step_node.label = step_completed(created_message, is_substep=True)

        return object_id

    async def create_all_objects(self, progress: Tree):
        """Create objects that have been defined but not created on the server."""
        for tag in self._app._blueprint.keys():
            obj = ref(None, tag)
            await self.load(obj, progress)

        # Create the app (and send a list of all tagged obs)
        # TODO(erikbern): we should delete objects from a previous version that are no longer needed
        # We just delete them from the app, but the actual objects will stay around
        object_ids = {tag: obj.object_id for tag, obj in self._tag_to_object.items()}
        req_set = api_pb2.AppSetObjectsRequest(
            app_id=self._app_id,
            object_ids=object_ids,
            client_id=self._client.client_id,
        )
        await self._client.stub.AppSetObjects(req_set)

        # Update all functions client-side to point to the running app
        for obj in self._app._blueprint.values():
            if isinstance(obj, _Function):
                obj.set_local_running_app(self)

    async def disconnect(self):
        # Stop app server-side. This causes:
        # 1. Server to kill any running task
        # 2. Logs to drain (stopping the _get_logs_loop coroutine)
        logger.debug("Stopping the app server-side")
        req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=self._app_id)
        await self._client.stub.AppClientDisconnect(req_disconnect)

    def __getitem__(self, tag):
        return self._tag_to_object[tag]

    @staticmethod
    async def init_container(client, app_id, task_id):
        """Used by the container to bootstrap the app and all its objects."""
        # This is a bit of a hacky thing:
        global _container_app, _is_container_app
        _is_container_app = True
        self = _container_app
        self._client = client
        self._app_id = app_id

        req = api_pb2.AppGetObjectsRequest(app_id=app_id, task_id=task_id)
        resp = await self._client.stub.AppGetObjects(req)
        for (
            tag,
            object_id,
        ) in resp.object_ids.items():
            self._tag_to_object[tag] = Object.from_id(object_id, self._client)

        return self

    @staticmethod
    async def init_existing(app, client, existing_app_id):
        # Get all the objects first
        obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
        obj_resp = await client.stub.AppGetObjects(obj_req)
        return _RunningApp(app, client, existing_app_id, tag_to_existing_id=dict(obj_resp.object_ids))

    @staticmethod
    async def init_new(app, client, name):
        # Start app
        # TODO(erikbern): maybe this should happen outside of this method?
        app_req = api_pb2.AppCreateRequest(client_id=client.client_id, name=name)
        app_resp = await client.stub.AppCreate(app_req)
        return _RunningApp(app, client, app_resp.app_id)

    @staticmethod
    def reset_container():
        global _is_container_app
        _is_container_app = False


RunningApp, AioRunningApp = synchronize_apis(_RunningApp)

_is_container_app = False
_container_app = _RunningApp(None, None, None)
container_app, aio_container_app = synchronize_apis(_container_app)
assert isinstance(container_app, RunningApp)
assert isinstance(aio_container_app, AioRunningApp)


class _App:
    """An App manages Objects (Functions, Images, Secrets, Schedules etc.) associated with your applications

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

    app = modal.App()

    @app.function(secret=modal.ref("some_secret"), schedule=modal.Period(days=1))
    def foo():
        ...
    ```
    In this example, both `foo`, the secret and the schedule are registered with the app.
    """

    _blueprint: Dict[str, Object]

    def __init__(self, name=None, *, image=None):
        if name is None:
            name = self._infer_app_name()
        self._name = name
        self._image = image
        self._blueprint = {}
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

    def is_inside(self, image: Optional[Union[Ref, _Image]] = None):
        if not _is_container_app:
            return False
        # TODO(erikbern): figure this out
        # if image is None:
        #    obj = _container_app._tag_to_object.get("_image")
        elif isinstance(image, Ref):
            obj = _container_app._tag_to_object.get(image.tag)
        elif isinstance(image, _Image):
            obj = image
        assert isinstance(obj, _Image)
        return obj._is_inside()

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
            app.deploy()
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
                'app.deploy("some_name")\n\n'
                "or\n"
                'app = App("some-name")'
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
        if self._image is None:
            self._image = _DebianSlim()
        return self._image

    def _get_function_mounts(self, raw_f):
        mounts = []

        # Create client mount
        if self._client_mount is None:
            if config["sync_entrypoint"]:
                self._client_mount = _create_client_mount()
            else:
                self._client_mount = ref(MODAL_CLIENT_MOUNT_NAME, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
        mounts.append(self._client_mount)

        # Create function mounts
        info = FunctionInfo(raw_f)
        for key, mount in info.get_mounts().items():
            if key not in self._function_mounts:
                self._function_mounts[key] = mount
            mounts.append(mount)

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
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_ASGI_APP, wait_for_response=wait_for_response
            ),
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
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION, method=method, wait_for_response=wait_for_response
            ),
        )
        self._blueprint[function.tag] = function
        return function

    async def interactive_shell(self, cmd=None, mounts=[], secrets=[], image_ref=None):
        """Run `cmd` interactively within this image. Similar to `docker run -it --entrypoint={cmd}`.

        If `cmd` is `None`, this falls back to the default shell within the image.
        """
        from ._image_pty import image_pty

        await image_pty(image_ref or self._image, self, cmd, mounts, secrets)


App, AioApp = synchronize_apis(_App)


def is_local() -> bool:
    """Returns whether we're running in the cloud or not."""
    return not _is_container_app
