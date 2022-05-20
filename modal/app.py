import asyncio
import os
import sys
from typing import Collection, Dict, Optional, Union

from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext, synchronize_apis, synchronizer
from modal_utils.decorator_utils import decorator_with_options

from ._blueprint import Blueprint
from ._function_utils import FunctionInfo
from ._output import OutputManager, step_completed, step_progress
from .client import _Client
from .config import config, logger
from .exception import InvalidError, NotFoundError
from .functions import _Function, _FunctionProxy
from .image import _DebianSlim, _Image
from .mount import MODAL_CLIENT_MOUNT_NAME, _create_client_mount, _Mount
from .object import Object, Ref, ref
from .rate_limit import RateLimit
from .schedule import Schedule
from .secret import _Secret


class _RunningApp:
    _tag_to_object: Dict[str, Object]
    _tag_to_existing_id: Dict[str, str]
    _client: _Client
    _app_id: str

    def __init__(
        self,
        client: _Client,
        app_id: str,
        tag_to_object: Optional[Dict[str, Object]] = None,
        tag_to_existing_id: Optional[Dict[str, str]] = None,
    ):
        self._app_id = app_id
        self._client = client
        self._tag_to_object = tag_to_object or {}
        self._tag_to_existing_id = tag_to_existing_id or {}

    @property
    def client(self):
        return self._client

    @property
    def app_id(self):
        return self._app_id

    async def lookup(self, obj: Object) -> str:
        """Takes a Ref object and looks up its id.

        It's either an object defined locally on this app, or one defined on a separate app
        """
        if not isinstance(obj, Ref):
            # TODO: explain these exception more, since I think it might be a common issue
            raise InvalidError(f"Object {obj} has no label. Make sure every object is defined on the app.")
        if not obj.app_name and not obj.tag:
            raise InvalidError(f"Object {obj} is a malformed reference to nothing.")

        if obj.app_name is not None:
            # A different app
            object_id = await self._include(obj.app_name, obj.tag, obj.namespace)
        else:
            # Same app, an object that was created earlier
            obj = self._tag_to_object[obj.tag]
            object_id = obj.object_id

        assert object_id
        return object_id

    async def _include(self, app_name: str, tag: Optional[str], namespace):
        """Internal method to resolve to an object id."""
        request = api_pb2.AppIncludeObjectRequest(
            app_id=self._app_id,
            name=app_name,  # TODO: update the protobuf field name
            object_label=tag,  # TODO: update the protobuf field name
            namespace=namespace,
        )
        response = await self._client.stub.AppIncludeObject(request)
        if not response.object_id:
            obj_repr = app_name
            if tag is not None:
                obj_repr += f".{tag}"
            if namespace != api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT:
                obj_repr += f" (namespace {api_pb2.DeploymentNamespace.Name(namespace)})"
            # TODO: disambiguate between app not found and object not found?
            err_msg = f"Could not find object {obj_repr}"
            raise NotFoundError(err_msg, obj_repr)
        return response.object_id

    async def include(self, app_name, tag=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
        """Looks up an object and return a newly constructed one."""
        object_id = await self._include(app_name, tag, namespace)
        return Object.from_id(object_id, self)

    async def _create_object(self, obj: Object, progress: Tree, existing_object_id: Optional[str] = None) -> str:
        """Takes an object as input, create it, and return an object id."""
        creating_message = obj.get_creating_message()
        if creating_message is not None:
            step_node = progress.add(step_progress(creating_message))

        if isinstance(obj, Ref):
            assert obj.app_name is not None
            # A different app
            object_id = await self._include(obj.app_name, obj.tag, obj.namespace)

        else:
            # Create object
            object_id = await obj.load(self, existing_object_id)
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
        if object_id is None:
            raise Exception(f"object_id for object of type {type(obj)} is None")

        if creating_message is not None:
            created_message = obj.get_created_message()
            assert created_message is not None
            step_node.label = step_completed(created_message, is_substep=True)

        return object_id

    async def create_all_objects(self, blueprint: Blueprint, progress: Tree):
        """Create objects that have been defined but not created on the server."""
        # Instead of doing a topological sort here, we rely on a sort of dumb "trick".
        # Functions are the only objects that "depend" on other objects, so we make sure
        # they are built last. In the future we might have some more complicated structure
        # where we actually have to model out the DAG
        tags = [tag for tag, obj in blueprint.get_objects()]
        tags.sort(key=lambda obj: obj.startswith("fu-"))
        for tag in tags:
            obj = blueprint.get_object(tag)
            existing_object_id = self._tag_to_existing_id.get(tag)
            logger.debug(f"Creating object {tag} with existing id {existing_object_id}")
            object_id = await self._create_object(obj, progress, existing_object_id)
            self._tag_to_object[tag] = Object.from_id(object_id, self)

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
            self._tag_to_object[tag] = Object.from_id(object_id, self)

        return self

    @staticmethod
    async def init_existing(client, existing_app_id):
        # Get all the objects first
        obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
        obj_resp = await client.stub.AppGetObjects(obj_req)
        return _RunningApp(client, existing_app_id, tag_to_existing_id=dict(obj_resp.object_ids))

    @staticmethod
    async def init_new(client, name):
        # Start app
        # TODO(erikbern): maybe this should happen outside of this method?
        app_req = api_pb2.AppCreateRequest(client_id=client.client_id, name=name)
        app_resp = await client.stub.AppCreate(app_req)
        return _RunningApp(client, app_resp.app_id)

    @staticmethod
    def reset_container():
        global _is_container_app
        _is_container_app = False


RunningApp, AioRunningApp = synchronize_apis(_RunningApp)

_is_container_app = False
_container_app = _RunningApp(None, None)
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

    def __init__(self, name=None, *, image=None):
        # TODO: we take a name in the app constructor, that can be different from the deployment name passed in later. Simplify this.
        self._name = name
        self.deployment_name = None
        self._blueprint = Blueprint()
        self._image = image
        self._running_app = None
        super().__init__()

    # needs to be a function since synchronicity hides other attributes.
    def provided_name(self):
        return self._name

    @property
    def name(self):
        return self._name or self._infer_app_name()

    def _infer_app_name(self):
        script_filename = os.path.split(sys.argv[0])[-1]
        args = [script_filename] + sys.argv[1:]
        return " ".join(args)

    def __getitem__(self, tag: str):
        assert isinstance(tag, str)
        # Return a reference to an object that will be created in the future
        return ref(None, tag)

    def __setitem__(self, tag, obj):
        self._blueprint.register(tag, obj)

    def is_inside(self, image: Optional[Union[Ref, _Image]] = None):
        if not _is_container_app:
            return False
        if image is None:
            obj = _container_app._tag_to_object.get("_image")
        elif isinstance(image, Ref):
            obj = _container_app._tag_to_object.get(image.tag)
        elif isinstance(image, _Image):
            obj = image
        assert isinstance(obj, _Image)
        return obj._is_inside()

    @synchronizer.asynccontextmanager
    async def _run(self, client, output_mgr, existing_app_id, last_log_entry_id=None):
        # TOOD: use something smarter than checking for the .client to exists in order to prevent
        # race conditions here!
        try:
            if existing_app_id is not None:
                self._running_app = await _RunningApp.init_existing(client, existing_app_id)
            else:
                self._running_app = await _RunningApp.init_new(client, self.name)

            # Start tracking logs and yield context
            async with TaskContext(grace=config["logs_timeout"]) as tc:
                with output_mgr.ctx_if_visible(output_mgr.make_live(step_progress("Initializing..."))):
                    live_task_status = output_mgr.make_live(step_progress("Running app..."))
                    app_id = self._running_app.app_id
                    tc.create_task(output_mgr.get_logs_loop(app_id, client, live_task_status, last_log_entry_id or ""))
                output_mgr.print_if_visible(step_completed("Initialized."))

                try:
                    # Create all members
                    progress = Tree(step_progress("Creating objects..."), guide_style="gray50")
                    with output_mgr.ctx_if_visible(output_mgr.make_live(progress)):
                        await self._running_app.create_all_objects(self._blueprint, progress)
                    progress.label = step_completed("Created objects.")
                    output_mgr.print_if_visible(progress)
                    with output_mgr.ctx_if_visible(live_task_status):
                        yield self._running_app

                finally:
                    await self._running_app.disconnect()

            output_mgr.print_if_visible(step_completed("App completed."))

        finally:
            self._running_app = None

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

        self.deployment_name = name

        async with self._get_client(client) as client:
            # Look up any existing deployment
            app_req = api_pb2.AppGetByDeploymentNameRequest(name=name, namespace=namespace, client_id=client.client_id)
            app_resp = await client.stub.AppGetByDeploymentName(app_req)
            existing_app_id = app_resp.app_id or None
            last_log_entry_id = app_resp.last_log_entry_id

            # The `_run` method contains the logic for starting and running an app
            output_mgr = OutputManager(stdout, show_progress)
            async with self._run(client, output_mgr, existing_app_id, last_log_entry_id) as running_app:
                # TODO: this could be simplified in case it's the same app id as previously
                deploy_req = api_pb2.AppDeployRequest(
                    app_id=running_app._app_id,
                    name=name,
                    namespace=namespace,
                )
                await client.stub.AppDeploy(deploy_req)
                return running_app._app_id

    def _register_function(self, function):
        self._blueprint.register(function.tag, function)
        function_proxy = _FunctionProxy(function, self, function.tag)
        return function_proxy

    def _get_default_image(self):
        # TODO(erikbern): instead of writing this to the same namespace
        # as the user's objects, we could use sub-blueprints in the future
        if not self._blueprint.has_object("_image"):
            if self._image is None:
                image = _DebianSlim()
            else:
                image = self._image
            self._blueprint.register("_image", image)
        return ref(None, "_image")

    def _get_function_mounts(self, raw_f):
        mounts = []

        # Create client mount
        if not self._blueprint.has_object("_client_mount"):
            if config["sync_entrypoint"]:
                client_mount = _create_client_mount()
            else:
                client_mount = ref(MODAL_CLIENT_MOUNT_NAME, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
            self._blueprint.register("_client_mount", client_mount)
        mounts.append(ref(None, "_client_mount"))

        # Create function mounts
        info = FunctionInfo(raw_f)
        for key, mount in info.get_mounts().items():
            if not self._blueprint.has_object(key):
                self._blueprint.register(key, mount)
            mounts.append(ref(None, key))

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
        return self._register_function(function)

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
        return self._register_function(function)

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
            is_generator=False,
            gpu=gpu,
            mounts=mounts,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_ASGI_APP, wait_for_response=wait_for_response
            ),
        )
        return self._register_function(function)

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
            is_generator=False,
            gpu=gpu,
            mounts=mounts,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION, method=method, wait_for_response=wait_for_response
            ),
        )
        return self._register_function(function)

    async def interactive_shell(self, image_ref, cmd=None, mounts=[], secrets=[]):
        """Run `cmd` interactively within this image. Similar to `docker run -it --entrypoint={cmd}`.

        If `cmd` is `None`, this falls back to the default shell within the image.
        """
        from ._image_pty import image_pty

        await image_pty(image_ref, self, cmd, mounts, secrets)


App, AioApp = synchronize_apis(_App)


def is_local() -> bool:
    """Returns whether we're running in the cloud or not."""
    return not _is_container_app
