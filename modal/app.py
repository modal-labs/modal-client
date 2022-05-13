import asyncio
import functools
import io
import os
import sys
from typing import Collection, Dict, Optional

import grpc
from rich.live import Live
from rich.tree import Tree

from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext, synchronize_apis, synchronizer
from modal_utils.decorator_utils import decorator_with_options

from ._app_singleton import get_container_app, set_container_app
from ._app_state import AppState
from ._blueprint import Blueprint
from ._output import LineBufferedOutput, OutputManager, step_completed, step_progress
from ._serialization import Pickler, Unpickler
from .client import _Client
from .config import config, logger
from .exception import InvalidError, NotFoundError
from .functions import _Function, _FunctionProxy
from .image import _DebianSlim, _Image
from .mount import _Mount
from .object import Object
from .rate_limit import RateLimit
from .schedule import Schedule
from .secret import Secret


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

    _tag_to_object: Dict[str, Object]
    _tag_to_existing_id: Dict[str, str]

    @classmethod
    def _initialize_container_app(cls):
        set_container_app(super().__new__(cls))

    def __new__(cls, *args, **kwargs):
        singleton = get_container_app()
        if singleton is not None and cls == _App:
            # If there's a singleton app, just return it for everything
            assert isinstance(singleton, cls)
            return singleton
        else:
            # Refer to the normal constructor
            app = super().__new__(cls)
            return app

    def __init__(self, name=None):
        if "_initialized" in self.__dict__:
            return  # Prevent re-initialization with the singleton

        self._initialized = True
        self._app_id = None
        self.client = None
        # TODO: we take a name in the app constructor, that can be different from the deployment name passed in later. Simplify this.
        self._name = name
        self.deployment_name = None
        self.state = AppState.NONE
        self._tag_to_object = {}
        self._tag_to_existing_id = {}
        self._blueprint = Blueprint()
        self._task_states: Dict[str, int] = {}
        self._progress: Optional[Tree] = None
        super().__init__()

    # needs to be a function since synchronicity hides other attributes.
    def provided_name(self):
        return self._name

    @property
    def name(self):
        return self._name or self._infer_app_name()

    @property
    def app_id(self):
        return self._app_id

    def _infer_app_name(self):
        script_filename = os.path.split(sys.argv[0])[-1]
        args = [script_filename] + sys.argv[1:]
        return " ".join(args)

    def _register_object(self, tag: str, obj: Object):
        """Registers an object to be created by the app so that it's available in modal.

        This is only used by factories and functions."""
        if get_container_app():
            return
        if self.state != AppState.NONE:
            raise InvalidError(f"Can only register objects on a app that's not running (state = {self.state}")
        # TODO(erikbern): there was a special case with a comment here about double-loading and cloudpickle,
        # but I have a feeling it's no longer an issue, so I remved it for now.
        self._blueprint.register(tag, obj)

    def _update_task_state(self, task_id: str, state: int) -> str:
        """Updates the state of a task, returning the new task status string."""
        self._task_states[task_id] = state

        all_states = self._task_states.values()
        states_set = set(all_states)

        def tasks_at_state(state):
            return sum(x == state for x in all_states)

        # The most advanced state that's present informs the message.
        if api_pb2.TASK_STATE_RUNNING in states_set:
            tasks_running = tasks_at_state(api_pb2.TASK_STATE_RUNNING)
            tasks_loading = tasks_at_state(api_pb2.TASK_STATE_LOADING_IMAGE)
            return f"Running ({tasks_running}/{tasks_running + tasks_loading} containers in use)..."
        elif api_pb2.TASK_STATE_LOADING_IMAGE in states_set:
            tasks_loading = tasks_at_state(api_pb2.TASK_STATE_LOADING_IMAGE)
            return f"Loading images ({tasks_loading} containers initializing)..."
        elif api_pb2.TASK_STATE_WORKER_ASSIGNED in states_set:
            return "Worker assigned..."
        elif api_pb2.TASK_STATE_QUEUED in states_set:
            return "Tasks queued..."
        else:
            return "Tasks created..."

    async def _get_logs_loop(self, output_mgr: OutputManager, live_task_status: Live, last_log_batch_entry_id: str):
        async def _get_logs():
            nonlocal last_log_batch_entry_id

            request = api_pb2.AppGetLogsRequest(
                app_id=self._app_id,
                timeout=60,
                last_entry_id=last_log_batch_entry_id,
            )
            log_batch: api_pb2.TaskLogsBatch
            line_buffers: Dict[int, LineBufferedOutput] = {}
            async for log_batch in self.client.stub.AppGetLogs(request):
                if log_batch.app_state:
                    logger.debug(f"App state now {api_pb2.AppState.Name(log_batch.app_state)}")
                    if log_batch.app_state not in (
                        api_pb2.APP_STATE_EPHEMERAL,
                        api_pb2.APP_STATE_DRAINING_LOGS,
                    ):
                        last_log_batch_entry_id = None
                        return
                else:
                    if log_batch.entry_id != "":
                        # log_batch entry_id is empty for fd="server" messages from AppGetLogs
                        last_log_batch_entry_id = log_batch.entry_id

                    for log in log_batch.items:
                        if log.task_state:
                            message = self._update_task_state(log_batch.task_id, log.task_state)
                            live_task_status.update(step_progress(message))
                        if log.data:
                            stream = line_buffers.get(log.file_descriptor)
                            if stream is None:
                                stream = LineBufferedOutput(
                                    functools.partial(output_mgr.print_log, log.file_descriptor)
                                )
                                line_buffers[log.file_descriptor] = stream
                            stream.write(log.data)
            for stream in line_buffers.values():
                stream.finalize()

        while True:
            try:
                await _get_logs()
            except asyncio.CancelledError:
                logger.info("Logging cancelled")
                raise
            except grpc.aio.AioRpcError as exc:
                if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    # try again if we had a temporary connection drop, for example if computer went to sleep
                    logger.info("Log fetching timed out - retrying")
                    continue
                raise

            if last_log_batch_entry_id is None:
                break
            # TODO: catch errors, sleep, and retry?
        logger.debug("Logging exited gracefully")

    async def _initialize_container(self, app_id, client, task_id):
        """Used by the container to bootstrap the app and all its objects."""
        self._app_id = app_id
        self.client = client

        req = api_pb2.AppGetObjectsRequest(app_id=app_id, task_id=task_id)
        resp = await self.client.stub.AppGetObjects(req)
        for (
            tag,
            object_id,
        ) in resp.object_ids.items():
            self._tag_to_object[tag] = Object.from_id(object_id, self)

        # In the container, run forever
        self.state = AppState.RUNNING

    async def create_object(self, obj: Object, existing_object_id: Optional[str] = None) -> str:
        """Takes an object as input, returns an object id.

        This is a noop for any object that's not a factory.
        """
        if synchronizer.is_synchronized(obj):
            raise Exception(f"{obj} is synchronized")

        if obj.object_id:
            # This object is already created, just return the id
            return obj.object_id

        # Already created
        # TODO: handle references to the current app
        # if obj.tag and obj.tag in self._tag_to_object:
        #    return self._tag_to_object[obj.tag].object_id

        creating_message = obj.get_creating_message()
        if creating_message is not None:
            step_node = self._progress.add(step_progress(creating_message))

        # Create object
        if obj.label is not None and obj.label.app_name is not None:
            # TODO: this is a bit of a special case that we should clean up later
            object_id = await self._include(obj.label.app_name, obj.label.object_label, obj.label.namespace)
        else:
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

    async def _flush_objects(self):
        """Create objects that have been defined but not created on the server."""
        for tag, obj in self._blueprint.get_objects():
            existing_object_id = self._tag_to_existing_id.get(tag)
            logger.debug(f"Creating object {obj} with existing id {existing_object_id}")
            object_id = await self.create_object(obj, existing_object_id)
            self._tag_to_object[tag] = Object.from_id(object_id, self)

    def __getitem__(self, tag):
        # TODO(erikbern): this should really be an app vs blueprint thing
        if self.state == AppState.RUNNING:
            # TODO: this is a terrible hack for now. For running apps inside the container,
            # because of the singleton thing, any unrelated app will also be RUNNING, so this
            # branch triggers. However we don't want this to cause a KeyError.
            # Let's revisit once we clean up the app singleton
            return self._tag_to_object.get(tag)
        else:
            return self._blueprint.get_object(tag)

    def __setitem__(self, tag, obj):
        self._register_object(tag, obj)

    @synchronizer.asynccontextmanager
    async def _run(self, client, output_mgr, existing_app_id, last_log_entry_id=None):
        # TOOD: use something smarter than checking for the .client to exists in order to prevent
        # race conditions here!
        if self.state != AppState.NONE:
            raise Exception(f"Can't start a app that's already in state {self.state}")
        self.state = AppState.STARTING
        self.client = client

        try:
            if existing_app_id is not None:
                # Get all the objects first
                obj_req = api_pb2.AppGetObjectsRequest(app_id=existing_app_id)
                obj_resp = await self.client.stub.AppGetObjects(obj_req)
                self._tag_to_existing_id = dict(obj_resp.object_ids)
                self._app_id = existing_app_id
            else:
                # Start app
                # TODO(erikbern): maybe this should happen outside of this method?
                app_req = api_pb2.AppCreateRequest(client_id=client.client_id, name=self.name)
                app_resp = await client.stub.AppCreate(app_req)
                self._tag_to_existing_id = {}
                self._app_id = app_resp.app_id

            # Start tracking logs and yield context
            async with TaskContext(grace=config["logs_timeout"]) as tc:
                with output_mgr.ctx_if_visible(output_mgr.make_live(step_progress("Initializing..."))):
                    live_task_status = output_mgr.make_live(step_progress("Running app..."))
                    tc.create_task(self._get_logs_loop(output_mgr, live_task_status, last_log_entry_id or ""))
                output_mgr.print_if_visible(step_completed("Intialized."))

                try:
                    progress = Tree(step_progress("Creating objects..."), guide_style="gray50")
                    self._progress = progress
                    with output_mgr.ctx_if_visible(output_mgr.make_live(progress)):
                        await self._flush_objects()
                    progress.label = step_completed("Created objects.")
                    output_mgr.print_if_visible(progress)

                    # Create all members
                    with output_mgr.ctx_if_visible(live_task_status):
                        # Create the app (and send a list of all tagged obs)
                        # TODO(erikbern): we should delete objects from a previous version that are no longer needed
                        # We just delete them from the app, but the actual objects will stay around
                        object_ids = {tag: obj.object_id for tag, obj in self._tag_to_object.items()}
                        req_set = api_pb2.AppSetObjectsRequest(
                            app_id=self._app_id,
                            object_ids=object_ids,
                        )
                        await self.client.stub.AppSetObjects(req_set)

                        self.state = AppState.RUNNING
                        yield self  # yield context manager to block
                        self.state = AppState.STOPPING

                finally:
                    # Stop app server-side. This causes:
                    # 1. Server to kill any running task
                    # 2. Logs to drain (stopping the _get_logs_loop coroutine)
                    logger.debug("Stopping the app server-side")
                    req_disconnect = api_pb2.AppClientDisconnectRequest(app_id=self._app_id)
                    await self.client.stub.AppClientDisconnect(req_disconnect)

            output_mgr.print_if_visible(step_completed("App completed."))

        finally:
            self.client = None
            self.state = AppState.NONE
            self._progress = None
            self._tag_to_object = {}

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
            async with self._run(client, output_mgr, None) as it:
                yield it  # ctx mgr

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

    async def detach(self):
        request = api_pb2.AppDetachRequest(app_id=self._app_id)
        await self.client.stub.AppDetach(request)

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
        if self.state != AppState.NONE:
            raise InvalidError("Can only deploy an app that isn't running")
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
            async with self._run(client, output_mgr, existing_app_id, last_log_entry_id):
                # TODO: this could be simplified in case it's the same app id as previously
                deploy_req = api_pb2.AppDeployRequest(
                    app_id=self._app_id,
                    name=name,
                    namespace=namespace,
                )
                await client.stub.AppDeploy(deploy_req)
        return self._app_id

    async def _include(self, name, object_label, namespace):
        """Internal method to resolve to an object id."""
        request = api_pb2.AppIncludeObjectRequest(
            app_id=self._app_id,
            name=name,
            object_label=object_label,
            namespace=namespace,
        )
        response = await self.client.stub.AppIncludeObject(request)
        if not response.object_id:
            obj_repr = name
            if object_label is not None:
                obj_repr += f".{object_label}"
            if namespace != api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT:
                obj_repr += f" (namespace {api_pb2.DeploymentNamespace.Name(namespace)})"
            # TODO: disambiguate between app not found and object not found?
            err_msg = f"Could not find object {obj_repr}"
            raise NotFoundError(err_msg, obj_repr)
        return response.object_id

    async def include(self, name, object_label=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
        """Looks up an object and return a newly constructed one."""
        object_id = await self._include(name, object_label, namespace)
        return Object.from_id(object_id, self)

    def _serialize(self, obj):
        """Serializes object and replaces all references to the client class by a placeholder."""
        buf = io.BytesIO()
        Pickler(self, buf).dump(obj)
        return buf.getvalue()

    def _deserialize(self, s: bytes):
        """Deserializes object and replaces all client placeholders by self."""
        return Unpickler(self, io.BytesIO(s)).load()

    def _register_function(self, function):
        self._register_object(function.tag, function)
        function_proxy = _FunctionProxy(function, self, function.tag)
        return function_proxy

    def _get_default_image(self):
        # TODO(erikbern): instead of writing this to the same namespace
        # as the user's objects, we could use sub-blueprints in the future
        try:
            return self._blueprint.get_object("_image")
        except KeyError:
            image = _DebianSlim()
            self._register_object("_image", image)
            return image

    @decorator_with_options
    def function(
        self,
        raw_f=None,  # The decorated function
        *,
        image: _Image = None,  # The image to run as the container for the function
        schedule: Optional[Schedule] = None,  # An optional Modal Schedule for the function
        secret: Optional[Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        rate_limit: Optional[RateLimit] = None,  # Optional RateLimit for the function
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Collection[_Mount] = (),
    ) -> _Function:  # Function object - callable as a regular function within a Modal app
        """Decorator to create Modal functions"""
        if image is None:
            image = self._get_default_image()
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
        image: _Image = None,  # The image to run as the container for the function
        secret: Optional[Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        rate_limit: Optional[RateLimit] = None,  # Optional RateLimit for the function
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Collection[_Mount] = (),
    ) -> _Function:
        """Decorator to create Modal generators"""
        if image is None:
            image = self._get_default_image()
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
        image: _Image = None,  # The image to run as the container for the function
        secret: Optional[Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        mounts: Collection[_Mount] = (),
    ):
        if image is None:
            image = self._get_default_image()

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
        image: _Image = None,  # The image to run as the container for the function
        secret: Optional[Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        mounts: Collection[_Mount] = (),
    ):
        if image is None:
            image = self._get_default_image()

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


App, AioApp = synchronize_apis(_App)


def is_local() -> bool:
    """Returns whether we're running in the cloud or not."""
    return not get_container_app()
