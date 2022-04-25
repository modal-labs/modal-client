import asyncio
import io
import os
import sys
from typing import Collection, Dict, Optional, Union

import grpc

from modal._progress import safe_progress
from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext, synchronize_apis, synchronizer
from modal_utils.decorator_utils import decorator_with_options

from ._app_singleton import get_container_app, set_container_app
from ._app_state import AppState
from ._blueprint import Blueprint
from ._factory import _local_construction
from ._logging import LogPrinter
from ._serialization import Pickler, Unpickler
from .client import _Client
from .config import config, logger
from .exception import InvalidError, NotFoundError
from .functions import _Function
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

    @app.function(secret=modal.Secret.include(app, "some_secret"), schedule=modal.Period(days=1))
    def foo():
        ...
    ```
    In this example, both `foo`, the secret and the schedule are registered with the app.
    """

    _created_tagged_objects: Dict[str, str]  # tag -> id

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
        if hasattr(self, "_initialized"):
            return  # Prevent re-initialization with the singleton

        self._initialized = True
        self._app_id = None
        self.client = None
        self.name = name or self._infer_app_name()
        self.state = AppState.NONE
        self._created_tagged_objects = {}  # tag -> object id
        self._blueprint = Blueprint()
        self._task_states = {}
        self._progress = None
        self._log_printer = LogPrinter()
        super().__init__()

    @property
    def app_id(self):
        return self._app_id

    def _infer_app_name(self):
        script_filename = os.path.split(sys.argv[0])[-1]
        args = [script_filename] + sys.argv[1:]
        return " ".join(args)

    def _get_object_id_by_tag(self, tag: str):
        """Assigns an id to the object if there is already one set.

        This happens inside a container in the global scope.
        """
        return self._created_tagged_objects.get(tag)

    def _register_object(self, obj):
        """Registers an object to be created by the app so that it's available in modal.

        This is only used by factories and functions."""
        if self.state != AppState.NONE:
            raise Exception(f"Can only register objects on a app that's not running (state = {self.state}")
        if obj.tag in self._created_tagged_objects:
            # in case of a double load of an object, which seems
            # to happen sometimes when cloudpickle loads an object whose
            # type is declared in a module with modal functions
            pass
        self._blueprint.register(obj)

    def _update_task_state(self, task_id, state):
        self._task_states[task_id] = state

        # Recompute task status string.

        all_states = self._task_states.values()
        states_set = set(all_states)

        def tasks_at_state(state):
            return sum(x == state for x in all_states)

        # The most advanced state that's present informs the message.
        if api_pb2.TASK_STATE_RUNNING in states_set:
            tasks_running = tasks_at_state(api_pb2.TASK_STATE_RUNNING)
            tasks_loading = tasks_at_state(api_pb2.TASK_STATE_LOADING_IMAGE)
            msg = f"Running ({tasks_running}/{tasks_running + tasks_loading} containers in use)..."
        elif api_pb2.TASK_STATE_LOADING_IMAGE in states_set:
            tasks_loading = tasks_at_state(api_pb2.TASK_STATE_LOADING_IMAGE)
            msg = f"Loading images ({tasks_loading} containers initializing)..."
        elif api_pb2.TASK_STATE_WORKER_ASSIGNED in states_set:
            msg = "Worker assigned..."
        elif api_pb2.TASK_STATE_QUEUED in states_set:
            msg = "Tasks queued..."
        else:
            msg = "Tasks created..."
        if not self._progress.is_stopped():
            self._progress.substep(msg)

    async def _get_logs_loop(self, stdout, stderr):
        last_log_batch_entry_id = ""

        async def _get_logs(stdout, stderr):
            nonlocal last_log_batch_entry_id

            request = api_pb2.AppGetLogsRequest(
                app_id=self._app_id,
                timeout=60,
                last_entry_id=last_log_batch_entry_id,
            )
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
                            self._update_task_state(log_batch.task_id, log.task_state)
                        if log.data:
                            self._log_printer.feed(log, stdout, stderr)

        while True:
            try:
                await _get_logs(stdout, stderr)
            except asyncio.CancelledError:
                logger.info("Logging cancelled")
                raise
            except grpc.aio._call.AioRpcError as exc:
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
        self._created_tagged_objects = dict(resp.object_ids)

        # In the container, run forever
        self.state = AppState.RUNNING

    async def create_object(self, obj: Object) -> str:
        """Takes an object as input, returns an object id.

        This is a noop for any object that's not a factory.
        """
        if synchronizer.is_synchronized(obj):
            raise Exception(f"{obj} is synchronized")

        if obj.object_id:
            # This object is already created, just return the id
            return obj.object_id

        # Already created
        if obj.tag and obj.tag in self._created_tagged_objects:
            return self._created_tagged_objects[obj.tag]

        progress_messages = obj.get_progress_messages()
        if progress_messages is not None:
            step_no = self._progress.substep(progress_messages[0], False, progress_messages[1])

        # Create object
        if obj.label is not None and obj.label.app_name is not None:
            # TODO: this is a bit of a special case that we should clean up later
            object_id = await self._include(obj.label.app_name, obj.label.object_label, obj.label.namespace)
        else:
            object_id = await obj.load(self)
        if object_id is None:
            raise Exception(f"object_id for object of type {type(obj)} is None")

        obj.set_object_id(object_id, self)
        if obj.tag:
            self._created_tagged_objects[obj.tag] = object_id

        if progress_messages is not None:
            self._progress.complete_substep(step_no)

        return object_id

    async def _flush_objects(self):
        "Create objects that have been defined but not created on the server."

        for obj in self._blueprint.get_objects():
            if obj.object_id is not None:
                # object is already created (happens due to object re-initialization in the container).
                # TODO: we should check that the object id isn't old
                continue

            logger.debug(f"Creating object {obj}")
            await self.create_object(obj)

    @synchronizer.asynccontextmanager
    async def _run(self, client, stdout, stderr, logs_timeout, show_progress=None):
        # TOOD: use something smarter than checking for the .client to exists in order to prevent
        # race conditions here!
        if self.state != AppState.NONE:
            raise Exception(f"Can't start a app that's already in state {self.state}")
        self.state = AppState.STARTING
        self.client = client

        if show_progress is None:
            visible_progress = (stdout or sys.stdout).isatty()
        else:
            visible_progress = show_progress

        try:
            # Start app
            req = api_pb2.AppCreateRequest(client_id=client.client_id, name=self.name)
            resp = await client.stub.AppCreate(req)
            self._app_id = resp.app_id

            # Start tracking logs and yield context
            async with TaskContext(grace=config["logs_timeout"]) as tc:
                async with safe_progress(tc, stdout, stderr, visible_progress) as progress_handler:
                    self._progress = progress_handler
                    self._progress.step("Initializing...", "Initialized.")

                    tc.create_task(self._get_logs_loop(stdout, stderr))

                    try:
                        self._progress.step("Creating objects...", "Created objects.")
                        # Create all members
                        await self._flush_objects()
                        self._progress.step("Running app...", "App completed.")

                        # Create the app (and send a list of all tagged obs)
                        req_set = api_pb2.AppSetObjectsRequest(
                            app_id=self._app_id,
                            object_ids=self._created_tagged_objects,
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
        finally:
            self.client = None
            self.state = AppState.NONE
            self._progress = None
            self._created_tagged_objects = {}

    @synchronizer.asynccontextmanager
    async def _get_client(self, client=None):
        if client is None:
            async with _Client.from_env() as client:
                yield client
        else:
            yield client

    @synchronizer.asynccontextmanager
    async def run(self, client=None, stdout=None, stderr=None, logs_timeout=None, show_progress=None):
        async with self._get_client(client) as client:
            async with self._run(client, stdout, stderr, logs_timeout, show_progress) as it:
                yield it  # ctx mgr

    async def detach(self):
        request = api_pb2.AppDetachRequest(app_id=self._app_id)
        await self.client.stub.AppDetach(request)

    async def deploy(
        self,
        name: str = None,  # Unique name of the deployment. Subsequent deploys with the same name overwrites previous ones. Falls back to the app name
        obj_or_objs: Union[
            Object, Dict[str, Object]
        ] = None,  # A single Modal *Object* or a `dict[str, Object]` of labels -> Objects to be exported for use by other apps
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT,
    ):
        """Deploys and exports objects in the app

        Usage:
        ```python
        if __name__ == "__main__":
            with app.run():
                app.deploy()
        ```

        Deployment has two primary purposes:
        * Persists all of the objects (Functions, Images, Schedules etc.) in the app, allowing them to live past the current app run
          Notably for Schedules this enables headless "cron"-like functionality where scheduled functions continue to be invoked after
          the client has closed.
        * Allows for certain of these objects, *deployment objects*, to be referred to and used by other apps
        """
        if self.client is None:
            raise InvalidError(
                "The app needs to be running to be deployed.\n\n"
                "Example usage:\n"
                "with app.run():\n"
                '    app.deploy("my_deployment")\n'
            )

        if name is None:
            name = self.name
        if name is None:
            raise InvalidError(
                "You need to either supply an explicit deployment name to the deploy command, or have a name set on the app.\n"
                "\n"
                "Examples:\n"
                'app.deploy("some_name")\n\n'
                "or\n"
                'app = App("some name")'
            )
        object_id = None
        object_ids = None  # name -> object_id
        if isinstance(obj_or_objs, Object):
            object_id = obj_or_objs.object_id
        elif isinstance(obj_or_objs, dict):
            object_ids = {label: obj.object_id for label, obj in obj_or_objs.items()}
        elif obj_or_objs is None:
            pass
        else:
            raise InvalidError(f"{obj_or_objs} not an Object or dict or None")
        request = api_pb2.AppDeployRequest(
            app_id=self._app_id,
            name=name,
            namespace=namespace,
            object_id=object_id,
            object_ids=object_ids,
        )
        await self.client.stub.AppDeploy(request)

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
        return Object._init_persisted(object_id, self)

    def _serialize(self, obj):
        """Serializes object and replaces all references to the client class by a placeholder."""
        buf = io.BytesIO()
        Pickler(self, buf).dump(obj)
        return buf.getvalue()

    def _deserialize(self, s: bytes):
        """Deserializes object and replaces all client placeholders by self."""
        return Unpickler(self, io.BytesIO(s)).load()

    @decorator_with_options
    def function(
        self,
        raw_f=None,  # The decorated function
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
            image = _DebianSlim(app=self)
        function = _Function(
            self,
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
        return function

    @decorator_with_options
    def generator(
        self,
        raw_f=None,  # The decorated function
        image: _Image = None,  # The image to run as the container for the function
        secret: Optional[Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: bool = False,  # Whether a GPU is required
        rate_limit: Optional[RateLimit] = None,  # Optional RateLimit for the function
        serialized: bool = False,  # Whether to send the function over using cloudpickle.
        mounts: Collection[_Mount] = (),
    ) -> _Function:
        if image is None:
            image = _DebianSlim(app=self)
        """Decorator to create Modal generators"""
        function = _Function(
            self,
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
        return function

    def local_construction(self, cls):
        """Decorator to create a custom initialization function for something that runs on app startup.

        This is useful if you need to define some object based on data on your development machine
        and access it later from Modal functions.

        The annotated function is called on app startup and persisted after that for the lifetime of
        the app.

        Example:
        ```python
        @app.local_construction(modal.Secret)
        def forward_local_secrets():
            return modal.Secret(app, os.environ)

        @app.function(secrets=forward_local_secrets)
        def editor():
            return os.environ["EDITOR"]
        ```
        """
        return _local_construction(self, cls)


App, AioApp = synchronize_apis(_App)
