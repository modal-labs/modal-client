import asyncio
import io
import os
import sys

import colorama

from modal._progress import safe_progress

from ._app_singleton import (
    get_container_app,
    get_default_app,
    set_container_app,
    set_running_app,
)
from ._app_state import AppState
from ._async_utils import TaskContext, run_coro_blocking, synchronizer
from ._client import Client
from ._grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIME_BUFFER
from ._object_meta import ObjectMeta
from ._serialization import Pickler, Unpickler
from .config import config, logger
from .exception import ExecutionError, InvalidError, NotFoundError
from .object import Object
from .proto import api_pb2


@synchronizer
class App:
    """The App manages objects in a few ways

    1. Every object belongs to an app
    2. Apps are responsible for syncing object identities across processes
    3. Apps manage all log collection for ephemeral functions
    """

    @classmethod
    def initialize_container_app(cls):
        set_container_app(super().__new__(cls))

    def __new__(cls, *args, **kwargs):
        singleton = get_container_app()
        if singleton is not None:
            # If there's a singleton app, just return it for everything
            return singleton
        else:
            # Refer to the normal constructor
            app = super().__new__(cls)
            return app

    def __init__(self, show_progress=None, blocking_late_creation_ok=False, name=None):
        if hasattr(self, "_initialized"):
            return  # Prevent re-initialization with the singleton

        self._initialized = True
        self.client = None
        self.name = name or self._infer_app_name()
        self.state = AppState.NONE
        self._pending_create_objects = []  # list of objects that haven't been created
        self._created_tagged_objects = {}  # tag -> object id
        self._show_progress = show_progress  # None = use sys.stdout.isatty()
        self._task_states = {}
        self._progress = None

        # TODO: this is a very hacky thing for notebooks. The problem is that
        # (a) notebooks require creating functions "late"
        # (b) notebooks run with an event loop, which makes synchronizer confused
        # We will have to rethink this soon.
        self._blocking_late_creation_ok = blocking_late_creation_ok
        super().__init__()

    def _infer_app_name(self):
        script_filename = os.path.split(sys.argv[0])[-1]
        args = [script_filename] + sys.argv[1:]
        return " ".join(args)

    def get_object_id_by_tag(self, tag):
        """Assigns an id to the object if there is already one set.

        This happens inside a container in the global scope."""
        return self._created_tagged_objects.get(tag)

    def register_object(self, obj):
        """Registers an object to be created by the app so that it's available in modal.

        This is only used by factories and functions."""
        if obj.tag is None:
            raise Exception("Can only register named objects")
        if self.state == AppState.NONE:
            self._pending_create_objects.append(obj)
        elif self._blocking_late_creation_ok:
            # See comment in constructor. This is a hacky hack to get notebooks working.
            # Let's revisit this shortly
            run_coro_blocking(self.create_object(obj))
        else:
            raise Exception("Can only register objects on a app that's not running")

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
        self._progress.set_substep_text(msg)

    async def _get_logs(self, stdout, stderr, last_log_batch_entry_id, timeout=BLOCKING_REQUEST_TIMEOUT):
        request = api_pb2.AppGetLogsRequest(
            app_id=self.app_id,
            timeout=timeout,
            last_entry_id=last_log_batch_entry_id,
        )
        add_newline = None
        async for log_batch in self.client.stub.AppGetLogs(request, timeout=timeout + GRPC_REQUEST_TIME_BUFFER):
            if log_batch.app_state:
                logger.info(f"App state now {api_pb2.AppState.Name(log_batch.app_state)}")
                if log_batch.app_state not in (
                    api_pb2.APP_STATE_EPHEMERAL,
                    api_pb2.APP_STATE_DRAINING_LOGS,
                ):
                    return None
            else:
                if log_batch.entry_id != "":
                    # log_batch entry_id is empty for fd="server" messages from AppGetLogs
                    last_log_batch_entry_id = log_batch.entry_id
                for log in log_batch.state_updates:
                    self._update_task_state(log_batch.task_id, log.task_state)

                if log_batch.items:
                    # HACK: to make partial line outputs (like when using a progress bar that uses
                    # ANSI escape chars) work. If the last log line doesn't end with a newline,
                    # add one manually, and take it back the next time we print something.
                    # TODO: this can cause problems if there are partial lines being printed as logs, and the user is also
                    # printing to stdout directly. Can be solved if we can print directly here and rely on the redirection
                    # (without calling suspend()), and then add the newline logic to `write_callback`.
                    last_item = log_batch.items[-1]
                    if add_newline:
                        print_logs("\033[A\r", "stdout", stdout, stderr)
                    add_newline = not last_item.data.endswith("\n")

                    with self._progress.suspend():
                        for log in log_batch.items:
                            assert not log.task_state
                            print_logs(log.data, log.fd, stdout, stderr)
                        if add_newline:
                            print_logs("\n", "stdout", stdout, stderr)
        return last_log_batch_entry_id

    async def _get_logs_loop(self, stdout, stderr):
        last_log_batch_entry_id = ""
        while True:
            try:
                last_log_batch_entry_id = await self._get_logs(stdout, stderr, last_log_batch_entry_id)
            except asyncio.CancelledError:
                logger.info("Logging cancelled")
                raise
            if last_log_batch_entry_id is None:
                break
            # TODO: catch errors, sleep, and retry?
        logger.info("Logging exited gracefully")

    async def initialize_container(self, app_id, client, task_id):
        """Used by the container to bootstrap the app and all its objects."""
        self.app_id = app_id
        self.client = client

        req = api_pb2.AppGetObjectsRequest(app_id=app_id, task_id=task_id)
        resp = await self.client.stub.AppGetObjects(req)
        self._created_tagged_objects = dict(resp.object_ids)

        # In the container, run forever
        self.state = AppState.RUNNING
        set_running_app(self)

    async def create_object(self, obj):
        """Takes an object as input, returns an object id.

        This is a noop for any object that's not a factory.
        """
        if not obj.is_factory():
            # This object is already created, just return the id
            return obj.object_id

        assert obj.tag
        self._progress.set_substep_text(f"Creating {obj.tag}...")

        # Already created
        if obj.tag in self._created_tagged_objects:
            return self._created_tagged_objects[obj.tag]

        # Create object
        object_id = await obj.load(self)
        if object_id is None:
            raise Exception(f"object_id for object of type {type(obj)} is None")

        obj.set_object_id(object_id, self)
        self._created_tagged_objects[obj.tag] = object_id
        return object_id

    async def _flush_objects(self):
        "Create objects that have been defined but not created on the server."

        while len(self._pending_create_objects) > 0:
            obj = self._pending_create_objects.pop()

            if obj.object_id is not None:
                # object is already created (happens due to object re-initialization in the container).
                # TODO: we should check that the object id isn't old
                continue

            logger.debug(f"Creating object {obj}")
            await self.create_object(obj)

    @synchronizer.asynccontextmanager
    async def _run(self, client, stdout, stderr, logs_timeout):
        # TOOD: use something smarter than checking for the .client to exists in order to prevent
        # race conditions here!
        if self.state != AppState.NONE:
            raise Exception(f"Can't start a app that's already in state {self.state}")
        self.state = AppState.STARTING
        self.client = client

        # We need to re-initialize all these objects. Needed if a app is reused.
        initial_objects = list(self._pending_create_objects)
        if self._show_progress is None:
            visible_progress = (stdout or sys.stdout).isatty()
        else:
            visible_progress = self._show_progress

        try:
            # Start app
            req = api_pb2.AppCreateRequest(client_id=client.client_id, name=self.name)
            resp = await client.stub.AppCreate(req)
            self.app_id = resp.app_id

            # Start tracking logs and yield context
            async with TaskContext(grace=config["logs_timeout"]) as tc:
                async with safe_progress(tc, stdout, stderr, visible_progress) as (
                    progress_handler,
                    real_stdout,
                    real_stderr,
                ):
                    self._progress = progress_handler
                    self._progress.step("Initializing...", "Initialized.")

                    tc.create_task(self._get_logs_loop(real_stdout, real_stderr))

                    self._progress.step("Creating objects...", "Created objects.")
                    # Create all members
                    await self._flush_objects()
                    self._progress.step("Running app...", "App completed.")

                    # Create the app (and send a list of all tagged obs)
                    req = api_pb2.AppSetObjectsRequest(
                        app_id=self.app_id,
                        object_ids=self._created_tagged_objects,
                    )
                    await self.client.stub.AppSetObjects(req)

                    try:
                        self.state = AppState.RUNNING
                        yield self  # yield context manager to block
                        self.state = AppState.STOPPING
                    finally:
                        # Stop app server-side. This causes:
                        # 1. Server to kill any running task
                        # 2. Logs to drain (stopping the _get_logs_loop coroutine)
                        logger.debug("Stopping the app server-side")
                        req = api_pb2.AppClientDisconnectRequest(app_id=self.app_id)
                        await self.client.stub.AppClientDisconnect(req)
                    if real_stdout:
                        real_stdout.flush()
                    if real_stderr:
                        real_stderr.flush()
        finally:
            self.client = None
            self.state = AppState.NONE
            self._progress = None
            self._pending_create_objects = initial_objects
            self._created_tagged_objects = {}

    @synchronizer.asynccontextmanager
    async def _get_client(self, client=None):
        if client is None:
            async with Client.from_env() as client:
                yield client
        else:
            yield client

    @synchronizer.asynccontextmanager
    async def run(self, client=None, stdout=None, stderr=None, logs_timeout=None):
        set_running_app(self)
        try:
            async with self._get_client(client) as client:
                async with self._run(client, stdout, stderr, logs_timeout) as it:
                    yield it  # ctx mgr
        finally:
            set_running_app(None)

    async def detach(self):
        request = api_pb2.AppDetachRequest(app_id=self.app_id)
        await self.client.stub.AppDetach(request)

    async def deploy(self, name, obj_or_objs=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
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
            app_id=self.app_id,
            name=name,
            namespace=namespace,
            object_id=object_id,
            object_ids=object_ids,
        )
        await self.client.stub.AppDeploy(request)

    async def include(self, name, object_label=None, namespace=api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT):
        request = api_pb2.AppIncludeObjectRequest(
            app_id=self.app_id,
            name=name,
            object_label=object_label,
            namespace=namespace,
        )
        response = await self.client.stub.AppIncludeObject(request)
        if not response.object_id:
            err_msg = f"Could not find object {name}"
            if object_label is not None:
                err_msg += f".{object_label}"
            if namespace != api_pb2.DEPLOYMENT_NAMESPACE_ACCOUNT:
                err_msg += f" (namespace {api_pb2.DeploymentNamespace.Name(namespace)})"
            # TODO: disambiguate between app not found and object not found?
            raise NotFoundError(err_msg)
        return Object._init_persisted(response.object_id, self)

    def serialize(self, obj):
        """Serializes object and replaces all references to the client class by a placeholder."""
        buf = io.BytesIO()
        Pickler(self, buf).dump(obj)
        return buf.getvalue()

    def deserialize(self, s: bytes):
        """Deserializes object and replaces all client placeholders by self."""
        return Unpickler(self, ObjectMeta.prefix_to_type, io.BytesIO(s)).load()


def run(*args, **kwargs):
    """Start up the default modal app"""
    if get_container_app() is not None:
        # TODO: we could probably capture whether this happens during an import
        raise ExecutionError("Cannot run modal.run() inside a container! You might have global code that does this.")
    app = get_default_app()
    return app.run(*args, **kwargs)


def print_logs(output: str, fd: str, stdout=None, stderr=None):
    if fd == "stdout":
        buf = stdout or sys.stdout
        color = colorama.Fore.BLUE
    elif fd == "stderr":
        buf = stderr or sys.stderr
        color = colorama.Fore.RED
    elif fd == "server":
        buf = stderr or sys.stderr
        color = colorama.Fore.YELLOW
    else:
        raise Exception(f"weird fd {fd} for log output")

    if buf.isatty():
        buf.write(color)

    buf.write(output)

    if buf.isatty():
        buf.write(colorama.Style.RESET_ALL)
        buf.flush()
