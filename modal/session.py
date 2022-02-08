import asyncio
import io
import os
import sys

import colorama

from modal._progress import safe_progress

from ._async_utils import TaskContext, run_coro_blocking, synchronizer
from ._client import Client
from ._grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIME_BUFFER
from ._object_meta import ObjectMeta
from ._serialization import Pickler, Unpickler
from ._session_singleton import (
    get_container_session,
    get_default_session,
    set_container_session,
    set_running_session,
)
from ._session_state import SessionState
from .config import logger
from .exception import ExecutionError, NotFoundError
from .object import Object
from .proto import api_pb2


@synchronizer
class Session:
    """The Session manages objects in a few ways

    1. Every object belongs to a session
    2. Sessions are responsible for syncing object identities across processes
    3. Sessions manage all log collection for ephemeral functions

    "session" isn't a great name, a better name is probably "scope".
    """

    @classmethod
    def initialize_container_session(cls):
        set_container_session(super().__new__(cls))

    def __new__(cls, *args, **kwargs):
        singleton = get_container_session()
        if singleton is not None:
            # If there's a singleton session, just return it for everything
            return singleton
        else:
            # Refer to the normal constructor
            session = super().__new__(cls)
            return session

    def __init__(self, show_progress=None, blocking_late_creation_ok=False, name=None):
        if hasattr(self, "_initialized"):
            return  # Prevent re-initialization with the singleton

        self._initialized = True
        self.client = None
        self.name = name or self._infer_session_name()
        self.state = SessionState.NONE
        self._pending_create_objects = []  # list of objects that haven't been created
        self._created_tagged_objects = {}  # tag -> object id
        self._show_progress = show_progress  # None = use sys.stdout.isatty()
        self._last_log_batch_entry_id = ""
        self._task_states = {}
        self._progress = None

        # TODO: this is a very hacky thing for notebooks. The problem is that
        # (a) notebooks require creating functions "late"
        # (b) notebooks run with an event loop, which makes synchronizer confused
        # We will have to rethink this soon.
        self._blocking_late_creation_ok = blocking_late_creation_ok
        super().__init__()

    def _infer_session_name(self):
        script_filename = os.path.split(sys.argv[0])[-1]
        args = [script_filename] + sys.argv[1:]
        return " ".join(args)

    def get_object_id_by_tag(self, tag):
        """Assigns an id to the object if there is already one set.

        This happens inside a container in the global scope."""
        return self._created_tagged_objects.get(tag)

    def register_object(self, obj):
        """Registers an object to be created by the session so that it's available in modal.

        This is only used by factories and functions."""
        if obj.tag is None:
            raise Exception("Can only register named objects")
        if self.state == SessionState.NONE:
            self._pending_create_objects.append(obj)
        elif self._blocking_late_creation_ok:
            # See comment in constructor. This is a hacky hack to get notebooks working.
            # Let's revisit this shortly
            run_coro_blocking(self.create_object(obj))
        else:
            raise Exception("Can only register objects on a session that's not running")

    def _update_task_state(self, task_id, state):
        self._task_states[task_id] = state

        # Recompute task status string.

        all_states = self._task_states.values()
        max_state = max(all_states)

        def tasks_at_state(state):
            return sum(x == state for x in all_states)

        if max_state == api_pb2.TaskState.TS_CREATED:
            msg = f"Tasks created..."
        elif max_state == api_pb2.TaskState.TS_QUEUED:
            msg = f"Tasks queued..."
        elif max_state == api_pb2.TaskState.TS_WORKER_ASSIGNED:
            msg = f"Worker assigned..."
        elif max_state == api_pb2.TaskState.TS_LOADING_IMAGE:
            tasks_loading = tasks_at_state(api_pb2.TaskState.TS_LOADING_IMAGE)
            msg = f"Loading images ({tasks_loading} containers initializing)..."
        else:
            tasks_running = tasks_at_state(api_pb2.TaskState.TS_RUNNING)
            tasks_loading = tasks_at_state(api_pb2.TaskState.TS_LOADING_IMAGE)
            msg = f"Running ({tasks_running}/{tasks_running + tasks_loading} containers in use)..."
        self._progress.set_substep_text(msg)

    async def _get_logs(self, stdout, stderr, timeout=BLOCKING_REQUEST_TIMEOUT):
        # control flow-wise, there should only be one _get_logs running for each session
        # i.e. maintain only one active SessionGetLogs grpc for each session
        request = api_pb2.SessionGetLogsRequest(
            session_id=self.session_id,
            timeout=timeout,
            last_entry_id=self._last_log_batch_entry_id,
        )
        n_running = None
        add_newline = None
        async for log_batch in self.client.stub.SessionGetLogs(request, timeout=timeout + GRPC_REQUEST_TIME_BUFFER):
            # if done or n_running, the batch was generated by the server
            if log_batch.session_state:
                logger.info(f"Session state now {api_pb2.SessionState.Name(log_batch.session_state)}")
                if api_pb2.SessionState == api_pb2.SessionState.SS_STOPPED:
                    return
            elif log_batch.n_running:
                n_running = log_batch.n_running
            else:
                if log_batch.entry_id != "":
                    # log_batch entry_id is empty for fd="server" messages from SessionGetLogs
                    self._last_log_batch_entry_id = log_batch.entry_id
                for log in log_batch.state_updates:
                    self._update_task_state(log.task_id, log.task_state)

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
                    add_newline = not last_item.data.endswith(b"\n")

                    with self._progress.suspend():
                        for log in log_batch.items:
                            assert not log.task_state
                            print_logs(log.data.decode("utf8"), log.fd, stdout, stderr)
                        if add_newline:
                            print_logs("\n", "stdout", stdout, stderr)

    async def initialize_container(self, session_id, client, task_id):
        """Used by the container to bootstrap the session and all its objects."""
        self.session_id = session_id
        self.client = client

        req = api_pb2.SessionGetObjectsRequest(session_id=session_id, task_id=task_id)
        resp = await self.client.stub.SessionGetObjects(req)
        self._created_tagged_objects = dict(resp.object_ids)

        # In the container, run forever
        self.state = SessionState.RUNNING
        set_running_session(self)

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
        if self.state != SessionState.NONE:
            raise Exception(f"Can't start a session that's already in state {self.state}")
        self.state = SessionState.STARTING
        self.client = client

        # We need to re-initialize all these objects. Needed if a session is reused.
        initial_objects = list(self._pending_create_objects)
        if self._show_progress is None:
            visible_progress = (stdout or sys.stdout).isatty()
        else:
            visible_progress = self._show_progress

        try:
            # Start session
            req = api_pb2.SessionCreateRequest(client_id=client.client_id, name=self.name)
            resp = await client.stub.SessionCreate(req)
            self.session_id = resp.session_id

            # Start tracking logs and yield context
            async with TaskContext(grace=1.0) as tc:
                async with safe_progress(tc, stdout, stderr, visible_progress) as (progress_handler, stdout, stderr):
                    self._progress = progress_handler
                    self._progress.step("Initializing...", "Initialized.")

                    async def get_logs():
                        try:
                            await self._get_logs(stdout, stderr)
                        except asyncio.CancelledError:
                            logger.info("Logging cancelled")
                            raise

                    logs_task = tc.infinite_loop(get_logs, sleep=0)

                    self._progress.step("Creating objects...", "Created objects.")
                    # Create all members
                    await self._flush_objects()
                    self._progress.step("Running session...", "Session completed.")

                    # Create the session (and send a list of all tagged obs)
                    req = api_pb2.SessionSetObjectsRequest(
                        session_id=self.session_id,
                        object_ids=self._created_tagged_objects,
                    )
                    await self.client.stub.SessionSetObjects(req)

                    self.state = SessionState.RUNNING
                    yield self  # yield context manager to block
                    self.state = SessionState.STOPPING

                    # Stop session (this causes the server to kill any running task)
                    logger.debug("Stopping the session server-side")
                    req = api_pb2.SessionClientDisconnectRequest(session_id=self.session_id)
                    await self.client.stub.SessionClientDisconnect(req)
        finally:
            if self.state == SessionState.RUNNING:
                logger.warn("Stopping running session...")
                req = api_pb2.SessionClientDisconnectRequest(session_id=self.session_id)
                await self.client.stub.SessionClientDisconnect(req)
            self.client = None
            self.state = SessionState.NONE
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
        set_running_session(self)
        try:
            async with self._get_client(client) as client:
                async with self._run(client, stdout, stderr, logs_timeout) as it:
                    yield it  # ctx mgr
        finally:
            set_running_session(None)

    async def detach(self):
        request = api_pb2.SessionDetachRequest(session_id=self.session_id)
        await self.client.stub.SessionDetach(request)

    async def deploy(self, name, obj_or_objs=None, namespace=api_pb2.ShareNamespace.SN_ACCOUNT):
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
        request = api_pb2.SessionDeployRequest(
            session_id=self.session_id,
            name=name,
            namespace=namespace,
            object_id=object_id,
            object_ids=object_ids,
        )
        await self.client.stub.SessionDeploy(request)

    async def include(self, name, object_label=None, namespace=api_pb2.ShareNamespace.SN_ACCOUNT):
        request = api_pb2.SessionIncludeObjectRequest(
            session_id=self.session_id,
            name=name,
            object_label=object_label,
            namespace=namespace,
        )
        response = await self.client.stub.SessionIncludeObject(request)
        if not response.object_id:
            err_msg = f"Could not find object {name}"
            if object_label is not None:
                err_msg += f".{object_label}"
            if namespace != api_pb2.ShareNamespace.SN_ACCOUNT:
                err_msg += f" (namespace {api_pb2.ShareNamespace.Name(namespace)})"
            # TODO: disambiguate between session not found and object not found?
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
    """Start up the default modal session"""
    if get_container_session() is not None:
        # TODO: we could probably capture whether this happens during an import
        raise ExecutionError("Cannot run modal.run() inside a container!" " You might have global code that does this.")
    session = get_default_session()
    return session.run(*args, **kwargs)


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
