import asyncio
import enum
import functools
import io
import sys
import warnings

from .async_utils import TaskContext, retry, synchronizer
from .client import Client
from .config import config, logger
from .function import Function
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIME_BUFFER, ChannelPool
from .image import DebianSlim  # TODO: ugly
from .object import Object, ObjectMeta
from .proto import api_pb2
from .serialization import Pickler, Unpickler
from .session_state import SessionState
from .utils import print_logs


@synchronizer
class Session:
    """The Session manages objects in a few ways

    1. Every object belongs to a session
    2. Sessions are responsible for syncing object identities across processes
    3. Sessions manage all log collection for ephemeral functions

    Sessions isn't a great name, a better name is probably "dependency set" or "application"
    or maybe "collection" or "bag"
    """

    _singleton = None  # When running inside a container, this is a singleton

    @classmethod
    def initialize_singleton(cls):
        cls._singleton = super().__new__(cls)

    @classmethod
    def reset_singleton(cls):
        # Just used in test code to reset
        cls._singleton = None

    def __new__(cls):
        if cls._singleton is not None:
            # If there's a singleton session, just return it for everything
            return cls._singleton
        else:
            # Refer to the normal constructor
            session = super().__new__(cls)
            return session

    def __init__(self):
        if hasattr(self, "_initialized"):
            return  # Prevent re-initialization with the singleton
        self._initialized = True
        self._flush_lock = None
        self.client = None
        self.state = SessionState.NONE
        self._pending_create_objects = []  # list of objects that haven't been created
        self._created_tagged_objects = {}  # tag -> object id
        super().__init__()

    def register(self, obj):
        """Registers an object to be created by the session.

        This is invoked by the constructor in Object."""
        if obj.tag and obj.tag in self._created_tagged_objects:
            # If this code runs inside the container, check if the object is already created
            # In that case, just set the id on it
            object_id = self._created_tagged_objects[obj.tag]
            obj.set_object_id(object_id, self.session_id)
        else:
            if self.state == SessionState.NONE and obj.tag is None:
                raise Exception(f"{obj}: Only objects with tags can be created prior to the session running")
            self._pending_create_objects.append(obj)

    def function(self, raw_f=None, image=None, env_dict=None, is_generator=False, gpu=False):
        if image is None:
            image = DebianSlim(session=self)

        def decorate(raw_f):
            return Function(self, raw_f, image=image, env_dict=env_dict, is_generator=is_generator, gpu=gpu)

        if raw_f is None:
            # called like @session.function(x=y)
            return decorate
        else:
            # called like @session.function
            return decorate(raw_f)

    def generator(self, *args, **kwargs):
        kwargs = dict(is_generator=True, **kwargs)
        return self.function(*args, **kwargs)

    async def _get_logs(self, stdout, stderr, draining=False, timeout=BLOCKING_REQUEST_TIMEOUT):
        request = api_pb2.SessionGetLogsRequest(session_id=self.session_id, timeout=timeout, draining=draining)
        n_running = None
        async for log_entry in self.client.stub.SessionGetLogs(request, timeout=timeout + GRPC_REQUEST_TIME_BUFFER):
            if log_entry.done:
                logger.info("No more logs")
                return
            elif log_entry.n_running:
                n_running = log_entry.n_running
            else:
                print_logs(log_entry.data, log_entry.fd, stdout, stderr)
        if draining:
            raise Exception(
                f"Failed waiting for all logs to finish. There are still {n_running} tasks the server will kill."
            )

    async def initialize_container(self, session_id, client):
        """Used by the container to bootstrap the session and all its objects."""
        self.session_id = session_id
        self.client = client

        if self._flush_lock is None:
            self._flush_lock = asyncio.Lock()

        req = api_pb2.SessionGetObjectsRequest(session_id=session_id)
        resp = await self.client.stub.SessionGetObjects(req)
        self._created_tagged_objects = dict(resp.object_ids)

        # In the container, run forever
        self.state = SessionState.RUNNING

    async def create_object(self, obj):
        if obj.tag is not None and obj.tag in self._created_tagged_objects:
            return self._created_tagged_objects[obj.tag]
        if obj.object_id is None:
            if obj.share_path:
                # This is a reference to a persistent object
                object_id = await self._use_object(obj.share_path)
            else:
                # This is something created locally
                object_id = await obj._create_impl()
            obj.set_object_id(object_id, self.session_id)
            if obj.tag:
                self._created_tagged_objects[obj.tag] = object_id
        return obj.object_id

    async def flush_objects(self):
        "Create objects that have been defined but not created on the server."

        async with self._flush_lock:
            while len(self._pending_create_objects) > 0:
                obj = self._pending_create_objects.pop()

                if obj.object_id is not None:
                    # object is already created (happens due to object re-initialization in the container).
                    # TODO: we should check that the object id isn't old
                    continue

                logger.debug(f"Creating object {obj}")
                await self.create_object(obj)

    async def share(self, obj, path):
        object_id = await self.create_object(obj)
        request = api_pb2.SessionShareObjectRequest(session_id=self.session_id, object_id=object_id, path=path)
        await self.client.stub.SessionShareObject(request)

    async def _use_object(self, path):
        request = api_pb2.SessionUseObjectRequest(session_id=self.session_id, path=path)
        response = await self.client.stub.SessionUseObject(request)
        return response.object_id

    @synchronizer.asynccontextmanager
    async def run(self, client=None, stdout=None, stderr=None):
        # HACK because Lock needs to be created on the synchronizer thread
        if self._flush_lock is None:
            self._flush_lock = asyncio.Lock()

        if client is None:
            client = await Client.from_env()
            async with client:
                async for it in self._run(client, stdout, stderr):
                    yield it  # ctx mgr
        else:
            async for it in self._run(client, stdout, stderr):
                yield it  # ctx mgr

    async def _run(self, client, stdout, stderr):
        # TOOD: use something smarter than checking for the .client to exists in order to prevent
        # race conditions here!
        if self.state != SessionState.NONE:
            raise Exception(f"Can't start a session that's already in state {self.state}")
        self.state = SessionState.STARTING
        self.client = client

        # We need to re-initialize all these objects. Needed if a session is reused.
        initial_objects = list(self._pending_create_objects)

        try:
            # Start session
            req = api_pb2.SessionCreateRequest(client_id=client.client_id)
            resp = await client.stub.SessionCreate(req)
            self.session_id = resp.session_id

            # Start tracking logs and yield context
            async with TaskContext(grace=1.0) as tc:
                get_logs_closure = functools.partial(self._get_logs, stdout, stderr)
                functools.update_wrapper(get_logs_closure, self._get_logs)  # Needed for debugging tasks
                tc.infinite_loop(get_logs_closure)

                # Create all members
                await self.flush_objects()

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
            req = api_pb2.SessionStopRequest(session_id=self.session_id)
            await self.client.stub.SessionStop(req)

            # Fetch any straggling logs
            logger.debug("Draining logs")
            await self._get_logs(stdout, stderr, draining=True, timeout=config["logs_timeout"])

        finally:
            self.client = None
            self.state = SessionState.NONE
            self._pending_create_objects = initial_objects
            self._created_tagged_objects = {}

    def serialize(self, obj):
        """Serializes object and replaces all references to the client class by a placeholder."""
        buf = io.BytesIO()
        Pickler(self, ObjectMeta.type_to_name, buf).dump(obj)
        return buf.getvalue()

    def deserialize(self, s: bytes):
        """Deserializes object and replaces all client placeholders by self."""
        return Unpickler(self, ObjectMeta.name_to_type, io.BytesIO(s)).load()
