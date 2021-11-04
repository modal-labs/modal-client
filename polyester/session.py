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
from .image import base_image
from .object import Object, ObjectMeta
from .serialization import Pickler, Unpickler
from .proto import api_pb2
from .session_state import SessionState
from .utils import print_logs


@synchronizer
class Session:  # (Object):
    def __init__(self):
        self._objects = {}  # tag -> object
        self._object_ids = None  # tag -> object id
        self.client = None
        self.state = SessionState.NONE
        super().__init__()

    def register(self, obj):
        if obj.tag in self._objects and self._objects[obj.tag] != obj:
            # TODO: this situation currently happens when two objects are different references to
            # what's basically the same object, eg. both are TaggedImage("foo") but different
            # instances. It should mostly go away once we support proper persistence, but might
            # still happen in some weird edge cases
            warnings.warn(f"tag: {obj.tag} used for object {self._objects[obj.tag]} now overwritten by {obj}")
        self._objects[obj.tag] = obj

    def function(self, raw_f=None, image=base_image, env_dict=None):
        def decorate(raw_f):
            return Function(self, raw_f, image=image, env_dict=env_dict)

        if raw_f is None:
            # called like @session.function(x=y)
            return decorate
        else:
            # called like @session.function
            return decorate(raw_f)

    async def _get_logs(self, stdout, stderr, draining=False, timeout=BLOCKING_REQUEST_TIMEOUT):
        request = api_pb2.SessionGetLogsRequest(session_id=self.session_id, timeout=timeout, draining=draining)
        async for log_entry in self.client.stub.SessionGetLogs(request, timeout=timeout + GRPC_REQUEST_TIME_BUFFER):
            if log_entry.done:
                logger.info("No more logs")
                return
            else:
                print_logs(log_entry.data, log_entry.fd, stdout, stderr)
        if draining:
            raise Exception("Failed waiting for all logs to finish, server will kill remaining tasks")

    async def initialize(self, session_id, client):
        """Used by the container to bootstrap the session and all its objects."""
        self.session_id = session_id
        self.client = client

        req = api_pb2.SessionGetObjectsRequest(session_id=session_id)
        resp = await self.client.stub.SessionGetObjects(req)

        # TODO: check duplicates???
        self._object_ids = dict(resp.object_ids.items())

        # In the container, run forever
        self.state = SessionState.RUNNING

    async def create_object(self, obj):
        # This just register + creates the object
        self.register(obj)
        if obj.tag not in self._object_ids:
            self._object_ids[obj.tag] = await obj._create_impl(self)
        return self._object_ids[obj.tag]

    def get_object_id(self, tag):
        if self.state != SessionState.RUNNING:  # Maybe also starting?
            raise Exception("Can only look up object ids for objects on a running session")
        return self._object_ids.get(tag)

    @synchronizer.asynccontextmanager
    async def run(self, client=None, stdout=None, stderr=None):
        if stdout is None:
            stdout = sys.stdout.buffer
        if stderr is None:
            stderr = sys.stderr.buffer
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
        self._object_ids = {}

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
                # TODO: do this in parallel
                all_objects = list(self._objects.values())
                for obj in all_objects:  # can't iterate over the original hash since it might change
                    logger.debug(f"Creating object {obj}")
                    await self.create_object(obj)

                # TODO: the below is a temporary thing until we unify object creation
                req = api_pb2.SessionSetObjectsRequest(
                    session_id=self.session_id,
                    object_ids=self._object_ids,
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
            self._object_ids = None
            self.state = SessionState.NONE

    def serialize(self, obj):
        """Serializes object and replaces all references to the client class by a placeholder."""
        # TODO: probably should not be here
        buf = io.BytesIO()
        Pickler(self, ObjectMeta.type_to_name, buf).dump(obj)
        return buf.getvalue()

    def deserialize(self, s: bytes):
        """Deserializes object and replaces all client placeholders by self."""
        # TODO: probably should not be here
        return Unpickler(self, ObjectMeta.name_to_type, io.BytesIO(s)).load()
