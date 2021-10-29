import asyncio
import functools
import sys

from .async_utils import TaskContext, retry, synchronizer
from .client import Client
from .config import config, logger
from .function import Function
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIME_BUFFER, ChannelPool
from .image import base_image
from .object import Object
from .proto import api_pb2
from .utils import print_logs


@synchronizer
class Session(Object):
    def __init__(self):
        self._objects = {}
        self._initialized_object_ids = None
        super().__init__()

    async def create_or_get_object(self, obj, tag=None, return_copy=False):
        """Used to create objects dynamically on a running session.

        if return_copy is True then a copy of the object is returned.
        """
        if return_copy:
            # Don't modify the underlying object, just return a new
            obj = obj.clone()

        obj.set_context(self, self.client)
        await obj.create_from_scratch()
        return obj

    def __setitem__(self, tag, obj):
        """Register any object that will auto-created.by the session and synced to all containers."""
        if tag in self._objects:
            raise KeyError(tag)
        self._objects[tag] = obj
        ## TODO: this is kind of weird, since we modify the RHS of an an assignment.
        ## Probably need to rethink the relationships b/w objects, sessions and clients.
        self._initialize_object(tag)

    def __getitem__(self, tag):
        return self._objects[tag]

    def function(self, raw_f=None, image=base_image, env_dict=None):
        def decorate(raw_f):
            fun = Function(raw_f, image=image, env_dict=env_dict)
            # TODO: we need the containers to locate the session somehow. Right now it happens in a bit of an indirect way
            # because they import the module and locate the function first, then look up the session through the function.
            # It might make more sense to find the session first and then find the function, in which case we don't have to
            # mutate the function here to add this reference.
            fun.session = self
            tag = f"{fun.info.module_name}.{fun.info.function_name}"
            self._objects[tag] = fun
            self._initialize_object(tag)
            return fun

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

    def _initialize_object(self, tag):
        """If this tag is already present on the server, set this object to use the same object_id."""

        if not self._initialized_object_ids or tag not in self._initialized_object_ids:
            return

        object_id = self._initialized_object_ids[tag]
        obj = self._objects[tag]
        obj.set_context(self, self.client)
        obj.create_from_id(object_id)

    async def initialize(self, session_id, client):
        """Used by the container to bootstrap the session and all its objects."""
        self.session_id = session_id
        self.client = client

        req = api_pb2.SessionGetObjectsRequest(session_id=session_id)
        resp = await self.client.stub.SessionGetObjects(req)

        self._initialized_object_ids = dict(resp.object_ids.items())

        # Initialize objects that are already present.
        for tag in self._objects.keys():
            self._initialize_object(tag)

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
                    yield it
        else:
            async for it in self._run(client, stdout, stderr):
                yield it

    async def _run(self, client, stdout, stderr):
        self.client = client  # TODO: do we need to mutate state like this?

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
            for tag, obj in self._objects.items():
                logger.debug(f"Creating object {obj} with tag {tag}")
                await self.create_or_get_object(obj, tag)

            # TODO: the below is a temporary thing until we unify object creation
            req = api_pb2.SessionSetObjectsRequest(
                session_id=self.session_id,
                object_ids={tag: obj.object_id for tag, obj in self._objects.items()},
            )
            await self.client.stub.SessionSetObjects(req)

            yield self

        # Stop session (this causes the server to kill any running task)
        logger.debug("Stopping the session server-side")
        req = api_pb2.SessionStopRequest(session_id=self.session_id)
        await self.client.stub.SessionStop(req)

        # Fetch any straggling logs
        logger.debug("Draining logs")
        await self._get_logs(stdout, stderr, draining=True, timeout=config["logs_timeout"])
