import asyncio
import contextlib

from .async_utils import infinite_loop, retry, synchronizer
from .client import Client
from .config import logger
from .ctx_mgr_utils import CtxMgr
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIME_BUFFER, ChannelPool
from .image import base_image
from .object import Object
from .proto import api_pb2
from .utils import print_logs


@synchronizer
class DeprecatedSession(CtxMgr):
    def __init__(self, client, stdout=None, stderr=None):
        self.client = client
        self.objects_by_tag = {}
        # log queues
        self._stdout = stdout
        self._stderr = stderr

    @classmethod
    async def _create(cls):
        client = await Client.current()
        return DeprecatedSession(client)

    async def _get_logs(self, draining=False, timeout=BLOCKING_REQUEST_TIMEOUT):
        request = api_pb2.SessionGetLogsRequest(session_id=self.session_id, timeout=timeout, draining=draining)
        async for log_entry in self.client.stub.SessionGetLogs(request, timeout=timeout + GRPC_REQUEST_TIME_BUFFER):
            if log_entry.done:
                logger.info("No more logs")
                break
            else:
                print_logs(log_entry.data, log_entry.fd, self._stdout, self._stderr)

    async def _start(self):
        req = api_pb2.SessionCreateRequest(client_id=self.client.client_id)
        resp = await self.client.stub.SessionCreate(req)
        self.session_id = resp.session_id

        # See comment about heartbeats task, same thing applies here
        self._logs_task = infinite_loop(self._get_logs)

    async def _stop(self, hard):
        # Stop session (this causes the server to kill any running task)
        logger.debug("Stopping the session server-side")
        req = api_pb2.SessionStopRequest(session_id=self.session_id)
        await self.client.stub.SessionStop(req)

        # Kill the existing log loop
        self._logs_task.cancel()

        # Fetch any straggling logs
        if not hard:
            logger.debug("Draining logs")
            await self._get_logs(draining=True, timeout=10.0)


@synchronizer
class Session(Object):
    def __init__(self):
        super().__init__()

    async def create_or_get(self, obj, tag=None, return_copy=False):
        if return_copy:
            # Don't modify the underlying object, just return a joined object
            cls = type(obj)
            new_obj = cls.__new__(cls)
            new_obj.args = obj.args
            obj = new_obj

        obj.session = self
        obj.tag = tag
        obj.client = self.client
        obj.object_id = await obj._create_or_get()
        obj.created  = True
        return obj

    def function(self, raw_f, image=base_image):
        fun = image.function(raw_f)
        return fun

    async def _start(self, client):
        if client is None:
            client = await Client.current()

        self.client = client

        # Get all objects on this session right now
        objects = {tag: getattr(self, tag)
                   for tag in dir(self)
                   if isinstance(getattr(self, tag), Object)}

        # Start session
        # TODO: pass in a list of tags that need to be pre-created
        req = api_pb2.SessionCreateRequest(client_id=client.client_id)
        resp = await client.stub.SessionCreate(req)
        self.session_id = resp.session_id

        # Create all members
        # TODO: do this in parallel
        for tag, obj in objects.items():
            await self.create_or_get(obj, tag)

    async def _stop(self):
        pass

    @contextlib.asynccontextmanager
    async def run(self, client=None):
        # TODO: support sync, and return_copy
        await self._start(client)
        yield
        await self._stop()
