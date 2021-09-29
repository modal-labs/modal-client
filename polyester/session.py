import asyncio

from .async_utils import infinite_loop, retry, synchronizer
from .client import Client
from .config import logger
from .ctx_mgr_utils import CtxMgr
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT, ChannelPool
from .proto import api_pb2
from .utils import print_logs


@synchronizer
class Session(CtxMgr):
    def __init__(self, client):
        self.client = client
        self.objects_by_tag = {}

    @classmethod
    async def _create(cls):
        client = await Client.current()
        return Session(client)

    async def _get_logs(self, draining=False, timeout=BLOCKING_REQUEST_TIMEOUT):
        request = api_pb2.SessionGetLogsRequest(session_id=self.session_id, timeout=timeout, draining=draining)
        async for log_entry in self.client.stub.SessionGetLogs(request, timeout=GRPC_REQUEST_TIMEOUT):
            if log_entry.done:
                logger.info("No more logs")
                break
            else:
                print_logs(log_entry.data, log_entry.fd)

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
