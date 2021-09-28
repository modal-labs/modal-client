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
    def __init__(self, client, wait_for_logs=True):
        self.client = client
        self.objects_by_tag = {}
        self.wait_for_logs = wait_for_logs

    @classmethod
    async def _create(cls):
        client = await Client.current()
        return Session(client)

    async def _track_logs(self):
        # TODO: break it out into its own class?
        # TODO: how do we break this loop?
        while True:
            request = api_pb2.SessionGetLogsRequest(session_id=self.session_id, timeout=BLOCKING_REQUEST_TIMEOUT)
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
        self._logs_task = asyncio.create_task(self._track_logs())

    async def _stop(self, hard):
        # TODO: resurrect the Bye thing as a part of StopSession
        # req = api_pb2.ByeRequest(client_id=self.client_id)
        # await self.stub.Bye(req)
        print("STOPPING SESSION", self.session_id)
        logger.debug("Waiting for logs to flush")
        if hard or not self.wait_for_logs:
            self._logs_task.cancel()
        else:
            try:
                await asyncio.wait_for(self._logs_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.exception("Timed out waiting for logs")
                self._logs_task.cancel()
