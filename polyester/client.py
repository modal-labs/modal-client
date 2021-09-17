import asyncio
import grpc.aio
import io
import os

from .async_utils import infinite_loop, retry, synchronizer
from .config import config, logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT, ChannelPool
from .local_server import LocalServer
from .object import ObjectMeta
from .proto import api_pb2, api_pb2_grpc
from .serialization import Pickler, Unpickler
from .server_connection import GRPCConnectionFactory
from .utils import print_logs


@synchronizer
class Client:
    _client_from_env = None

    def __init__(
        self,
        server_url,
        client_type,
        credentials,
    ):
        self.server_url = server_url
        self.client_type = client_type
        self.credentials = credentials
        self._heartbeats_task = None
        self._task_logs_task = None

    async def _start(self):
        logger.debug("Client: Starting")
        self.connection_factory = GRPCConnectionFactory(
            self.server_url,
            self.client_type,
            self.credentials,
        )
        self._channel_pool = ChannelPool(self.connection_factory)
        await self._channel_pool.start()
        self.stub = api_pb2_grpc.PolyesterClientStub(self._channel_pool)

    async def _start_client(self):
        req = api_pb2.ClientCreateRequest(client_type=self.client_type)
        resp = await self.stub.ClientCreate(req)
        # except grpc.aio.AioRpcError as e:
        #    # gRPC creates super cryptic error messages so we mask it this time and tell the user we couldn't connect
        #    raise Exception('gRPC failed saying hello with error "%s"' % e.details()) from None
        self.client_id = resp.client_id

        # Start heartbeats
        # TODO: would be nice to have some proper ownership of these tasks so they are garbage collected
        # TODO: we should have some more graceful termination of these
        self._heartbeats_task = infinite_loop(self._heartbeats, timeout=None)

        logger.debug("Client: Done starting")

    async def _start_session(self):
        req = api_pb2.SessionCreateRequest(client_id=self.client_id)
        resp = await self.stub.SessionCreate(req)
        self.session_id = resp.session_id

        # See comment about heartbeats task, same thing applies here
        self._logs_task = asyncio.create_task(self._track_logs())

    async def _close(self):
        # TODO: when is this actually called?
        logger.debug("Client: Shutting down")
        # TODO: resurrect the Bye thing as a part of StopSession
        # req = api_pb2.ByeRequest(client_id=self.client_id)
        # await self.stub.Bye(req)
        if self._task_logs_task:
            logger.debug("Waiting for logs to flush")
            try:
                await asyncio.wait_for(self._task_logs_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.exception("Timed out waiting for logs")
        if self._heartbeats_task:
            self._heartbeats_task.cancel()
        await self._channel_pool.close()
        logger.debug("Client: Done shutting down")

    async def _heartbeats(self, sleep=3):
        async def loop():
            while True:
                yield api_pb2.ClientHeartbeatRequest(client_id=self.client_id)
                await asyncio.sleep(sleep)

        await self.stub.ClientHeartbeats(loop())

    async def _track_logs(self):
        # TODO: break it out into its own class?
        # TODO: how do we break this loop?
        while True:
            request = api_pb2.SessionGetLogsRequest(session_id=self.session_id, timeout=BLOCKING_REQUEST_TIMEOUT)
            async for log_entry in self.stub.SessionGetLogs(request, timeout=GRPC_REQUEST_TIMEOUT):
                if log_entry.done:
                    logger.info("No more logs")
                    break
                else:
                    print_logs(log_entry.data, log_entry.fd)

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

    @classmethod
    async def from_env(cls, reuse=True):
        if cls._client_from_env is not None and reuse:
            return cls._client_from_env

        server_url = config["server.url"]
        token_id = config["token.id"]
        token_secret = config["token.secret"]
        task_id = config["task.id"]
        task_secret = config["task.secret"]

        if task_id and task_secret:
            client_type = api_pb2.ClientType.CONTAINER
            credentials = (task_id, task_secret)
        elif token_id and token_secret:
            client_type = api_pb2.ClientType.CLIENT
            credentials = (token_id, token_secret)
        else:
            client_type = api_pb2.ClientType.CLIENT
            credentials = None

        client = cls._client_from_env = Client(server_url, client_type, credentials)
        await client._start()
        await client._start_client()
        if client_type == api_pb2.ClientType.CLIENT:
            await client._start_session()
        return client
