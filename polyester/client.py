import asyncio
import grpc.aio
import io
import os

from .async_utils import infinite_loop, retry, synchronizer
from .ctx_mgr_utils import CtxMgr
from .config import config, logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT, ChannelPool
from .local_server import LocalServer
from .object import ObjectMeta
from .proto import api_pb2, api_pb2_grpc
from .serialization import Pickler, Unpickler
from .server_connection import GRPCConnectionFactory


@synchronizer
class Client(CtxMgr):
    def __init__(
        self,
        server_url,
        client_type,
        credentials,
    ):
        self.server_url = server_url
        self.client_type = client_type
        self.credentials = credentials

    async def _start(self):
        logger.debug("Client: Starting")
        try:
            self.connection_factory = GRPCConnectionFactory(
                self.server_url,
                self.client_type,
                self.credentials,
            )
            self._channel_pool = ChannelPool(self.connection_factory)
            await self._channel_pool.start()
            self.stub = api_pb2_grpc.PolyesterClientStub(self._channel_pool)
            req = api_pb2.ClientCreateRequest(client_type=self.client_type)
            resp = await self.stub.ClientCreate(req)
            self.client_id = resp.client_id
        except:
            # Just helpful for debugging
            logger.info(f"{self.server_url=}")
            raise

        # Start heartbeats
        # TODO: would be nice to have some proper ownership of these tasks so they are garbage collected
        # TODO: we should have some more graceful termination of these
        self._heartbeats_task = infinite_loop(self._heartbeats, timeout=None)

        logger.debug("Client: Done starting")

    async def _stop(self, hard):
        logger.debug("Client: Shutting down")
        self._heartbeats_task.cancel()
        await self._channel_pool.close()
        logger.debug("Client: Done shutting down")

    async def _heartbeats(self, sleep=3):
        async def loop():
            while True:
                yield api_pb2.ClientHeartbeatRequest(client_id=self.client_id)
                await asyncio.sleep(sleep)

        await self.stub.ClientHeartbeats(loop())

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
    async def _create(cls):
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

        return Client(server_url, client_type, credentials)
