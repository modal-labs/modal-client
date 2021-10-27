import asyncio
import io
import os

import grpc.aio

from .async_utils import retry, synchronizer, TaskContext
from .config import config, logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT, ChannelPool
from .object import ObjectMeta
from .proto import api_pb2, api_pb2_grpc
from .serialization import Pickler, Unpickler
from .server_connection import GRPCConnectionFactory


@synchronizer
class Client:
    current_from_env = None

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
        self.stopped = asyncio.Event()
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
        except Exception:
            # Just helpful for debugging
            logger.info(f"server_url={self.server_url}")
            raise

        # Start heartbeats
        # TODO: this is an ugly abuse of the task context
        self._task_context = TaskContext()
        await self._task_context.__aenter__()
        self._task_context.infinite_loop(self._heartbeats, timeout=None)

        logger.debug("Client: Done starting")

    async def _stop(self):
        # TODO: we should trigger this using an exit handler
        self.stopped.set()  # notify heartbeat loop to quit.
        logger.debug("Client: Shutting down")
        await self._task_context.__aexit__(None, None, None)
        await self._channel_pool.close()
        logger.debug("Client: Done shutting down")
        # Needed to catch straggling CancelledErrors and GeneratorExits that propagate
        # through our chains of async generators.
        await asyncio.sleep(0.01)

    async def _heartbeats(self, sleep=3):
        async def loop():
            while not self.stopped.is_set():
                yield api_pb2.ClientHeartbeatRequest(client_id=self.client_id)
                # Wait for event, or until sleep seconds have passed.
                try:
                    await asyncio.wait_for(self.stopped.wait(), sleep)
                except asyncio.TimeoutError:
                    continue

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

    async def __aenter__(self):
        await self._start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._stop()

    @classmethod
    async def from_env(cls):
        if cls.current_from_env is not None:
            return cls.current_from_env

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

        client = Client(server_url, client_type, credentials)
        await client._start()
        cls.current_from_env = client  # TODO: minor risk of overwriting one
        return client
