import asyncio
import time

import grpc
import grpc.aio

from modal_proto import api_pb2, api_pb2_grpc
from modal_utils.async_utils import TaskContext, synchronize_apis
from modal_utils.grpc_utils import ChannelPool
from modal_utils.server_connection import GRPCConnectionFactory

from .config import config, logger
from .exception import AuthError, ConnectionError, InvalidError, VersionError
from .version import __version__

CLIENT_CREATE_TIMEOUT = 5.0
HEARTBEAT_INTERVAL = 3.0


class _Client:
    def __init__(
        self,
        server_url,
        client_type,
        credentials,
        version=__version__,
    ):
        self.server_url = server_url
        self.client_type = client_type
        self.credentials = credentials
        self.version = version
        self._task_context = None
        self._channel_pool = None
        self._stub = None

    @property
    def stub(self):
        if self._stub is None:
            raise ConnectionError("The client is not connected to the modal server")
        return self._stub

    async def _start(self):
        logger.debug("Client: Starting")
        self.stopped = asyncio.Event()
        self._task_context = TaskContext()
        await self._task_context.start()
        self._connection_factory = GRPCConnectionFactory(
            self.server_url,
            self.client_type,
            self.credentials,
        )
        self._channel_pool = ChannelPool(self._task_context, self._connection_factory)
        await self._channel_pool.start()
        self._stub = api_pb2_grpc.ModalClientStub(self._channel_pool)
        try:
            t0 = time.time()
            req = api_pb2.ClientCreateRequest(
                client_type=self.client_type,
                version=self.version,
            )
            resp = await self.stub.ClientCreate(req, timeout=CLIENT_CREATE_TIMEOUT)
            self._client_id = resp.client_id
        except grpc.aio._call.AioRpcError as exc:
            ms = int(1000 * (time.time() - t0))
            if exc.code() == grpc.StatusCode.UNAUTHENTICATED:
                raise AuthError(f"Connecting to {self.server_url}: {exc.details()} (after {ms} ms)")
            elif exc.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                raise ConnectionError(f"Connecting to {self.server_url}: {exc.details()} (after {ms} ms)")
            elif exc.code() == grpc.StatusCode.FAILED_PRECONDITION:
                # TODO: include a link to the latest package
                raise VersionError(
                    f"The client version {self.version} is too old. Please update to the latest package."
                )
            else:
                raise
        if not self._client_id:
            raise InvalidError("Did not get a client id from server")

        # Start heartbeats
        self._task_context.infinite_loop(self._heartbeat, sleep=HEARTBEAT_INTERVAL)

        logger.debug("Client: Done starting")

    async def _stop(self):
        # TODO: we should trigger this using an exit handler
        logger.debug("Client: Shutting down")
        self._stub = None  # prevent any additional calls
        if self._task_context:
            await self._task_context.stop()
            self._task_context = None
        if self._channel_pool:
            await self._channel_pool.close()
            self._channel_pool = None
        logger.debug("Client: Done shutting down")
        # Needed to catch straggling CancelledErrors and GeneratorExits that propagate
        # through our chains of async generators.
        await asyncio.sleep(0.01)
        self.stopped.set()

    async def _heartbeat(self):
        if self._stub is not None:
            req = api_pb2.ClientHeartbeatRequest(client_id=self._client_id, num_connections=self._channel_pool.size())
            response: api_pb2.ClientHeartbeatResponse = await self.stub.ClientHeartbeat(req)
            if response.status == api_pb2.ClientHeartbeatResponse.CLIENT_HEARTBEAT_STATUS_GONE:
                # server has deleted this client - perform graceful shutdown
                # can't simply await self._stop here since it recursively wait for this task as well
                asyncio.ensure_future(self._stop())

    async def __aenter__(self):
        try:
            await self._start()
        except BaseException:
            await self._stop()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._stop()

    async def verify(self):
        async with self:
            # Just connect and disconnect
            pass

    @property
    def client_id(self):
        return self._client_id

    @classmethod
    def from_env(cls):
        server_url = config["server_url"]
        token_id = config["token_id"]
        token_secret = config["token_secret"]
        task_id = config["task_id"]
        task_secret = config["task_secret"]

        if task_id and task_secret:
            client_type = api_pb2.CLIENT_TYPE_CONTAINER
            credentials = (task_id, task_secret)
        elif token_id and token_secret:
            client_type = api_pb2.CLIENT_TYPE_CLIENT
            credentials = (token_id, token_secret)
        else:
            client_type = api_pb2.CLIENT_TYPE_CLIENT
            credentials = None

        client = _Client(server_url, client_type, credentials)
        return client


Client, AioClient = synchronize_apis(_Client)
