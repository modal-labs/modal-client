import asyncio
import grpc.aio
import os
import uuid

from .async_utils import infinite_loop, synchronizer, retry
from .grpc_utils import ChannelPool, GRPC_REQUEST_TIMEOUT, BLOCKING_REQUEST_TIMEOUT
from .config import config, logger
from .local_server import LocalServer
from .proto import api_pb2, api_pb2_grpc
from .serialization import serializable
from .server_connection import GRPCConnectionFactory
from .utils import print_logs


@synchronizer
class Client:
    _default_client = None
    _default_container_client = None

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
        logger.debug('Client: Starting')
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
        #except grpc.aio.AioRpcError as e:
        #    # gRPC creates super cryptic error messages so we mask it this time and tell the user we couldn't connect
        #    raise Exception('gRPC failed saying hello with error "%s"' % e.details()) from None
        self.client_id = resp.client_id

        # Start heartbeats
        # TODO: would be nice to have some proper ownership of these tasks so they are garbage collected
        # TODO: we should have some more graceful termination of these
        self._heartbeats_task = infinite_loop(self._heartbeats, timeout=None)

        logger.debug('Client: Done starting')

    async def _start_session(self):
        req = api_pb2.SessionCreateRequest(client_id=self.client_id)
        resp = await self.stub.SessionCreate(req)
        self.session_id = resp.session_id

        # See comment about heartbeats task, same thing applies here
        self._logs_task = asyncio.create_task(self._track_logs())

    async def _close(self):
        # TODO: when is this actually called?
        logger.debug('Client: Shutting down')
        # TODO: resurrect the Bye thing as a part of StopSession
        #req = api_pb2.ByeRequest(client_id=self.client_id)
        #await self.stub.Bye(req)
        if self.task_logs_loop:
            logger.debug('Waiting for logs to flush')
            try:
                await asyncio.wait_for(self._logs_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.exception('Timed out waiting for logs')
        if self.heartbeat_loop:
            self._heartbeats_task.cancel()
        await self._channel_pool.close()
        logger.debug('Client: Done shutting down')

    async def _heartbeats(self, sleep=3):
        async def loop():
            while True:
                yield api_pb2.ClientHeartbeatRequest(client_id=self.client_id)
                await asyncio.sleep(sleep)
        await self.stub.ClientHeartbeats(loop())

    def serialize(self, obj):
        return serializable.serialize(self, obj)

    def deserialize(self, s: bytes):
        return serializable.deserialize(self, s)

    async def _track_logs(self):
        # TODO: break it out into its own class?
        # TODO: how do we break this loop?
        while True:
            request = api_pb2.SessionGetLogsRequest(
                session_id=self.session_id,
                timeout=BLOCKING_REQUEST_TIMEOUT
            )
            async for log_entry in self.stub.SessionGetLogs(request, timeout=GRPC_REQUEST_TIMEOUT):
                if log_entry.done:
                    logger.info('No more logs')
                    break
                else:
                    print_logs(log_entry.data, log_entry.fd)

    @classmethod
    async def get_client(cls):
        if cls._default_client is None:
            server_url = config['server.url']
            token_id = config['token.id']
            token_secret = config['token.secret']
            if token_id and token_secret:
                credentials = (token_id, token_secret)
            else:
                credentials = None
            cls._default_client = Client(server_url, api_pb2.ClientType.CLIENT, credentials)
            await cls._default_client._start()
            await cls._default_client._start_client()
            await cls._default_client._start_session()
        return cls._default_client

    @classmethod
    async def get_container_client(cls):
        if cls._default_container_client is None:
            server_url = config['server.url']
            credentials = (config['task.id'], config['task.secret'])
            cls._default_container_client = Client(server_url, api_pb2.ClientType.CONTAINER, credentials)
            await cls._default_container_client._start()
            await cls._default_container_client._start_client()
        return cls._default_container_client

    # TODO: code below is container-specific and we should probably move it out of here

    async def function_get_next_input(self, task_id, function_id):
        while True:
            idempotency_key = str(uuid.uuid4())
            request = api_pb2.FunctionGetNextInputRequest(
                task_id=task_id, function_id=function_id, idempotency_key=idempotency_key, timeout=BLOCKING_REQUEST_TIMEOUT)
            response = await retry(self.stub.FunctionGetNextInput)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if response.stop:
                return (None, None, True)
            if response.input_id:
                break

        return (self.deserialize(response.data), response.input_id, False)

    async def function_output(self, input_id, status, data, exception: str, traceback: str):
        data_serialized = self.serialize(data)
        output = api_pb2.GenericResult(status=status, data=data_serialized, exception=exception, traceback=traceback)
        idempotency_key = str(uuid.uuid4())
        request = api_pb2.FunctionOutputRequest(input_id=input_id, idempotency_key=idempotency_key, output=output)
        await retry(self.stub.FunctionOutput)(request)
