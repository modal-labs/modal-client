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
            heartbeat_loop,  # TODO: rethink
            task_logs_loop,  # TODO: rethink
    ):
        self.server_url = server_url
        self.client_type = client_type
        self.credentials = credentials
        self.heartbeat_loop = heartbeat_loop
        self.task_logs_loop = task_logs_loop

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

        req = api_pb2.HelloRequest(client_type=self.client_type)
        try:
            # Verify that we are connected
            response = await retry(self.stub.Hello)(req, timeout=5.0)
        except grpc.aio.AioRpcError as e:
            # gRPC creates super cryptic error messages so we mask it this time and tell the user we couldn't connect
            raise Exception('gRPC failed saying hello with error "%s"' % e.details()) from None
        if response.error:
            raise Exception('Error during handshake: %s' % response.error)
        elif not response.client_id:
            raise Exception('No client id returned from handshake')

        self.client_id = response.client_id

        # Start heartbeats and logs tracking, which are long-running client-wide things
        # TODO: would be nice to have some proper ownership of these tasks so they are garbage collected
        # TODO: we should have some more graceful termination of these
        if self.task_logs_loop:
            self._logs_task = asyncio.create_task(self._track_logs())
        if self.heartbeat_loop:
            self._heartbeats_task = infinite_loop(self._heartbeats, timeout=None)

        logger.debug('Client: Done starting')

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
                yield api_pb2.HeartbeatRequest(client_id=self.client_id)
                await asyncio.sleep(sleep)
        await self.stub.Heartbeats(loop())

    def serialize(self, obj):
        return serializable.serialize(self, obj)

    def deserialize(self, s: bytes):
        return serializable.deserialize(self, s)

    async def _track_logs(self):
        # TODO: how do we break this loop?
        while True:
            request = api_pb2.TaskLogsGetRequest(client_id=self.client_id, timeout=BLOCKING_REQUEST_TIMEOUT)
            async for log_entry in self.stub.TaskLogsGet(request, timeout=GRPC_REQUEST_TIMEOUT):
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
            cls._default_client = Client(server_url, api_pb2.ClientType.CLIENT, credentials, True, True)
            await cls._default_client._start()
        return cls._default_client

    @classmethod
    async def get_container_client(cls):
        if cls._default_container_client is None:
            server_url = config['server.url']
            credentials = (config['task.id'], config['task.secret'])
            cls._default_container_client = Client(server_url, api_pb2.ClientType.CONTAINER, credentials, True, False)
            await cls._default_container_client._start()
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
