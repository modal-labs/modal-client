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


async def _handshake(stub, req):
    # Handshake with server
    try:
        # Verify that we are connected
        response = await retry(stub.Hello)(req, timeout=5.0)
    except grpc.aio.AioRpcError as e:
        # gRPC creates super cryptic error messages so we mask it this time and tell the user we couldn't connect
        raise Exception('gRPC failed saying hello with error "%s"' % e.details()) from None
    if response.error:
        raise Exception('Error during handshake: %s' % response.error)
    elif not response.client_id:
        raise Exception('No client id returned from handshake')
    else:
        return response.client_id


async def _heartbeats(stub, client_id, sleep=3):
    async def loop():
        while True:
            yield api_pb2.HeartbeatRequest(client_id=client_id)
            await asyncio.sleep(sleep)
    await stub.Heartbeats(loop())


def _default_from_config(z, config_key):
    return z if z is not None else config[config_key]


@synchronizer
class Client:
    ''' This class does two things:
    1. Handles serialization/deserialization of async primitives (Queue, Process)
    2. Code to track the logs loop + maybe future RPC calls on a client level
    '''
    def __init__(self, server_url=None, token_id=None, token_secret=None, loops=True):
        self.server_url = _default_from_config(server_url, 'server.url')
        self.token_id = _default_from_config(token_id, 'token.id')
        self.token_secret = _default_from_config(token_secret, 'token.secret')
        assert self.token_id and self.token_secret
        self.loops = loops

    async def start(self):
        # TODO: rewrite this to be an async context manager?
        logger.debug('Client: Starting')
        self.connection_factory = GRPCConnectionFactory(self.server_url)
        self._channel_pool = ChannelPool(self.connection_factory)
        await self._channel_pool.start()
        self.stub = api_pb2_grpc.PolyesterClientStub(self._channel_pool)

        # TODO: we probably should use the API keys on every single request, not just the handshake
        # TODO: should we encrypt the API key so it's not sent over the wire?
        req = api_pb2.HelloRequest(
            client_type=api_pb2.ClientType.CLIENT,
            token_id=self.token_id,
            token_secret=self.token_secret,
        )
        self.client_id = await _handshake(self.stub, req)

        # Start heartbeats and logs tracking, which are long-running client-wide things
        # TODO: would be nice to have some proper ownership of these tasks so they are garbage collected
        # TODO: we should have some more graceful termination of these
        if self.loops:
            self._logs_task = infinite_loop(self._track_logs, timeout=None)
            self._heartbeats_task = infinite_loop(lambda: _heartbeats(self.stub, self.client_id), timeout=None)

        logger.debug('Client: Done starting')

    async def __aenter__(self):
        await self.start()
        return self

    async def close(self):
        logger.debug('Client: Shutting down')
        await self._channel_pool.close()
        if self.loops:
            self._logs_task.cancel()
            self._heartbeats_task.cancel()
        logger.debug('Client: Done shutting down')

    async def __aexit__(self, type, value, tb):
        # Note: does not work well with KeyboardInterrupt
        await self.close()

    def serialize(self, obj):
        return serializable.serialize(self, obj)

    def deserialize(self, s: bytes):
        return serializable.deserialize(self, s)

    async def _track_logs(self):
        request = api_pb2.TaskLogsGetRequest(client_id=self.client_id, timeout=BLOCKING_REQUEST_TIMEOUT)
        async for log_entry in self.stub.TaskLogsGet(request, timeout=GRPC_REQUEST_TIMEOUT):
            if log_entry.done:
                # TODO: we can't break here, we should only break if we're in closing mode
                logger.info('No more logs!!!')
            else:
                print_logs(log_entry.data, log_entry.fd)


@synchronizer
class ContainerClient:
    def __init__(self, task_id, server_url=None, task_secret=None):
        self.task_id = task_id
        self.server_url = _default_from_config(server_url, 'server.url')
        self.task_secret = _default_from_config(task_secret, 'task.secret')
        assert self.task_secret

    async def start(self):
        # TODO: rewrite this to be an async context manager?
        self.connection_factory = GRPCConnectionFactory(self.server_url)
        self._channel_pool = ChannelPool(self.connection_factory)
        await self._channel_pool.start()
        self.stub = api_pb2_grpc.PolyesterClientStub(self._channel_pool)
        req = api_pb2.HelloRequest(
            client_type=api_pb2.ClientType.CONTAINER,
            task_id=self.task_id,
            task_secret=self.task_secret,
        )
        self.client_id = await _handshake(self.stub, req)
        self._heartbeats_task = infinite_loop(lambda: _heartbeats(self.stub, self.client_id), timeout=None)

    async def __aenter__(self):
        await self.start()
        return self

    async def close(self):
        await self._channel_pool.close()
        self._heartbeats_task.cancel()

    async def __aexit__(self, type, value, tb):
        await self.close()

    def serialize(self, obj):
        return serializable.serialize(self, obj)

    def deserialize(self, s: bytes):
        return serializable.deserialize(self, s)

    async def function_get(self, function_id):
        request = api_pb2.FunctionGetRequest(function_id=function_id)
        response = await self.stub.FunctionGet(request)
        return self.deserialize(response.data)

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
