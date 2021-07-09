import asyncio
import contextlib
import grpc
import pytest
import random
import typing

from polyester.client import Client, ContainerClient
from polyester.proto import api_pb2, api_pb2_grpc


class GRPCClientServicer(api_pb2_grpc.PolyesterClient):
    def __init__(self):
        self.requests = []

    async def Hello(self, request: api_pb2.HelloRequest, context: grpc.aio.ServicerContext) -> api_pb2.HelloResponse:
        self.requests.append(request)
        client_id = 'cl-123'
        return api_pb2.HelloResponse(client_id=client_id)

    async def Heartbeats(self, requests: typing.AsyncIterator[api_pb2.HeartbeatRequest], context: grpc.aio.ServicerContext) -> api_pb2.Empty:
        async for request in requests:
            self.requests.append(request)
        return api_pb2.Empty()

    async def TaskLogsGet(self, request: api_pb2.TaskLogsGetRequest, context: grpc.aio.ServicerContext) -> typing.AsyncIterator[api_pb2.TaskLogs]:
        if False:
            yield

    @contextlib.asynccontextmanager
    async def run(self, port):
        server = grpc.aio.server()
        api_pb2_grpc.add_PolyesterClientServicer_to_server(self, server)
        server.add_insecure_port('[::]:%d' % port)
        await server.start()
        yield
        await server.stop(0)        


@pytest.mark.asyncio
async def test_client():
    servicer = GRPCClientServicer()
    port = random.randint(8000, 8999)

    async with servicer.run(port):
        async with Client('http://localhost:%d' % port):
            await asyncio.sleep(0.1)  # enough for a handshake to go through

    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.HelloRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CLIENT
    assert isinstance(servicer.requests[1], api_pb2.HeartbeatRequest)


@pytest.mark.asyncio
async def test_container_client():
    servicer = GRPCClientServicer()
    port = random.randint(8000, 8999)

    async with servicer.run(port):
        async with ContainerClient('ta-123', 'http://localhost:%d' % port):
            await asyncio.sleep(0.1)  # enough for a handshake to go through

    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.HelloRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CONTAINER
    assert servicer.requests[0].task_id == 'ta-123'
    assert isinstance(servicer.requests[1], api_pb2.HeartbeatRequest)
