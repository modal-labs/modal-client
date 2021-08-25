import asyncio
import grpc
import pytest
import random
import typing

from polyester.async_utils import synchronizer
from polyester.client import Client
from polyester.proto import api_pb2, api_pb2_grpc


class GRPCClientServicer(api_pb2_grpc.PolyesterClient):
    def __init__(self):
        self.requests = []
        self.done = False

    async def ClientCreate(
            self,
            request: api_pb2.ClientCreateRequest,
            context: grpc.aio.ServicerContext,
    ) -> api_pb2.ClientCreateResponse:
        self.requests.append(request)
        client_id = 'cl-123'
        return api_pb2.ClientCreateResponse(client_id=client_id)

    async def SessionCreate(
            self,
            request: api_pb2.SessionCreateRequest,
            context: grpc.aio.ServicerContext,
    ) -> api_pb2.SessionCreateResponse:
        self.requests.append(request)
        session_id = 'se-123'
        return api_pb2.SessionCreateResponse(session_id=session_id)

    #async def ClientStop(self, request: api_pb2.ByeRequest, context: grpc.aio.ServicerContext) -> api_pb2.Empty:
    #    self.requests.append(request)
    #    self.done = True
    #    return api_pb2.Empty()

    async def ClientHeartbeats(
            self,
            requests: typing.AsyncIterator[api_pb2.ClientHeartbeatRequest],
            context: grpc.aio.ServicerContext
    ) -> api_pb2.Empty:
        async for request in requests:
            self.requests.append(request)
        return api_pb2.Empty()

    async def SessionGetLogs(
            self,
            request: api_pb2.SessionGetLogsRequest,
            context: grpc.aio.ServicerContext
    ) -> typing.AsyncIterator[api_pb2.TaskLogs]:
        await asyncio.sleep(1.0)
        if self.done:
            yield api_pb2.TaskLogs(done=True)


@pytest.fixture(scope='package')
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    synchronizer._loop = loop  # TODO: SUPER HACKY
    yield loop
    loop.close()


@pytest.fixture(scope='function')
async def servicer():
    servicer = GRPCClientServicer()
    port = random.randint(8000, 8999)
    servicer.remote_addr = 'http://localhost:%d' % port
    server = grpc.aio.server()
    api_pb2_grpc.add_PolyesterClientServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:%d' % port)
    await server.start()
    yield servicer
    await server.stop(0)


@pytest.mark.asyncio
async def test_client(servicer):
    client = Client(servicer.remote_addr, api_pb2.ClientType.CLIENT, ('foo-id', 'foo-secret'))

    # TODO: let's rethink how we're doing it, should we bring the context mgr back maybe?
    await client._start()
    await client._start_client()
    await asyncio.sleep(0.1)
    await client._start_session()
    await asyncio.sleep(0.1)
    await client._close()

    assert len(servicer.requests) == 3
    for req in servicer.requests:
        print(type(req))
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CLIENT
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)
    assert isinstance(servicer.requests[2], api_pb2.SessionCreateRequest)
    # assert isinstance(servicer.requests[2], api_pb2.ByeRequest)


@pytest.mark.asyncio
async def test_container_client(servicer):
    client = Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ('ta-123', 'task-secret'))

    # TODO: let's rethink how we're doing it, should we bring the context mgr back maybe?
    await client._start()
    await client._start_client()
    await asyncio.sleep(0.1)  # enough for a handshake to go through
    await client._close()

    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CONTAINER
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)
    # assert isinstance(servicer.requests[2], api_pb2.ByeRequest)
