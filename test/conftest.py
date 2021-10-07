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
        self.inputs = []
        self.outputs = []

    async def ClientCreate(
        self,
        request: api_pb2.ClientCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.ClientCreateResponse:
        self.requests.append(request)
        client_id = "cl-123"
        return api_pb2.ClientCreateResponse(client_id=client_id)

    async def SessionCreate(
        self,
        request: api_pb2.SessionCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.SessionCreateResponse:
        self.requests.append(request)
        session_id = "se-123"
        return api_pb2.SessionCreateResponse(session_id=session_id)

    # async def ClientStop(self, request: api_pb2.ByeRequest, context: grpc.aio.ServicerContext) -> api_pb2.Empty:
    #    self.requests.append(request)
    #    self.done = True
    #    return api_pb2.Empty()

    async def ClientHeartbeats(
        self, requests: typing.AsyncIterator[api_pb2.ClientHeartbeatRequest], context: grpc.aio.ServicerContext
    ) -> api_pb2.Empty:
        async for request in requests:
            self.requests.append(request)
        return api_pb2.Empty()

    async def SessionGetLogs(
        self, request: api_pb2.SessionGetLogsRequest, context: grpc.aio.ServicerContext
    ) -> typing.AsyncIterator[api_pb2.TaskLogs]:
        await asyncio.sleep(1.0)
        if self.done:
            yield api_pb2.TaskLogs(done=True)

    async def FunctionGetNextInput(
        self, request: api_pb2.FunctionGetNextInputRequest, context: grpc.aio.ServicerContext
    ) -> typing.AsyncIterator[api_pb2.BufferReadResponse]:
        for input in self.inputs:
            yield input

    async def FunctionOutput(
        self, requests: typing.AsyncIterator[api_pb2.FunctionOutputRequest], context: grpc.aio.ServicerContext
    ) -> api_pb2.Empty:
        num_pushed = 0
        async for request in requests:
            self.outputs.append(request)
            num_pushed += 1
        return api_pb2.BufferWriteResponse(num_pushed=num_pushed, space_left=10000)


@pytest.fixture(scope="package")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    synchronizer._start_loop(loop)
    yield loop
    synchronizer._close_loop()


@pytest.fixture(scope="function")
async def servicer():
    servicer = GRPCClientServicer()
    port = random.randint(8000, 8999)
    servicer.remote_addr = "http://localhost:%d" % port
    server = grpc.aio.server()
    api_pb2_grpc.add_PolyesterClientServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:%d" % port)
    await server.start()
    yield servicer
    await server.stop(0)
