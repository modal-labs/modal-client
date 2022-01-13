import asyncio
import typing

import grpc
import pkg_resources
import pytest

from modal import Session
from modal._client import Client
from modal._session_singleton import (
    set_container_session,
    set_default_session,
    set_running_session,
)
from modal.proto import api_pb2, api_pb2_grpc
from modal.version import __version__


class GRPCClientServicer(api_pb2_grpc.ModalClient):
    def __init__(self):
        self.requests = []
        self.done = False
        self.inputs = []
        self.outputs = []
        self.object_ids = {}
        self.queue = []

    async def ClientCreate(
        self,
        request: api_pb2.ClientCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.ClientCreateResponse:
        self.requests.append(request)
        client_id = "cl-123"
        if pkg_resources.parse_version(request.version) < pkg_resources.parse_version(__version__):
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Old client")
            return
        return api_pb2.ClientCreateResponse(client_id=client_id)

    async def SessionCreate(
        self,
        request: api_pb2.SessionCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.SessionCreateResponse:
        self.requests.append(request)
        session_id = "se-123"
        return api_pb2.SessionCreateResponse(session_id=session_id)

    async def SessionStop(
        self, request: api_pb2.SessionStopRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.Empty:
        self.requests.append(request)
        self.done = True
        return api_pb2.Empty()

    async def ClientHeartbeat(
        self, request: api_pb2.ClientHeartbeatRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.Empty:
        self.requests.append(request)
        return api_pb2.Empty()

    async def SessionGetLogs(
        self, request: api_pb2.SessionGetLogsRequest, context: grpc.aio.ServicerContext
    ) -> typing.AsyncIterator[api_pb2.TaskLogsBatch]:
        await asyncio.sleep(1.0)
        if self.done:
            yield api_pb2.TaskLogsBatch(done=True)

    async def FunctionGetNextInput(
        self, request: api_pb2.FunctionGetNextInputRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.BufferReadResponse:
        return self.inputs.pop(0)

    async def FunctionOutput(
        self, request: api_pb2.FunctionOutputRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.BufferWriteResponse:
        self.outputs.append(request)
        return api_pb2.BufferWriteResponse(status=api_pb2.BufferWriteResponse.BufferWriteStatus.SUCCESS)

    async def SessionGetObjects(
        self, request: api_pb2.SessionGetObjectsRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.SessionGetObjectsResponse:
        return api_pb2.SessionGetObjectsResponse(object_ids=self.object_ids)

    async def SessionSetObjects(
        self, request: api_pb2.SessionSetObjectsRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.Empty:
        self.objects = dict(request.object_ids)
        return api_pb2.Empty()

    async def QueueCreate(
        self, request: api_pb2.QueueCreateRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.QueueCreateResponse:
        return api_pb2.QueueCreateResponse(queue_id="qu-123456")

    async def QueuePut(self, request: api_pb2.QueuePutRequest, context: grpc.aio.ServicerContext) -> api_pb2.Empty:
        self.queue += request.values
        return api_pb2.Empty()

    async def QueueGet(
        self, request: api_pb2.QueueGetRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.QueueGetResponse:
        return api_pb2.QueueGetResponse(values=[self.queue.pop(0)])

    async def SessionUseObject(
        self, request: api_pb2.SessionUseObjectRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.SessionUseObjectResponse:
        return api_pb2.SessionUseObjectResponse(found=True, object_id="qu-98765")


@pytest.fixture(scope="function")
async def servicer():
    servicer = GRPCClientServicer()
    server = grpc.aio.server()
    api_pb2_grpc.add_ModalClientServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    servicer.remote_addr = "http://localhost:%d" % port
    await server.start()
    yield servicer
    await server.stop(0)


@pytest.fixture(scope="function")
async def client(servicer):
    async with Client(servicer.remote_addr, api_pb2.ClientType.CLIENT, ("foo-id", "foo-secret")) as client:
        yield client


@pytest.fixture(scope="function")
async def container_client(servicer):
    async with Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ("ta-123", "task-secret")) as client:
        yield client


@pytest.fixture
def reset_global_sessions():
    yield
    set_default_session(None)
    set_running_session(None)
    set_container_session(None)
