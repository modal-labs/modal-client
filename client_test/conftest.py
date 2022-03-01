import asyncio
import pytest
import typing

import cloudpickle
import grpc
import pkg_resources
from google.protobuf.empty_pb2 import Empty

from modal._app_singleton import set_container_app, set_default_app, set_running_app
from modal._client import Client
from modal.functions import MODAL_CLIENT_MOUNT_NAME
from modal.image import _dockerhub_python_version
from modal.proto import api_pb2, api_pb2_grpc, web_pb2
from modal.version import __version__


class GRPCClientServicer(api_pb2_grpc.ModalClient):
    def __init__(self):
        self.requests = []
        self.done = False
        self.container_inputs = []
        self.container_outputs = []
        self.object_ids = {}
        self.queue = []
        self.deployments = {
            MODAL_CLIENT_MOUNT_NAME: "mo-123",
            "foo-queue": "qu-foo",
            f"debian-slim-{_dockerhub_python_version()}": "im-123",
        }
        self.n_queues = 0
        self.files_name2sha = {}
        self.files_sha2data = {}
        self.client_calls = []
        self.n_functions = 0
        self.n_schedules = 0
        self.function2schedule = {}
        self.function_create_error = False

    async def ClientCreate(
        self,
        request: api_pb2.ClientCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.ClientCreateResponse:
        self.requests.append(request)
        client_id = "cl-123"
        if pkg_resources.parse_version(request.version) < pkg_resources.parse_version(__version__):
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Old client")
        return api_pb2.ClientCreateResponse(client_id=client_id)

    async def AppCreate(
        self,
        request: api_pb2.AppCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.AppCreateResponse:
        self.requests.append(request)
        app_id = "se-123"
        return api_pb2.AppCreateResponse(app_id=app_id)

    async def AppClientDisconnect(
        self, request: api_pb2.AppClientDisconnectRequest, context: grpc.aio.ServicerContext
    ) -> Empty:
        self.requests.append(request)
        self.done = True
        return Empty()

    async def ClientHeartbeat(
        self, request: api_pb2.ClientHeartbeatRequest, context: grpc.aio.ServicerContext
    ) -> Empty:
        self.requests.append(request)
        return Empty()

    async def AppGetLogs(
        self, request: api_pb2.AppGetLogsRequest, context: grpc.aio.ServicerContext
    ) -> typing.AsyncIterator[web_pb2.TaskLogsBatch]:
        await asyncio.sleep(0.1)
        if self.done:
            yield web_pb2.TaskLogsBatch(app_state=web_pb2.APP_STATE_STOPPED)

    async def FunctionGetInputs(
        self, request: api_pb2.FunctionGetInputsRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.FunctionGetInputsResponse:
        return self.container_inputs.pop(0)

    async def FunctionPutOutputs(
        self, request: api_pb2.FunctionPutOutputsRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.FunctionPutOutputsResponse:
        self.container_outputs.append(request)
        return api_pb2.FunctionPutOutputsResponse(status=api_pb2.WRITE_STATUS_SUCCESS)

    async def AppGetObjects(
        self, request: api_pb2.AppGetObjectsRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.AppGetObjectsResponse:
        return api_pb2.AppGetObjectsResponse(object_ids=self.object_ids)

    async def AppSetObjects(self, request: api_pb2.AppSetObjectsRequest, context: grpc.aio.ServicerContext) -> Empty:
        self.objects = dict(request.object_ids)
        return Empty()

    async def QueueCreate(
        self, request: api_pb2.QueueCreateRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.QueueCreateResponse:
        self.n_queues += 1
        return api_pb2.QueueCreateResponse(queue_id=f"qu-{self.n_queues}")

    async def QueuePut(self, request: api_pb2.QueuePutRequest, context: grpc.aio.ServicerContext) -> Empty:
        self.queue += request.values
        return Empty()

    async def QueueGet(
        self, request: api_pb2.QueueGetRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.QueueGetResponse:
        return api_pb2.QueueGetResponse(values=[self.queue.pop(0)])

    async def AppDeploy(self, request: api_pb2.AppDeployRequest, context: grpc.aio.ServicerContext) -> Empty:
        if request.object_id:
            self.deployments[request.name] = request.object_id
        elif request.object_ids:
            for label, object_id in request.object_ids.items():
                self.deployments[(request.name, label)] = object_id
        return Empty()

    async def AppIncludeObject(
        self, request: api_pb2.AppIncludeObjectRequest, context: grpc.aio.ServicerContext
    ) -> api_pb2.AppIncludeObjectResponse:
        if request.object_label:
            object_id = self.deployments.get((request.name, request.object_label))
        else:
            object_id = self.deployments.get(request.name)
        return api_pb2.AppIncludeObjectResponse(object_id=object_id)

    async def MountCreate(
        self,
        request: api_pb2.MountCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.MountCreateResponse:
        return api_pb2.MountCreateResponse(mount_id="mo-123")

    async def MountRegisterFile(
        self,
        request: api_pb2.MountRegisterFileRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.MountRegisterFileResponse:
        self.files_name2sha[request.filename] = request.sha256_hex
        return api_pb2.MountRegisterFileResponse(filename=request.filename, exists=False)

    async def MountUploadFile(
        self,
        request: api_pb2.MountUploadFileRequest,
        context: grpc.aio.ServicerContext,
    ) -> Empty:
        self.files_sha2data[request.sha256_hex] = request.data
        return Empty()

    async def MountDone(
        self,
        request: api_pb2.MountDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> Empty:
        return Empty()

    async def FunctionCreate(
        self,
        request: api_pb2.FunctionCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.FunctionCreateResponse:
        if self.function_create_error:
            raise Exception("Function create failed")
        self.n_functions += 1
        function_id = f"fu-{self.n_functions}"
        if request.schedule:
            self.function2schedule[function_id] = request.schedule
        return api_pb2.FunctionCreateResponse(function_id=function_id)

    async def FunctionMap(
        self,
        request: api_pb2.FunctionMapRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.FunctionMapResponse:
        return api_pb2.FunctionMapResponse(function_call_id="fc-out")

    async def FunctionPutInputs(
        self,
        request: api_pb2.FunctionPutInputsRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.FunctionPutInputsResponse:
        for function_input in request.inputs:
            args = cloudpickle.loads(function_input.args) if function_input.args else ()
            kwargs = cloudpickle.loads(function_input.kwargs) if function_input.kwargs else {}
            self.client_calls.append((args, kwargs))
        return api_pb2.FunctionPutInputsResponse(status=api_pb2.WRITE_STATUS_SUCCESS)

    async def FunctionGetOutputs(
        self,
        request: api_pb2.FunctionGetOutputsRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.FunctionGetOutputsResponse:
        if self.client_calls:
            args, kwargs = self.client_calls.pop(0)
            # Just return the sum of squares of all args
            res = sum(arg**2 for arg in args) + sum(value**2 for key, value in kwargs.items())
            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                data=cloudpickle.dumps(res),
            )
            return api_pb2.FunctionGetOutputsResponse(
                status=api_pb2.READ_STATUS_SUCCESS,
                outputs=[result],
            )
        else:
            return api_pb2.FunctionGetOutputsResponse(status=api_pb2.READ_STATUS_TIMEOUT)

    async def SecretCreate(
        self,
        request: api_pb2.SecretCreateRequest,
        context: grpc.aio.ServicerContext,
    ) -> api_pb2.SecretCreateResponse:
        return api_pb2.SecretCreateResponse(secret_id="st-123")


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
    async with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        yield client


@pytest.fixture(scope="function")
async def container_client(servicer):
    async with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret")) as client:
        yield client


@pytest.fixture
def reset_global_apps():
    yield
    set_default_app(None)
    set_running_app(None)
    set_container_app(None)
