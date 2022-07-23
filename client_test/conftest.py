import asyncio
import contextlib
import os
import pytest
import shutil
import tempfile
import typing
from pathlib import Path

import aiohttp.web
import aiohttp.web_runner
import cloudpickle
import grpc
import pkg_resources
from google.protobuf.empty_pb2 import Empty
from grpc import StatusCode
from grpc.aio import ServicerContext

from modal.app import _App
from modal.client import AioClient, Client
from modal.image import _dockerhub_python_version
from modal.mount import client_mount_name
from modal.version import __version__
from modal_proto import api_pb2, api_pb2_grpc
from modal_utils.async_utils import synchronize_apis
from modal_utils.http_utils import run_temporary_http_server


class GRPCClientServicer(api_pb2_grpc.ModalClient):
    def __init__(self, blob_host, blobs):
        self.n_blobs = 0
        self.blob_host = blob_host
        self.blobs = blobs  # shared dict
        self.requests = []
        self.done = False
        self.rate_limit_times = 0
        self.fail_get_inputs = False
        self.container_inputs = []
        self.container_outputs = []
        self.object_ids = None
        self.queue = []
        self.deployed_apps = {
            client_mount_name(): "ap-x",
            "foo-queue": "ap-y",
            f"debian-slim-{_dockerhub_python_version()}": "ap-z",
            "conda": "ap-c",
        }
        self.app_objects = {
            "ap-x": {"": "mo-123"},
            "ap-y": {"foo-queue": "qu-foo"},
            "ap-z": {"": "im-123"},
            "ap-c": {"": "im-456"},
        }
        self.n_queues = 0
        self.files_name2sha = {}
        self.files_sha2data = {}
        self.client_calls = []
        self.n_functions = 0
        self.n_schedules = 0
        self.function2schedule = {}
        self.function_create_error = False
        self.heartbeat_status_code = None
        self.n_apps = 0
        self.output_idx = 0

        self.shared_volume_files = []
        self.images = {}

    async def BlobCreate(
        self, request: api_pb2.BlobCreateRequest, context: ServicerContext = None, timeout=None
    ) -> api_pb2.BlobCreateResponse:
        self.n_blobs += 1
        blob_id = f"bl-{self.n_blobs}"
        upload_url = f"{self.blob_host}/upload?blob_id={blob_id}"
        return api_pb2.BlobCreateResponse(blob_id=blob_id, upload_url=upload_url)

    async def BlobGet(
        self, request: api_pb2.BlobGetRequest, context: ServicerContext = None, timeout=None
    ) -> api_pb2.BlobGetResponse:
        download_url = f"{self.blob_host}/download?blob_id={request.blob_id}"
        return api_pb2.BlobGetResponse(download_url=download_url)

    async def ClientCreate(
        self, request: api_pb2.ClientCreateRequest, context: ServicerContext = None, timeout=None
    ) -> api_pb2.ClientCreateResponse:
        self.requests.append(request)
        client_id = "cl-123"
        if request.version == "deprecated":
            return api_pb2.ClientCreateResponse(client_id=client_id, deprecation_warning="SUPER OLD")
        elif pkg_resources.parse_version(request.version) < pkg_resources.parse_version(__version__):
            await context.abort(StatusCode.FAILED_PRECONDITION, "Old client")
        else:
            return api_pb2.ClientCreateResponse(client_id=client_id)

    async def AppCreate(
        self,
        request: api_pb2.AppCreateRequest,
        context: ServicerContext = None,
    ) -> api_pb2.AppCreateResponse:
        self.requests.append(request)
        self.n_apps += 1
        app_id = f"ap-{self.n_apps}"
        return api_pb2.AppCreateResponse(app_id=app_id)

    async def AppClientDisconnect(
        self, request: api_pb2.AppClientDisconnectRequest, context: ServicerContext = None
    ) -> Empty:
        self.requests.append(request)
        self.done = True
        return Empty()

    async def ClientHeartbeat(self, request: api_pb2.ClientHeartbeatRequest, context: ServicerContext = None) -> Empty:
        self.requests.append(request)
        if self.heartbeat_status_code:
            await context.abort(self.heartbeat_status_code, f"Client {request.client_id} heartbeat failed.")
        return Empty()

    async def ImageGetOrCreate(
        self, request: api_pb2.ImageGetOrCreateRequest, context: ServicerContext = None
    ) -> api_pb2.ImageGetOrCreateResponse:
        idx = len(self.images)
        self.images[idx] = request.image
        return api_pb2.ImageGetOrCreateResponse(image_id=f"im-{idx}")

    async def ImageJoin(
        self, request: api_pb2.ImageJoinRequest, context: ServicerContext = None
    ) -> api_pb2.ImageJoinResponse:
        return api_pb2.ImageJoinResponse(
            result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
        )

    async def AppGetLogs(
        self, request: api_pb2.AppGetLogsRequest, context: ServicerContext = None, timeout=None
    ) -> typing.AsyncIterator[api_pb2.TaskLogsBatch]:
        await asyncio.sleep(0.1)
        if self.done:
            yield api_pb2.TaskLogsBatch(app_done=True)

    async def FunctionGetInputs(
        self, request: api_pb2.FunctionGetInputsRequest, context: ServicerContext = None
    ) -> api_pb2.FunctionGetInputsResponse:
        assert request.function_id
        if self.fail_get_inputs:
            await context.abort(StatusCode.INTERNAL)
        elif self.rate_limit_times > 0:
            self.rate_limit_times -= 1
            await context.abort(StatusCode.RESOURCE_EXHAUSTED, "Rate limit exceeded")
        elif not self.container_inputs:
            await asyncio.sleep(request.timeout)
            return api_pb2.FunctionGetInputsResponse(inputs=[])
        else:
            return self.container_inputs.pop(0)

    async def FunctionPutOutputs(
        self, request: api_pb2.FunctionPutOutputsRequest, context: ServicerContext = None
    ) -> Empty:
        self.container_outputs.append(request)
        return Empty()

    async def AppGetObjects(
        self, request: api_pb2.AppGetObjectsRequest, context: ServicerContext = None
    ) -> api_pb2.AppGetObjectsResponse:
        if self.object_ids:
            object_ids = self.object_ids
        else:
            object_ids = self.app_objects.get(request.app_id, {})
        return api_pb2.AppGetObjectsResponse(object_ids=object_ids)

    async def AppSetObjects(self, request: api_pb2.AppSetObjectsRequest, context: ServicerContext = None) -> Empty:
        self.app_objects[request.app_id] = dict(request.object_ids)
        return Empty()

    async def QueueCreate(
        self, request: api_pb2.QueueCreateRequest, context: ServicerContext = None
    ) -> api_pb2.QueueCreateResponse:
        self.n_queues += 1
        return api_pb2.QueueCreateResponse(queue_id=f"qu-{self.n_queues}")

    async def QueuePut(self, request: api_pb2.QueuePutRequest, context: ServicerContext = None) -> Empty:
        self.queue += request.values
        return Empty()

    async def QueueGet(
        self, request: api_pb2.QueueGetRequest, context: ServicerContext = None
    ) -> api_pb2.QueueGetResponse:
        return api_pb2.QueueGetResponse(values=[self.queue.pop(0)])

    async def AppDeploy(self, request: api_pb2.AppDeployRequest, context: ServicerContext = None) -> Empty:
        self.deployed_apps[request.name] = request.app_id
        return Empty()

    async def AppGetByDeploymentName(
        self, request: api_pb2.AppGetByDeploymentNameRequest, context: ServicerContext = None
    ) -> api_pb2.AppGetByDeploymentNameResponse:
        return api_pb2.AppGetByDeploymentNameResponse(app_id=self.deployed_apps.get(request.name))

    async def AppLookupObject(
        self, request: api_pb2.AppLookupObjectRequest, context: ServicerContext = None
    ) -> api_pb2.AppLookupObjectResponse:
        object_id = None
        app_id = self.deployed_apps.get(request.app_name)
        if app_id is not None:
            app_objects = self.app_objects[app_id]
            if request.object_tag:
                object_id = app_objects.get(request.object_tag)
            else:
                (object_id,) = list(app_objects.values())
        return api_pb2.AppLookupObjectResponse(object_id=object_id)

    async def MountPutFile(
        self,
        request: api_pb2.MountPutFileRequest,
        context: ServicerContext,
    ) -> api_pb2.MountPutFileResponse:
        if request.WhichOneof("data_oneof") is not None:
            self.files_sha2data[request.sha256_hex] = {"data": request.data, "data_blob_id": request.data_blob_id}
            return api_pb2.MountPutFileResponse(exists=True)
        else:
            return api_pb2.MountPutFileResponse(exists=False)

    async def MountBuild(
        self,
        request: api_pb2.MountBuildRequest,
        context: ServicerContext,
    ) -> api_pb2.MountBuildResponse:
        for file in request.files:
            self.files_name2sha[file.filename] = file.sha256_hex
        return api_pb2.MountBuildResponse(mount_id="mo-123")

    async def SharedVolumeCreate(
        self,
        request: api_pb2.SharedVolumeCreateRequest,
        context: ServicerContext,
    ) -> api_pb2.SharedVolumeCreateResponse:
        return api_pb2.SharedVolumeCreateResponse(shared_volume_id="sv-123")

    async def FunctionCreate(
        self,
        request: api_pb2.FunctionCreateRequest,
        context: ServicerContext,
    ) -> api_pb2.FunctionCreateResponse:
        if self.function_create_error:
            raise Exception("Function create failed")
        if request.existing_function_id:
            function_id = request.existing_function_id
        else:
            self.n_functions += 1
            function_id = f"fu-{self.n_functions}"
        if request.schedule:
            self.function2schedule[function_id] = request.schedule
        if request.function.webhook_config.type:
            web_url = "http://xyz.internal"
        else:
            web_url = None
        return api_pb2.FunctionCreateResponse(function_id=function_id, web_url=web_url)

    async def FunctionMap(
        self,
        request: api_pb2.FunctionMapRequest,
        context: ServicerContext,
    ) -> api_pb2.FunctionMapResponse:
        self.output_idx = 0
        return api_pb2.FunctionMapResponse(function_call_id="fc-out")

    async def FunctionPutInputs(
        self,
        request: api_pb2.FunctionPutInputsRequest,
        context: ServicerContext,
    ) -> Empty:
        for function_input in request.inputs:
            args, kwargs = cloudpickle.loads(function_input.args) if function_input.args else ((), {})
            self.client_calls.append((args, kwargs))
        return Empty()

    async def FunctionGetOutputs(
        self,
        request: api_pb2.FunctionGetOutputsRequest,
        context: ServicerContext,
    ) -> api_pb2.FunctionGetOutputsResponse:
        if self.client_calls:
            args, kwargs = self.client_calls.pop(0)
            # Just return the sum of squares of all args
            res = sum(arg**2 for arg in args) + sum(value**2 for key, value in kwargs.items())
            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                data=cloudpickle.dumps(res),
                idx=self.output_idx,
            )
            self.output_idx += 1
            return api_pb2.FunctionGetOutputsResponse(outputs=[result])
        else:
            await context.abort(StatusCode.DEADLINE_EXCEEDED, "Read timeout")

    async def SecretCreate(
        self,
        request: api_pb2.SecretCreateRequest,
        context: ServicerContext,
    ) -> api_pb2.SecretCreateResponse:
        return api_pb2.SecretCreateResponse(secret_id="st-123")


@pytest.fixture(scope="session")
async def blob_server(event_loop):
    blobs = {}

    async def upload(request):
        blob_id = request.query["blob_id"]
        content = await request.content.read()
        if content == b"FAILURE":
            return aiohttp.web.Response(status=500)
        blobs[blob_id] = content
        return aiohttp.web.Response(text="Hello, world")

    async def download(request):
        blob_id = request.query["blob_id"]
        if blob_id == "bl-failure":
            return aiohttp.web.Response(status=500)
        return aiohttp.web.Response(body=blobs[blob_id])

    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.put("/upload", upload)])
    app.add_routes([aiohttp.web.get("/download", download)])

    async with run_temporary_http_server(app) as host:
        yield host, blobs


@pytest.fixture(scope="function")
async def servicer(blob_server):
    blob_host, blobs = blob_server
    servicer = GRPCClientServicer(blob_host, blobs)
    server = None

    async def _start_servicer():
        nonlocal server
        server = grpc.aio.server()
        api_pb2_grpc.add_ModalClientServicer_to_server(servicer, server)
        port = server.add_insecure_port("[::]:0")
        servicer.remote_addr = "http://localhost:%d" % port
        await server.start()

    async def _stop_servicer():
        await server.stop(0)

    _, aio_start_servicer = synchronize_apis(_start_servicer)
    _, aio_stop_servicer = synchronize_apis(_stop_servicer)

    await aio_start_servicer()
    try:
        yield servicer
    finally:
        await aio_stop_servicer()


@pytest.fixture(scope="function")
async def aio_client(servicer):
    async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        yield client


@pytest.fixture(scope="function")
async def client(servicer):
    with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        yield client


@pytest.fixture(scope="function")
async def aio_container_client(servicer):
    async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret")) as client:
        yield client


@pytest.fixture(name="mock_dir", scope="session")
def mock_dir_factory():
    """Sets up a temp dir with content as specified in a nested dict

    Example usage:
    spec = {
        "foo": {
            "bar.txt": "some content"
        },
    }

    with mock_dir(spec) as root_dir:
        assert os.path.exists(os.path.join(root_dir, "foo", "bar.txt"))
    """

    @contextlib.contextmanager
    def mock_dir(root_spec):
        def rec_make(dir, dir_spec):
            for filename, spec in dir_spec.items():
                path = os.path.join(dir, filename)
                if isinstance(spec, str):
                    with open(path, "w") as f:
                        f.write(spec)
                else:
                    os.mkdir(path)
                    rec_make(path, spec)

        # Windows has issues cleaning up TempDirectory: https://www.scivision.dev/python-tempfile-permission-error-windows
        # Seems to have been fixed for some python versions in https://github.com/python/cpython/pull/10320.
        root_dir = tempfile.mkdtemp()
        rec_make(root_dir, root_spec)
        yield root_dir
        shutil.rmtree(root_dir, ignore_errors=True)

    return mock_dir


@pytest.fixture(autouse=True)
def reset_container_app():
    try:
        yield
    finally:
        _App.reset_container()


@pytest.fixture(scope="module")
def test_dir(request):
    """Absolute path to directory containing test file."""
    root_dir = Path(request.config.rootdir)
    test_dir = Path(os.getenv("PYTEST_CURRENT_TEST")).parent
    return root_dir / test_dir
