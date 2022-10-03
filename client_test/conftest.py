import asyncio
import contextlib
import inspect
import os
import pytest
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import aiohttp.web
import aiohttp.web_runner
import cloudpickle
import grpclib.server
import pkg_resources
import pytest_asyncio
from google.protobuf.empty_pb2 import Empty
from grpclib import GRPCError, Status

from modal import __version__
from modal.app import _App
from modal.client import AioClient, Client
from modal.image import _dockerhub_python_version
from modal.mount import client_mount_name
from modal_proto import api_grpc, api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import find_free_port, patch_mock_servicer
from modal_utils.http_utils import run_temporary_http_server


@patch_mock_servicer
class MockClientServicer(api_grpc.ModalClientBase):
    def __init__(self, blob_host, blobs):
        self.n_blobs = 0
        self.blob_host = blob_host
        self.blobs = blobs  # shared dict
        self.requests = []
        self.done = False
        self.rate_limit_sleep_duration = None
        self.fail_get_inputs = False
        self.container_inputs = []
        self.container_outputs = []
        self.queue = []
        self.deployed_apps = {
            client_mount_name(): "ap-x",
            "foo-queue": "ap-y",
            f"debian-slim-{_dockerhub_python_version()}-{__version__}": "ap-z",
            f"conda-{__version__}": "ap-c",
            "my-proxy": "ap-proxy",
        }
        self.app_objects = {
            "ap-x": {"": "mo-123"},
            "ap-y": {"foo-queue": "qu-foo"},
            "ap-z": {"": "im-123"},
            "ap-c": {"": "im-456"},
            "ap-proxy": {"": "pr-123"},
        }
        self.n_queues = 0
        self.files_name2sha = {}
        self.files_sha2data = {}
        self.client_calls = {}
        self.function_is_running = False
        self.n_functions = 0
        self.n_schedules = 0
        self.function2schedule = {}
        self.function_create_error = False
        self.heartbeat_status_code = None
        self.n_apps = 0

        self.task_result = None

        self.shared_volume_files = []
        self.images = {}
        self.fail_blob_create = []
        self.blob_create_metadata = None

        self.app_functions = {}
        self.fcidx = 0
        self.created_secrets = 0

        @self.function_body
        def default_function_body(*args, **kwargs):
            return sum(arg**2 for arg in args) + sum(value**2 for key, value in kwargs.items())

    def function_body(self, func):
        """Decorator for setting the function that will be called for any FunctionGetOutputs calls"""
        self._function_body = func
        return func

    async def BlobCreate(self, stream):
        await stream.recv_message()
        # This is used to test retry_transient_errors, see grpc_utils_test.py
        self.blob_create_metadata = stream.metadata
        if len(self.fail_blob_create) > 0:
            status_code = self.fail_blob_create.pop()
            raise GRPCError(status_code, "foobar")
        else:
            self.n_blobs += 1
            blob_id = f"bl-{self.n_blobs}"
            upload_url = f"{self.blob_host}/upload?blob_id={blob_id}"
            await stream.send_message(api_pb2.BlobCreateResponse(blob_id=blob_id, upload_url=upload_url))

    async def BlobGet(self, stream):
        request = await stream.recv_message()
        download_url = f"{self.blob_host}/download?blob_id={request.blob_id}"
        await stream.send_message(api_pb2.BlobGetResponse(download_url=download_url))

    async def ClientCreate(self, stream):
        request = await stream.recv_message()
        self.requests.append(request)
        client_id = "cl-123"
        if request.version == "timeout":
            await asyncio.sleep(60)
            await stream.send_message(api_pb2.ClientCreateResponse(client_id=client_id))
        elif request.version == "unauthenticated":
            raise GRPCError(Status.UNAUTHENTICATED, "failed authentication")
        elif request.version == "deprecated":
            await stream.send_message(
                api_pb2.ClientCreateResponse(client_id=client_id, deprecation_warning="SUPER OLD")
            )
        elif pkg_resources.parse_version(request.version) < pkg_resources.parse_version(__version__):
            raise GRPCError(Status.FAILED_PRECONDITION, "Old client")
        else:
            await stream.send_message(api_pb2.ClientCreateResponse(client_id=client_id))

    async def AppCreate(self, stream):
        request = await stream.recv_message()
        self.requests.append(request)
        self.n_apps += 1
        app_id = f"ap-{self.n_apps}"
        await stream.send_message(api_pb2.AppCreateResponse(app_id=app_id))

    async def AppClientDisconnect(self, stream):
        request = await stream.recv_message()
        self.requests.append(request)
        self.done = True
        await stream.send_message(Empty())

    async def ClientHeartbeat(self, stream) -> Empty:
        request = await stream.recv_message()
        self.requests.append(request)
        if self.heartbeat_status_code:
            raise GRPCError(self.heartbeat_status_code, f"Client {request.client_id} heartbeat failed.")
        await stream.send_message(Empty())

    async def ImageGetOrCreate(self, stream):
        request = await stream.recv_message()
        idx = len(self.images)
        self.images[idx] = request.image
        await stream.send_message(api_pb2.ImageGetOrCreateResponse(image_id=f"im-{idx}"))

    async def ImageJoin(self, stream):
        await stream.recv_message()
        await stream.send_message(
            api_pb2.ImageJoinResponse(result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS))
        )

    async def AppGetLogs(self, stream):
        await stream.recv_message()
        await asyncio.sleep(0.1)
        if self.done:
            await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    async def FunctionGetInputs(self, stream):
        request = await stream.recv_message()
        assert request.function_id
        if self.fail_get_inputs:
            raise GRPCError(Status.INTERNAL)
        elif self.rate_limit_sleep_duration is not None:
            s = self.rate_limit_sleep_duration
            self.rate_limit_sleep_duration = None
            await stream.send_message(api_pb2.FunctionGetInputsResponse(rate_limit_sleep_duration=s))
        elif not self.container_inputs:
            await asyncio.sleep(request.timeout)
            await stream.send_message(api_pb2.FunctionGetInputsResponse(inputs=[]))
        else:
            await stream.send_message(self.container_inputs.pop(0))

    async def FunctionPutOutputs(self, stream):
        request = await stream.recv_message()
        self.container_outputs.append(request)
        await stream.send_message(Empty())

    async def AppGetObjects(self, stream):
        request = await stream.recv_message()
        object_ids = self.app_objects.get(request.app_id, {})
        await stream.send_message(api_pb2.AppGetObjectsResponse(object_ids=object_ids))

    async def AppSetObjects(self, stream):
        request = await stream.recv_message()
        self.app_objects[request.app_id] = dict(request.indexed_object_ids)
        await stream.send_message(Empty())

    async def QueueCreate(self, stream):
        await stream.recv_message()
        self.n_queues += 1
        await stream.send_message(api_pb2.QueueCreateResponse(queue_id=f"qu-{self.n_queues}"))

    async def QueuePut(self, stream):
        request = await stream.recv_message()
        self.queue += request.values
        await stream.send_message(Empty())

    async def QueueGet(self, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.QueueGetResponse(values=[self.queue.pop(0)]))

    async def AppDeploy(self, stream):
        request = await stream.recv_message()
        self.deployed_apps[request.name] = request.app_id
        await stream.send_message(api_pb2.AppDeployResponse(url="http://test.modal.com/foo/bar"))

    async def AppGetByDeploymentName(self, stream):
        request = await stream.recv_message()
        await stream.send_message(api_pb2.AppGetByDeploymentNameResponse(app_id=self.deployed_apps.get(request.name)))

    async def AppLookupObject(self, stream):
        request = await stream.recv_message()
        object_id = None
        app_id = self.deployed_apps.get(request.app_name)
        if app_id is not None:
            app_objects = self.app_objects[app_id]
            if request.object_tag:
                object_id = app_objects.get(request.object_tag)
            else:
                (object_id,) = list(app_objects.values())
        await stream.send_message(api_pb2.AppLookupObjectResponse(object_id=object_id))

    async def MountPutFile(self, stream):
        request = await stream.recv_message()
        if request.WhichOneof("data_oneof") is not None:
            self.files_sha2data[request.sha256_hex] = {"data": request.data, "data_blob_id": request.data_blob_id}
            await stream.send_message(api_pb2.MountPutFileResponse(exists=True))
        else:
            await stream.send_message(api_pb2.MountPutFileResponse(exists=False))

    async def MountBuild(self, stream):
        request = await stream.recv_message()
        for file in request.files:
            self.files_name2sha[file.filename] = file.sha256_hex
        await stream.send_message(api_pb2.MountBuildResponse(mount_id="mo-123"))

    async def SharedVolumeCreate(self, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.SharedVolumeCreateResponse(shared_volume_id="sv-123"))

    async def FunctionCreate(self, stream):
        request = await stream.recv_message()
        if self.function_create_error:
            raise GRPCError(Status.INTERNAL, "Function create failed")
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
        await stream.send_message(api_pb2.FunctionCreateResponse(function_id=function_id, web_url=web_url))

    async def FunctionMap(self, stream):
        self.fcidx += 1
        await stream.recv_message()
        await stream.send_message(api_pb2.FunctionMapResponse(function_call_id=f"fc-{self.fcidx}"))

    async def FunctionPutInputs(self, stream):
        request = await stream.recv_message()
        response_items = []
        function_calls = self.client_calls.setdefault(request.function_call_id, [])
        for item in request.inputs:
            args, kwargs = cloudpickle.loads(item.input.args) if item.input.args else ((), {})
            input_id = f"in-{len(function_calls)}"
            response_items.append(api_pb2.FunctionPutInputsResponseItem(input_id=input_id, idx=item.idx))
            function_calls.append(((item.idx, input_id), (args, kwargs)))
        await stream.send_message(api_pb2.FunctionPutInputsResponse(inputs=response_items))

    async def FunctionGetOutputs(self, stream):
        request = await stream.recv_message()

        client_calls = self.client_calls.get(request.function_call_id, [])
        if client_calls and not self.function_is_running:
            popidx = len(client_calls) // 2  # simulate that results don't always come in order
            (idx, input_id), (args, kwargs) = client_calls.pop(popidx)
            # Just return the sum of squares of all args
            try:
                res = self._function_body(*args, **kwargs)
            except Exception as exc:
                serialized_exc = cloudpickle.dumps(exc)
                result = api_pb2.GenericResult(
                    status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                    data=serialized_exc,
                    exception=repr(exc),
                    traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                )
                output = api_pb2.FunctionGetOutputsItem(input_id="in-123", idx=0, result=result)
                await stream.send_message(api_pb2.FunctionGetOutputsResponse(outputs=[output]))
                return

            if inspect.iscoroutine(res):
                results = [await res]
            elif inspect.isgenerator(res):
                results = list(res)
            else:
                results = [res]

            outputs = []
            for index, value in enumerate(results):
                result = api_pb2.GenericResult(
                    status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                    data=cloudpickle.dumps(value),
                )
                item = api_pb2.FunctionGetOutputsItem(
                    input_id=f"in-123-{index}",
                    idx=idx,
                    result=result,
                )
                outputs.append(item)

            await stream.send_message(api_pb2.FunctionGetOutputsResponse(outputs=outputs))
        else:
            await stream.send_message(api_pb2.FunctionGetOutputsResponse(outputs=[]))

    async def SecretCreate(self, stream):
        await stream.recv_message()
        self.created_secrets += 1
        await stream.send_message(api_pb2.SecretCreateResponse(secret_id="st-123"))

    async def SecretList(self, stream):
        await stream.recv_message()
        items = [api_pb2.SecretListItem(label=f"dummy-secret-{i}") for i in range(self.created_secrets)]
        await stream.send_message(api_pb2.SecretListResponse(items=items))

    async def TaskResult(self, stream):
        request = await stream.recv_message()
        self.task_result = request.result
        await stream.send_message(Empty())


@pytest_asyncio.fixture(scope="session")
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


@pytest_asyncio.fixture(scope="function")
async def servicer(blob_server):
    blob_host, blobs = blob_server
    servicer = MockClientServicer(blob_host, blobs)
    server = None

    async def _start_servicer():
        nonlocal server
        server = grpclib.server.Server([servicer])
        port = find_free_port()
        servicer.remote_addr = "http://localhost:%d" % port
        await server.start("0.0.0.0", port)

    async def _stop_servicer():
        server.close()
        await server.wait_closed()

    _, aio_start_servicer = synchronize_apis(_start_servicer)
    _, aio_stop_servicer = synchronize_apis(_stop_servicer)

    await aio_start_servicer()
    try:
        yield servicer
    finally:
        await aio_stop_servicer()


@pytest_asyncio.fixture(scope="function")
async def aio_client(servicer):
    async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def client(servicer):
    with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
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
        cwd = os.getcwd()
        try:
            os.chdir(root_dir)
            yield
        finally:
            os.chdir(cwd)
            shutil.rmtree(root_dir, ignore_errors=True)

    return mock_dir


@pytest.fixture(autouse=True)
def reset_sys_modules():
    # Needed since some tests will import dynamic modules
    backup = sys.modules.copy()
    try:
        yield
    finally:
        sys.modules = backup


@pytest.fixture(autouse=True)
def reset_container_app():
    try:
        yield
    finally:
        _App._reset_container()


@pytest.fixture(scope="module")
def test_dir(request):
    """Absolute path to directory containing test file."""
    root_dir = Path(request.config.rootdir)
    test_dir = Path(os.getenv("PYTEST_CURRENT_TEST")).parent
    return root_dir / test_dir
