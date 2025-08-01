# Copyright Modal Labs 2024
from __future__ import annotations

import asyncio
import builtins
import contextlib
import dataclasses
import datetime
import hashlib
import inspect
import os
import platform
import pytest
import random
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, Union, get_args

import aiohttp.web
import aiohttp.web_runner
import grpclib.server
import jwt
import pkg_resources
import pytest_asyncio
from google.protobuf.empty_pb2 import Empty
from grpclib import GRPCError, Status
from grpclib.events import RecvRequest, listen

from modal import __version__, config
from modal._functions import _Function
from modal._runtime.container_io_manager import _ContainerIOManager
from modal._serialization import deserialize, deserialize_params, serialize_data_format
from modal._utils.async_utils import asyncify, synchronize_api
from modal._utils.blob_utils import BLOCK_SIZE, MAX_OBJECT_SIZE_BYTES
from modal._utils.grpc_testing import patch_mock_servicer
from modal._utils.grpc_utils import find_free_port
from modal._utils.http_utils import run_temporary_http_server
from modal._utils.jwt_utils import DecodedJwt
from modal._vendor import cloudpickle
from modal.app import _App
from modal.client import Client
from modal.cls import _Cls
from modal.image import ImageBuilderVersion
from modal.mount import PYTHON_STANDALONE_VERSIONS, client_mount_name, python_standalone_mount_name
from modal_proto import api_grpc, api_pb2

VALID_GPU_TYPES = ["ANY", "T4", "L4", "A10G", "L40S", "A100", "A100-40GB", "A100-80GB", "H100"]
VALID_CLOUD_PROVIDERS = ["AWS", "GCP", "OCI", "AUTO", "XYZ"]


@dataclasses.dataclass
class Volume:
    version: "api_pb2.VolumeFsVersion.ValueType"
    files: dict[str, VolumeFile]


@dataclasses.dataclass
class VolumeFile:
    data: bytes
    mode: int
    data_blob_id: Optional[str] = None
    data_sha256_hex: Optional[str] = None
    block_hashes: Optional[list[bytes]] = None


@dataclasses.dataclass
class GrpcErrorAndCount:
    """Helper class that holds a gRPC error and the number of times it should be raised."""

    grpc_error: Status
    count: int


# TODO: Isolate all test config from the host
@pytest.fixture(scope="function", autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")


@pytest.fixture(scope="function", autouse=True)
def ignore_local_config():
    # When running tests locally, we don't want to pick up the local .modal.toml file
    config._user_config = {}
    yield


@pytest.fixture
def tmp_path_with_content(tmp_path):
    (tmp_path / "data.txt").write_text("hello")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sub").write_text("world")
    (tmp_path / "module").mkdir()
    (tmp_path / "module" / "__init__.py").write_text("foo")
    (tmp_path / "module" / "sub.py").write_text("bar")
    (tmp_path / "module" / "sub").mkdir()
    (tmp_path / "module" / "sub" / "__init__.py").write_text("baz")
    (tmp_path / "module" / "sub" / "foo.pyc").write_text("baz")
    (tmp_path / "module" / "sub" / "sub.py").write_text("qux")

    return tmp_path


class FunctionsRegistry:
    def __init__(self):
        self._functions: dict[str, api_pb2.Function] = {}
        self._functions_data: dict[str, api_pb2.FunctionData] = {}

    def __getitem__(self, key):
        if key in self._functions:
            return self._functions[key]
        return self._functions_data[key]

    def __setitem__(self, key, value):
        if isinstance(value, api_pb2.FunctionData):
            self._functions_data[key] = value
        else:
            self._functions[key] = value

    def __len__(self):
        return len(self._functions) + len(self._functions_data)

    def __contains__(self, key):
        return key in self._functions or key in self._functions_data

    def get(self, key, default=None):
        try:
            return self._functions[key]
        except KeyError:
            return self._functions_data.get(key, default)

    def values(self):
        return list(self._functions.values()) + list(self._functions_data.values())

    def items(self):
        return list(self._functions.items()) + list(self._functions_data.items())


@patch_mock_servicer
class MockClientServicer(api_grpc.ModalClientBase):
    # TODO(erikbern): add more annotations
    container_inputs: list[api_pb2.FunctionGetInputsResponse]
    container_outputs: list[api_pb2.FunctionPutOutputsRequest]
    fc_data_in: defaultdict[str, asyncio.Queue[api_pb2.DataChunk]]
    fc_data_out: defaultdict[str, asyncio.Queue[api_pb2.DataChunk]]

    # Set when the server runs
    client_addr: str
    container_addr: str

    def __init__(self, blob_host, blobs, blocks, files_sha2data, credentials, port):
        self.default_published_client_mount = "mo-123"
        self.default_username = "test-user"
        self.use_blob_outputs = False
        self.put_outputs_barrier = threading.Barrier(
            1, timeout=10
        )  # set to non-1 to get lock-step of output pushing within a test
        self.get_inputs_barrier = threading.Barrier(
            1, timeout=10
        )  # set to non-1 to get lock-step of input releases within a test

        self.app_state_history = defaultdict(list)
        self.app_heartbeats: dict[str, int] = defaultdict(int)
        self.container_snapshot_requests = 0
        self.n_blobs = 0
        self.blob_host = blob_host
        self.blobs = blobs  # shared dict
        self.blocks = blocks  # shared dict
        self.requests = []
        self.done = False
        self.rate_limit_sleep_duration = None
        self.fail_get_inputs = False
        self.fail_put_inputs_with_grpc_error: GrpcErrorAndCount | None = None
        self.fail_put_inputs_with_stream_terminated_error = 0
        self.failure_status = api_pb2.GenericResult.GENERIC_STATUS_FAILURE
        self.slow_put_inputs = False
        self.container_inputs = []
        self.container_outputs = []
        self.fail_get_data_out = []
        self.fc_data_in = defaultdict(lambda: asyncio.Queue())  # unbounded
        self.fc_data_out = defaultdict(lambda: asyncio.Queue())  # unbounded
        self.queue: dict[bytes, list[bytes]] = {b"": []}
        self.deployed_apps: dict[tuple[str, str], str] = {}
        self.app_environments: dict[str, str] = {}
        self.app_deployment_history: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self.app_deployment_history["ap-x"] = [
            {
                "app_id": "ap-x",
                "deployed_at": datetime.datetime.now().timestamp(),
                "version": 1,
                "client_version": str(pkg_resources.parse_version(__version__)),
                "deployed_by": "foo-user",
                "tag": "latest",
            }
        ]
        self.app_objects = {}
        self.app_unindexed_objects = {}
        self.max_object_size_bytes = MAX_OBJECT_SIZE_BYTES
        self.n_inputs = 0
        self.n_entry_ids = 0
        self.n_queues = 0
        self.n_dict_heartbeats = 0
        self.n_queue_heartbeats = 0
        self.n_nfs_heartbeats = 0
        self.n_vol_heartbeats = 0
        self.n_mounts = 0
        self.n_mount_files = 0
        self.mount_contents = {self.default_published_client_mount: {"/pkg/modal_client.py": "0x1337"}}
        self.files_name2sha = {}
        self.files_sha2data = files_sha2data
        self.function_id_for_function_call = {}
        self.function_call_inputs = {}
        self.function_call_inputs_update_event = asyncio.Event()
        self.sync_client_retries_enabled = False
        self.function_is_running = False
        self.n_functions = 0
        self.n_schedules = 0
        self.function2schedule = {}
        self.function_create_error: BaseException | None = None
        self.heartbeat_status_code = None
        self.n_apps = 0
        self.classes = []
        self.environments = {"main": "en-1"}

        self.task_result = None

        self.nfs_files: dict[str, dict[str, api_pb2.SharedVolumePutFileRequest]] = defaultdict(dict)
        self.volumes: dict[str, Volume] = {}
        self.images = {}
        self.image_build_function_ids = {}
        self.image_builder_versions = {}
        self.force_built_images = []
        self.fail_blob_create = []
        self.blob_create_metadata = None
        self.blob_multipart_threshold = 10_000_000

        self.precreated_functions = set()

        self.app_functions: FunctionsRegistry = FunctionsRegistry()
        self.bound_functions: dict[tuple[str, bytes], str] = {}
        self.function_params: dict[str, tuple[tuple, dict[str, Any]]] = {}
        self.function_options: dict[str, api_pb2.FunctionOptions] = {}
        self.fcidx = 0

        self.function_serialized = None
        self.class_serialized = None

        self.client_hello_metadata = None

        self.dicts = {}
        self.secrets = {}

        self.deployed_dicts = {}
        self.default_published_client_mount = "mo-123"
        self.deployed_mounts = {
            (client_mount_name(), api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL): self.default_published_client_mount,
            **{
                (
                    python_standalone_mount_name(version),
                    api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
                ): f"mo-py{version.replace('.', '')}"
                for version in PYTHON_STANDALONE_VERSIONS
            },
        }

        self.deployed_nfss = {}
        self.deployed_queues = {}
        self.deployed_secrets = {}
        self.deployed_volumes = {}

        self.cleared_function_calls = set()

        self.cancelled_calls = []

        self.app_client_disconnect_count = 0
        self.app_get_logs_initial_count = 0
        self.app_set_objects_count = 0
        self.app_publish_count = 0

        self.volume_counter = 0
        # Volume-id -> commit/reload count
        self.volume_commits: dict[str, int] = defaultdict(int)
        self.volume_reloads: dict[str, int] = defaultdict(int)

        self.sandbox_defs = []
        self.sandbox_app_id = None
        self.sandbox: asyncio.subprocess.Process = None
        self.sandbox_result: api_pb2.GenericResult | None = None

        self.shell_prompt = None
        self.container_exec: asyncio.subprocess.Process = None
        self.container_exec_result: api_pb2.GenericResult | None = None

        self.token_flow_localhost_port = None
        self.queue_max_len = 100

        self.container_heartbeat_response = None
        self.container_heartbeat_abort = threading.Event()

        self.image_join_sleep_duration = None

        token_id, token_secret = credentials
        self.required_creds = {token_id: token_secret}  # Any of this will be accepted
        self.last_metadata = None

        self.function_get_server_warnings = None
        self.resp_jitter_secs: float = 0.0
        self.port = port
        # AttemptAwait will return a failure until this is 0. It is decremented by 1 each time AttemptAwait is called.
        self.attempt_await_failures_remaining = 0
        # Value returned by AuthTokenGet
        self.auth_token = jwt.encode({"exp": int(time.time()) + 3600}, "my-secret-key", algorithm="HS256")
        self.auth_tokens_generated = 0

        @self.function_body
        def default_function_body(*args, **kwargs):
            return sum(arg**2 for arg in args) + sum(value**2 for key, value in kwargs.items())

    def get_data_chunks(self, function_call_id) -> list[api_pb2.DataChunk]:
        # ugly - get data chunks associated with the first input to this servicer
        data_chunks: list[api_pb2.DataChunk] = []
        if function_call_id in self.fc_data_out:
            try:
                while True:
                    chunk = self.fc_data_out[function_call_id].get_nowait()
                    data_chunks.append(chunk)
            except asyncio.QueueEmpty:
                pass
        return data_chunks

    def set_resp_jitter(self, secs: float) -> None:
        # TODO: It'd be great to make this easy to apply to all gRPC method handlers.
        # Some way to decorate `stream.send_message`.
        self.resp_jitter_secs = secs

    async def recv_request(self, event: RecvRequest):
        # Make sure metadata is correct
        self.last_metadata = event.metadata
        for header in [
            "x-modal-python-version",
            "x-modal-client-version",
            "x-modal-client-type",
        ]:
            if header not in event.metadata:
                raise GRPCError(Status.FAILED_PRECONDITION, f"Missing {header} header")

        client_version = event.metadata["x-modal-client-version"]
        assert isinstance(client_version, str)
        if client_version == "unauthenticated":
            raise GRPCError(Status.UNAUTHENTICATED, "failed authentication")
        elif client_version == "timeout":
            await asyncio.sleep(60)
        elif client_version == "deprecated":
            pass  # dumb magic fixture constant
        elif pkg_resources.parse_version(client_version) < pkg_resources.parse_version(__version__):
            raise GRPCError(Status.FAILED_PRECONDITION, "Old client")

        if event.metadata["x-modal-client-type"] == str(api_pb2.CLIENT_TYPE_CLIENT):
            if event.method_name in [
                "/modal.client.ModalClient/TokenFlowCreate",
                "/modal.client.ModalClient/TokenFlowWait",
            ]:
                pass  # Methods that don't require authentication
            else:
                token_id = event.metadata.get("x-modal-token-id")
                token_secret = event.metadata.get("x-modal-token-secret")
                if not token_id or not token_secret:
                    raise GRPCError(Status.UNAUTHENTICATED, f"No credentials for method {event.method_name}")
                elif token_id not in self.required_creds:
                    raise GRPCError(Status.UNAUTHENTICATED, f"Invalid {token_id=!r} for method {event.method_name}")
                elif self.required_creds[token_id] != token_secret:
                    raise GRPCError(Status.UNAUTHENTICATED, f"Invalid token secret for for method {event.method_name}")
        elif event.metadata["x-modal-client-type"] == str(api_pb2.CLIENT_TYPE_CONTAINER):
            for header in [
                "x-modal-token-id",
                "x-modal-token-secret",
                "x-modal-task-id",  # old
                "x-modal-task-secret",  # old
            ]:
                if header in event.metadata:
                    raise GRPCError(Status.FAILED_PRECONDITION, f"Container client should not set header {header}")
        else:
            raise GRPCError(Status.FAILED_PRECONDITION, "Unknown client type")

    def function_body(self, func):
        """Decorator for setting the function that will be called for any FunctionGetOutputs calls"""
        self._function_body = func
        return func

    def function_by_name(self, name: str, params: tuple[tuple, dict[str, Any]] | None = None) -> api_pb2.Function:
        matches = []
        all_names = []
        for function_id, fun in self.app_functions.items():
            all_names.append(fun.function_name)
            if fun.function_name != name:
                continue
            if fun.is_class and params:
                if self.function_params.get(function_id, ((), {})) != params:
                    continue

            matches.append(fun)
        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise ValueError("More than 1 matching function")
        raise ValueError(f"No function with name {name=} {params=} ({all_names=})")

    def container_heartbeat_return_now(self, response: api_pb2.ContainerHeartbeatResponse):
        self.container_heartbeat_response = response
        self.container_heartbeat_abort.set()

    def _get_input_plane_url(self, definition: Union[api_pb2.Function, api_pb2.FunctionData]) -> Optional[str]:
        input_plane_region = definition.experimental_options.get("input_plane_region")
        return f"http://127.0.0.1:{self.port}" if input_plane_region else None

    def _get_input_plane_region(self, definition: Union[api_pb2.Function, api_pb2.FunctionData]) -> Optional[str]:
        input_plane_region = definition.experimental_options.get("input_plane_region")
        return input_plane_region

    def get_function_metadata(self, object_id: str) -> api_pb2.FunctionHandleMetadata:
        definition: api_pb2.Function = self.app_functions[object_id]
        return api_pb2.FunctionHandleMetadata(
            function_name=definition.function_name,
            function_type=definition.function_type,
            web_url=definition.web_url,
            is_method=definition.is_method,
            use_method_name=definition.use_method_name,
            use_function_id=definition.use_function_id,
            class_parameter_info=definition.class_parameter_info,
            method_handle_metadata={
                method_name: api_pb2.FunctionHandleMetadata(
                    function_name=method_definition.function_name,
                    function_type=method_definition.function_type,
                    web_url=method_definition.web_url,
                    is_method=True,
                    use_method_name=method_name,
                    function_schema=method_definition.function_schema,
                )
                for method_name, method_definition in definition.method_definitions.items()
            },
            function_schema=definition.function_schema,
            input_plane_url=self._get_input_plane_url(definition),
            input_plane_region=self._get_input_plane_region(definition),
            max_object_size_bytes=self.max_object_size_bytes,
        )

    def get_object_metadata(self, object_id) -> api_pb2.Object:
        if object_id.startswith("fu-"):
            res = api_pb2.Object(function_handle_metadata=self.get_function_metadata(object_id))

        elif object_id.startswith("cs-"):
            res = api_pb2.Object(class_handle_metadata=api_pb2.ClassHandleMetadata())

        elif object_id.startswith("mo-"):
            mount_handle_metadata = api_pb2.MountHandleMetadata(content_checksum_sha256_hex="abc123")
            res = api_pb2.Object(mount_handle_metadata=mount_handle_metadata)

        elif object_id.startswith("sb-"):
            sandbox_handle_metadata = api_pb2.SandboxHandleMetadata(result=self.sandbox_result)
            res = api_pb2.Object(sandbox_handle_metadata=sandbox_handle_metadata)

        elif object_id.startswith("vo-"):
            volume_metadata = api_pb2.VolumeMetadata(version=self.volumes[object_id].version)
            res = api_pb2.Object(volume_metadata=volume_metadata)

        else:
            res = api_pb2.Object()

        res.object_id = object_id
        return res

    def get_environment(self, environment_name: Optional[str] = None) -> str:
        if environment_name is None:
            return next(iter(self.environments))  # Use first environment as default
        return environment_name

    def mounts_excluding_published_client(self):
        return {
            mount_id: content
            for mount_id, content in self.mount_contents.items()
            if mount_id != self.default_published_client_mount
        }

    def app_get_layout(self, app_id):
        # Returns the app layout for any deployed app
        app_objects = self.app_objects.get(app_id, {})
        function_ids = {}
        class_ids = {}
        object_ids = set(app_objects.values()) | set(self.app_unindexed_objects.get(app_id, []))
        for tag, object_id in app_objects.items():
            if _Function._is_id_type(object_id):
                function_ids[tag] = object_id
                definition = self.app_functions[object_id]
                if isinstance(definition, api_pb2.FunctionData):
                    for ranked_fn in definition.ranked_functions:
                        object_ids |= {obj.object_id for obj in ranked_fn.function.object_dependencies}
                else:
                    object_ids |= {obj.object_id for obj in definition.object_dependencies}
            elif _Cls._is_id_type(object_id):
                class_ids[tag] = object_id

        return api_pb2.AppLayout(
            function_ids=function_ids,
            class_ids=class_ids,
            objects=[self.get_object_metadata(object_id) for object_id in object_ids],
        )

    ### App

    async def AppCreate(self, stream):
        request: api_pb2.AppCreateRequest = await stream.recv_message()
        self.requests.append(request)
        self.n_apps += 1
        app_id = f"ap-{self.n_apps}"
        self.app_state_history[app_id].append(api_pb2.APP_STATE_INITIALIZING)
        self.app_environments[app_id] = self.get_environment(request.environment_name)
        await stream.send_message(
            api_pb2.AppCreateResponse(app_id=app_id, app_page_url="https://modaltest.com/apps/ap-123")
        )

    async def AppGetOrCreate(self, stream):
        request: api_pb2.AppGetOrCreateRequest = await stream.recv_message()
        self.requests.append(request)

        environment_name = self.get_environment(request.environment_name)
        try:
            app_id = self.deployed_apps[(environment_name, request.app_name)]
        except KeyError:
            if request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
                app_id = f"ap-{self.n_apps}"
                self.deployed_apps[(environment_name, request.app_name)] = app_id
                self.app_state_history[app_id].append(api_pb2.APP_STATE_DEPLOYED)
                self.app_environments[app_id] = environment_name
            else:
                raise GRPCError(Status.NOT_FOUND, f"App '{request.app_name}' not found")

        await stream.send_message(api_pb2.AppGetOrCreateResponse(app_id=app_id))

    async def AppClientDisconnect(self, stream):
        request: api_pb2.AppClientDisconnectRequest = await stream.recv_message()
        self.requests.append(request)
        self.done = True
        self.app_client_disconnect_count += 1
        state_history = self.app_state_history[request.app_id]
        if state_history[-1] not in [api_pb2.APP_STATE_DETACHED, api_pb2.APP_STATE_DEPLOYED]:
            state_history.append(api_pb2.APP_STATE_STOPPED)
        await stream.send_message(Empty())
        # introduce jitter to simulate network latency
        await asyncio.sleep(random.uniform(0.0, self.resp_jitter_secs))

    async def AppGetLayout(self, stream):
        request: api_pb2.AppGetLayoutRequest = await stream.recv_message()
        app_layout = self.app_get_layout(request.app_id)
        await stream.send_message(api_pb2.AppGetLayoutResponse(app_layout=app_layout))

    async def AppGetLogs(self, stream):
        request: api_pb2.AppGetLogsRequest = await stream.recv_message()
        if not request.last_entry_id:
            # Just count initial requests
            self.app_get_logs_initial_count += 1
            last_entry_id = "1"
        else:
            last_entry_id = str(int(request.last_entry_id) + 1)
        for _ in range(50):
            await asyncio.sleep(0.5)
            log = api_pb2.TaskLogs(
                data=f"hello, world ({last_entry_id})\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT
            )
            await stream.send_message(api_pb2.TaskLogsBatch(entry_id=last_entry_id, items=[log]))
            last_entry_id = str(int(last_entry_id) + 1)
            if self.done:
                await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))
                return

    async def AppRollback(self, stream):
        request: api_pb2.AppRollbackRequest = await stream.recv_message()
        current_version = self.app_deployment_history[request.app_id][-1]["version"]
        if request.version < 0:
            rollback_version = current_version + request.version
        else:
            rollback_version = request.version
        rollback_client = self.app_deployment_history[request.app_id][rollback_version - 1]["client_version"]
        self.app_deployment_history[request.app_id].append(
            {
                "app_id": request.app_id,
                "deployed_at": datetime.datetime.now().timestamp(),
                "version": current_version + 1,
                "client_version": rollback_client,
                "deployed_by": "foo-user",
                "tag": "latest",
                "rollback_version": rollback_version,
            }
        )

        self.app_state_history[request.app_id].append(api_pb2.APP_STATE_DEPLOYED)
        await stream.send_message(Empty())

    async def AppPublish(self, stream):
        request: api_pb2.AppPublishRequest = await stream.recv_message()
        for key, val in request.definition_ids.items():
            assert key.startswith("fu-")
            assert val.startswith("de-")
        # TODO(michael) add some other assertions once we make the mock server represent real RPCs more accurately
        self.app_publish_count += 1
        self.app_objects[request.app_id] = {**request.function_ids, **request.class_ids}
        self.app_state_history[request.app_id].append(request.app_state)
        if request.app_state == api_pb2.AppState.APP_STATE_DEPLOYED:
            app_key = (self.app_environments[request.app_id], request.name)
            self.deployed_apps[app_key] = request.app_id
            await stream.send_message(api_pb2.AppPublishResponse(url="http://test.modal.com/foo/bar"))
        else:
            await stream.send_message(api_pb2.AppPublishResponse())

        if current_history := self.app_deployment_history[request.app_id]:
            current_version = current_history[-1]["version"]
        else:
            current_version = 0
        self.app_deployment_history[request.app_id].append(
            {
                "app_id": request.app_id,
                "deployed_at": datetime.datetime.now().timestamp(),
                "version": current_version + 1,
                "client_version": str(pkg_resources.parse_version(__version__)),
                "deployed_by": "foo-user",
                "tag": "latest",
                "rollback_version": None,
                "commit_info": request.commit_info,
            }
        )

    async def AppGetByDeploymentName(self, stream):
        request: api_pb2.AppGetByDeploymentNameRequest = await stream.recv_message()
        app_key = (self.get_environment(request.environment_name), request.name)
        app_id = self.deployed_apps.get(app_key)
        await stream.send_message(api_pb2.AppGetByDeploymentNameResponse(app_id=app_id))

    async def AppHeartbeat(self, stream):
        request: api_pb2.AppHeartbeatRequest = await stream.recv_message()
        self.requests.append(request)
        if self.app_state_history[request.app_id][-1] == api_pb2.APP_STATE_STOPPED:
            raise GRPCError(Status.FAILED_PRECONDITION, "App is stopped")
        else:
            self.app_heartbeats[request.app_id] += 1
            await stream.send_message(Empty())

    async def AppDeploymentHistory(self, stream):
        request: api_pb2.AppHeartbeatRequest = await stream.recv_message()
        app_deployment_histories = []

        for app_deployment_history in self.app_deployment_history.get(request.app_id, []):
            app_deployment_histories.append(
                api_pb2.AppDeploymentHistory(
                    app_id=request.app_id,
                    deployed_at=app_deployment_history["deployed_at"],
                    version=app_deployment_history["version"],
                    client_version=app_deployment_history["client_version"],
                    deployed_by=app_deployment_history["deployed_by"],
                    tag=app_deployment_history["tag"],
                    commit_info=app_deployment_history.get("commit_info", None),
                )
            )

        await stream.send_message(
            api_pb2.AppDeploymentHistoryResponse(app_deployment_histories=app_deployment_histories)
        )

    async def AppList(self, stream):
        req = await stream.recv_message()
        apps = []
        requested_environment = self.get_environment(req.environment_name)
        for (environment_name, app_name), app_id in self.deployed_apps.items():
            if environment_name != requested_environment:
                continue
            apps.append(
                api_pb2.AppListResponse.AppListItem(
                    name=app_name,
                    description=app_name,
                    app_id=app_id,
                    state=api_pb2.APP_STATE_DEPLOYED,
                )
            )
        await stream.send_message(api_pb2.AppListResponse(apps=apps))

    async def AppStop(self, stream):
        request: api_pb2.AppStopRequest = await stream.recv_message()
        self.deployed_apps = {k: v for k, v in self.deployed_apps.items() if v != request.app_id}
        await stream.send_message(Empty())

    async def AuthTokenGet(self, stream):
        response = api_pb2.AuthTokenGetResponse(token=self.auth_token)
        self.auth_tokens_generated += 1
        await stream.send_message(response)

    ### Checkpoint

    async def ContainerCheckpoint(self, stream):
        request: api_pb2.ContainerCheckpointRequest = await stream.recv_message()
        self.requests.append(request)
        self.container_snapshot_requests += 1
        await stream.send_message(Empty())

    async def ContainerExecPutInput(self, stream):
        request = await stream.recv_message()

        self.container_exec.stdin.write(request.input.message)
        await self.container_exec.stdin.drain()

        if request.input.eof:
            self.container_exec.stdin.close()

        await stream.send_message(Empty())

    ### Blob

    async def BlobCreate(self, stream):
        req = await stream.recv_message()
        # This is used to test retry_transient_errors, see grpc_utils_test.py
        self.blob_create_metadata = stream.metadata
        if len(self.fail_blob_create) > 0:
            status_code = self.fail_blob_create.pop()
            raise GRPCError(status_code, "foobar")
        elif req.content_length > self.blob_multipart_threshold:
            blob_id = await self.next_blob_id()
            num_parts = (req.content_length + self.blob_multipart_threshold - 1) // self.blob_multipart_threshold
            upload_urls = []
            for part_number in range(num_parts):
                upload_url = f"{self.blob_host}/upload?blob_id={blob_id}&part_number={part_number}"
                upload_urls.append(upload_url)

            multipart = api_pb2.MultiPartUpload(
                part_length=self.blob_multipart_threshold,
                upload_urls=upload_urls,
                completion_url=f"{self.blob_host}/complete_multipart?blob_id={blob_id}",
            )
            await stream.send_message(
                api_pb2.BlobCreateResponse(
                    blob_ids=[blob_id, blob_id],
                    multiparts=api_pb2.MultiPartUploadList(items=[multipart, multipart]),
                )
            )
        else:
            blob_id = await self.next_blob_id()
            upload_url = f"{self.blob_host}/upload?blob_id={blob_id}"
            await stream.send_message(
                api_pb2.BlobCreateResponse(
                    blob_ids=[blob_id, blob_id],
                    upload_urls=api_pb2.UploadUrlList(items=[upload_url, upload_url]),
                )
            )

    async def next_blob_id(self):
        self.n_blobs += 1
        blob_id = f"bl-{self.n_blobs}"
        return blob_id

    def next_entry_id(self) -> str:
        entry_id = f"1738286390000-{self.n_entry_ids}"
        self.n_entry_ids += 1
        return entry_id

    async def BlobGet(self, stream):
        request: api_pb2.BlobGetRequest = await stream.recv_message()
        download_url = f"{self.blob_host}/download?blob_id={request.blob_id}"
        await stream.send_message(api_pb2.BlobGetResponse(download_url=download_url))

    ### Class

    async def ClassCreate(self, stream):
        request: api_pb2.ClassCreateRequest = await stream.recv_message()
        assert request.app_id
        class_id = "cs-" + str(len(self.classes))
        self.classes.append(class_id)
        await stream.send_message(api_pb2.ClassCreateResponse(class_id=class_id))

    async def ClassGet(self, stream):
        request: api_pb2.ClassGetRequest = await stream.recv_message()
        app_key = (self.get_environment(request.environment_name), request.app_name)
        if not (app_id := self.deployed_apps.get(app_key)):
            raise GRPCError(Status.NOT_FOUND, f"can't find app {request.app_name}")
        app_objects = self.app_objects[app_id]
        object_id = app_objects.get(request.object_tag)
        if object_id is None:
            raise GRPCError(Status.NOT_FOUND, f"can't find object {request.object_tag}")
        await stream.send_message(api_pb2.ClassGetResponse(class_id=object_id))

    ### Client

    async def ClientHello(self, stream):
        request: Empty = await stream.recv_message()
        self.requests.append(request)
        if stream.metadata["x-modal-client-version"] == "deprecated":
            warnings = [
                api_pb2.Warning(
                    type=api_pb2.Warning.WARNING_TYPE_CLIENT_DEPRECATION,
                    message="SUPER OLD",
                )
            ]
        else:
            warnings = []
        resp = api_pb2.ClientHelloResponse(server_warnings=warnings)
        await stream.send_message(resp)

    # Container

    async def ContainerHeartbeat(self, stream):
        request: api_pb2.ContainerHeartbeatRequest = await stream.recv_message()
        self.requests.append(request)
        # Return earlier than the usual 15-second heartbeat to avoid suspending tests.
        await asyncify(self.container_heartbeat_abort.wait)(5)
        if self.container_heartbeat_response:
            await stream.send_message(self.container_heartbeat_response)
            self.container_heartbeat_response = None
        else:
            await stream.send_message(api_pb2.ContainerHeartbeatResponse())

    async def ContainerExec(self, stream):
        request: api_pb2.ContainerExecRequest = await stream.recv_message()
        self.container_exec = await asyncio.subprocess.create_subprocess_exec(
            *request.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
        )
        await stream.send_message(api_pb2.ContainerExecResponse(exec_id="container_exec_id"))

    async def ContainerExecWait(self, stream):
        request: api_pb2.ContainerExecWaitRequest = await stream.recv_message()
        try:
            await asyncio.wait_for(self.container_exec.wait(), request.timeout)
        except asyncio.TimeoutError:
            pass

        if self.container_exec.returncode is None:
            await stream.send_message(api_pb2.ContainerExecWaitResponse(completed=False))
        else:
            await stream.send_message(
                api_pb2.ContainerExecWaitResponse(completed=True, exit_code=self.container_exec.returncode)
            )

    async def ContainerExecGetOutput(self, stream):
        request: api_pb2.ContainerExecGetOutputRequest = await stream.recv_message()
        if request.file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            if self.shell_prompt:
                await stream.send_message(
                    api_pb2.RuntimeOutputBatch(
                        items=[
                            api_pb2.RuntimeOutputMessage(
                                file_descriptor=request.file_descriptor,
                                message_bytes=self.shell_prompt,
                            )
                        ]
                    )
                )
            read_stream = self.container_exec.stdout
        else:
            read_stream = self.container_exec.stderr

        async for message in read_stream:
            await stream.send_message(
                api_pb2.RuntimeOutputBatch(
                    items=[
                        api_pb2.RuntimeOutputMessage(
                            message=message.decode("utf-8"),
                            file_descriptor=request.file_descriptor,
                            message_bytes=message,
                        )
                    ]
                )
            )

        await stream.send_message(api_pb2.RuntimeOutputBatch(exit_code=0))

    async def ContainerHello(self, stream):
        await stream.recv_message()
        await stream.send_message(Empty())

    ### Dict

    async def DictGetOrCreate(self, stream):
        request: api_pb2.DictGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.environment_name)
        if k in self.deployed_dicts:
            dict_id = self.deployed_dicts[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
            dict_id = f"di-{len(self.dicts)}"
            self.dicts[dict_id] = {entry.key: entry.value for entry in request.data}
            self.deployed_dicts[k] = dict_id
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL:
            dict_id = f"di-{len(self.dicts)}"
            self.dicts[dict_id] = {entry.key: entry.value for entry in request.data}
        else:
            raise GRPCError(Status.NOT_FOUND, f"Dict {k} not found")
        creation_info = api_pb2.CreationInfo(created_by=self.default_username)
        metadata = api_pb2.DictMetadata(name=request.deployment_name, creation_info=creation_info)
        await stream.send_message(api_pb2.DictGetOrCreateResponse(dict_id=dict_id, metadata=metadata))

    async def DictHeartbeat(self, stream):
        await stream.recv_message()
        self.n_dict_heartbeats += 1
        await stream.send_message(Empty())

    async def DictDelete(self, stream):
        request: api_pb2.DictDeleteRequest = await stream.recv_message()
        self.deployed_dicts = {k: v for k, v in self.deployed_dicts.items() if v != request.dict_id}
        await stream.send_message(Empty())

    async def DictClear(self, stream):
        request: api_pb2.DictGetRequest = await stream.recv_message()
        self.dicts[request.dict_id] = {}
        await stream.send_message(Empty())

    async def DictGet(self, stream):
        request: api_pb2.DictGetRequest = await stream.recv_message()
        d = self.dicts[request.dict_id]
        await stream.send_message(api_pb2.DictGetResponse(value=d.get(request.key), found=bool(request.key in d)))

    async def DictLen(self, stream):
        request: api_pb2.DictLenRequest = await stream.recv_message()
        await stream.send_message(api_pb2.DictLenResponse(len=len(self.dicts[request.dict_id])))

    async def DictList(self, stream):
        dicts = [api_pb2.DictListResponse.DictInfo(name=name, created_at=1) for name, _ in self.deployed_dicts]
        await stream.send_message(api_pb2.DictListResponse(dicts=dicts))

    async def DictUpdate(self, stream):
        request: api_pb2.DictUpdateRequest = await stream.recv_message()
        if len(request.updates) == 1:
            if request.if_not_exists and request.updates[0].key in self.dicts[request.dict_id]:
                await stream.send_message(api_pb2.DictUpdateResponse(created=False))
                return
        else:
            if request.if_not_exists:
                raise GRPCError(Status.INVALID_ARGUMENT)

        for update in request.updates:
            self.dicts[request.dict_id][update.key] = update.value
        await stream.send_message(api_pb2.DictUpdateResponse(created=True))

    async def DictContents(self, stream):
        request: api_pb2.DictGetRequest = await stream.recv_message()
        for k, v in self.dicts[request.dict_id].items():
            await stream.send_message(api_pb2.DictEntry(key=k, value=v))

    ### Environment

    async def EnvironmentCreate(self, stream):
        await stream.send_message(Empty())

    async def EnvironmentUpdate(self, stream):
        await stream.send_message(api_pb2.EnvironmentListItem())

    async def EnvironmentGetOrCreate(self, stream):
        request: api_pb2.EnvironmentGetOrCreateRequest = await stream.recv_message()
        name = request.deployment_name
        if name in self.environments:
            environment_id = self.environments[name]
        else:
            environment_id = f"en-{len(self.environments) + 1}"
            self.environments[name] = environment_id
        image_builder_version = max(get_args(ImageBuilderVersion))
        settings = api_pb2.EnvironmentSettings(image_builder_version=image_builder_version)
        metadata = api_pb2.EnvironmentMetadata(name=name, settings=settings)
        await stream.send_message(
            api_pb2.EnvironmentGetOrCreateResponse(environment_id=environment_id, metadata=metadata)
        )

    ### Function

    async def FunctionBindParams(self, stream):
        request: api_pb2.FunctionBindParamsRequest = await stream.recv_message()
        assert request.function_id
        assert request.serialized_params
        base_function = self.app_functions[request.function_id]
        existing_func_id = self.bound_functions.get((request.function_id, request.serialized_params), None)
        if existing_func_id:
            function_id = existing_func_id
        else:
            self.n_functions += 1
            function_id = f"fu-{self.n_functions}"
            assert not base_function.use_method_name

            bound_func = api_pb2.Function()
            bound_func.CopyFrom(base_function)
            self.app_functions[function_id] = bound_func
            self.bound_functions[(request.function_id, request.serialized_params)] = function_id
            self.function_params[function_id] = deserialize_params(request.serialized_params, bound_func, None)
            self.function_options[function_id] = request.function_options

        await stream.send_message(
            api_pb2.FunctionBindParamsResponse(
                bound_function_id=function_id,
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    function_name=base_function.function_name,
                    function_type=base_function.function_type,
                    web_url=base_function.web_url,
                    use_function_id=function_id,
                    use_method_name="",
                ),
            )
        )

    @contextlib.contextmanager
    def input_lockstep(self) -> Iterator[threading.Barrier]:
        self.get_inputs_barrier = threading.Barrier(2, timeout=10)
        yield self.get_inputs_barrier
        self.get_inputs_barrier = threading.Barrier(1)

    @contextlib.contextmanager
    def output_lockstep(self) -> Iterator[threading.Barrier]:
        self.put_outputs_barrier = threading.Barrier(2, timeout=10)
        yield self.put_outputs_barrier
        self.put_outputs_barrier = threading.Barrier(1)

    async def FunctionGetDynamicConcurrency(self, stream):
        await stream.send_message(api_pb2.FunctionGetDynamicConcurrencyResponse(concurrency=5))

    async def FunctionGetInputs(self, stream):
        await asyncio.get_running_loop().run_in_executor(None, self.get_inputs_barrier.wait)
        request: api_pb2.FunctionGetInputsRequest = await stream.recv_message()
        assert request.function_id
        if self.fail_get_inputs:
            raise GRPCError(Status.INTERNAL)
        elif self.rate_limit_sleep_duration is not None:
            s = self.rate_limit_sleep_duration
            self.rate_limit_sleep_duration = None
            await stream.send_message(api_pb2.FunctionGetInputsResponse(rate_limit_sleep_duration=s))
        elif not self.container_inputs:
            await asyncio.sleep(10.0)
            await stream.send_message(api_pb2.FunctionGetInputsResponse(inputs=[]))
        else:
            await stream.send_message(self.container_inputs.pop(0))

    async def FunctionPutOutputs(self, stream):
        await asyncio.get_running_loop().run_in_executor(None, self.put_outputs_barrier.wait)
        request: api_pb2.FunctionPutOutputsRequest = await stream.recv_message()
        self.container_outputs.append(request)
        await stream.send_message(Empty())

    async def FunctionPrecreate(self, stream):
        req: api_pb2.FunctionPrecreateRequest = await stream.recv_message()
        if not req.existing_function_id:
            self.n_functions += 1
            function_id = f"fu-{self.n_functions}"
        else:
            function_id = req.existing_function_id

        self.precreated_functions.add(function_id)

        web_url = "http://xyz.internal" if req.HasField("webhook_config") and req.webhook_config.type else None

        # This loop is for class service functions, where req.method_definitions will be non-empty
        method_handle_metadata: dict[str, api_pb2.FunctionHandleMetadata] = {}
        for method_name, method_definition in req.method_definitions.items():
            method_web_url = (
                f"http://{method_name}.internal"
                if method_definition.HasField("webhook_config") and method_definition.webhook_config.type
                else None
            )
            method_handle_metadata[method_name] = api_pb2.FunctionHandleMetadata(
                function_name=method_definition.function_name,
                function_type=method_definition.function_type,
                web_url=method_web_url,
                function_schema=method_definition.function_schema,
            )
        await stream.send_message(
            api_pb2.FunctionPrecreateResponse(
                function_id=function_id,
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    function_name=req.function_name,
                    function_type=req.function_type,
                    web_url=web_url,
                    use_function_id=req.use_function_id or function_id,
                    use_method_name=req.use_method_name,
                    method_handle_metadata=method_handle_metadata,
                    function_schema=req.function_schema,
                ),
            )
        )

    async def FunctionCreate(self, stream):
        request: api_pb2.FunctionCreateRequest = await stream.recv_message()
        if self.function_create_error:
            raise self.function_create_error
        if request.function.resources.gpu_config.count > 0:
            if request.function.resources.gpu_config.gpu_type not in VALID_GPU_TYPES:
                raise GRPCError(Status.INVALID_ARGUMENT, "Not a valid GPU type")
        if request.function.cloud_provider_str:
            if request.function.cloud_provider_str.upper() not in VALID_CLOUD_PROVIDERS:
                raise GRPCError(Status.INVALID_ARGUMENT, "Not a valid cloud provider")

        if request.existing_function_id:
            function_id = request.existing_function_id
        else:
            self.n_functions += 1
            function_id = f"fu-{self.n_functions}"
        function: api_pb2.Function | None = None
        function_data: api_pb2.FunctionData | None = None
        if len(request.function_data.ranked_functions) > 0:
            function_data = api_pb2.FunctionData()
            function_data.CopyFrom(request.function_data)
        else:
            assert request.function
            function = api_pb2.Function()
            function.CopyFrom(request.function)

        assert (function is None) != (function_data is None)
        function_defn: Union[api_pb2.Function, api_pb2.FunctionData] = function or function_data
        assert function_defn
        if function_defn.webhook_config.type:
            function_defn.web_url = "http://xyz.internal"
        for method_name, method_definition in function_defn.method_definitions.items():
            if method_definition.webhook_config.type:
                method_definition.web_url = f"http://{method_name}.internal"
        self.app_functions[function_id] = function_defn

        if function_defn.schedule:
            self.function2schedule[function_id] = function_defn.schedule

        warnings = []
        if int(function_defn.experimental_options.get("warn_me", "0")):
            warnings.append(api_pb2.Warning(message="You have been warned!"))

        await stream.send_message(
            api_pb2.FunctionCreateResponse(
                function_id=function_id,
                function=function,
                # TODO: use self.get_function_metadata here
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    function_name=function_defn.function_name,
                    function_type=function_defn.function_type,
                    web_url=function_defn.web_url,
                    use_function_id=function_defn.use_function_id or function_id,
                    use_method_name=function_defn.use_method_name,
                    definition_id=f"de-{self.n_functions}",
                    method_handle_metadata={
                        method_name: api_pb2.FunctionHandleMetadata(
                            function_name=method_definition.function_name,
                            function_type=method_definition.function_type,
                            web_url=method_definition.web_url,
                            is_method=True,
                            use_method_name=method_name,
                            function_schema=method_definition.function_schema,
                        )
                        for method_name, method_definition in function_defn.method_definitions.items()
                    },
                    class_parameter_info=function_defn.class_parameter_info,
                    function_schema=function_defn.function_schema,
                    input_plane_url=self._get_input_plane_url(function_defn),
                    input_plane_region=self._get_input_plane_region(function_defn),
                    max_object_size_bytes=self.max_object_size_bytes,
                ),
                server_warnings=warnings,
            )
        )

    async def FunctionGet(self, stream):
        request: api_pb2.FunctionGetRequest = await stream.recv_message()
        app_key = (self.get_environment(request.environment_name), request.app_name)
        if not (app_id := self.deployed_apps.get(app_key)):
            raise GRPCError(Status.NOT_FOUND, f"can't find app {request.app_name}")

        app_objects = self.app_objects[app_id]
        object_id = app_objects.get(request.object_tag)
        if object_id is None:
            raise GRPCError(Status.NOT_FOUND, f"can't find object {request.object_tag}")
        function_metadata = self.get_function_metadata(object_id)
        await stream.send_message(
            api_pb2.FunctionGetResponse(
                function_id=object_id,
                handle_metadata=function_metadata,
                server_warnings=self.function_get_server_warnings,
            )
        )

    async def FunctionMap(self, stream):
        self.fcidx += 1
        request: api_pb2.FunctionMapRequest = await stream.recv_message()
        function_call_id = f"fc-{self.fcidx}"
        self.function_id_for_function_call[function_call_id] = request.function_id
        fn_definition = self.app_functions.get(request.function_id)
        retry_policy = None
        if fn_definition and hasattr(fn_definition, "retry_policy"):
            retry_policy = fn_definition.retry_policy
        function_call_jwt = encode_function_call_jwt(request.function_id, function_call_id)

        response_inputs = []
        for input_item in request.pipelined_inputs:
            input_id = f"in-{self.n_inputs}"
            self.n_inputs += 1
            self.add_function_call_input(function_call_id, input_item, input_id, 0)
            response_inputs.append(
                api_pb2.FunctionPutInputsResponseItem(
                    idx=self.fcidx,
                    input_id=input_id,
                    input_jwt=encode_input_jwt(self.fcidx, input_id, function_call_id, self.next_entry_id(), 0),
                )
            )

        await stream.send_message(
            api_pb2.FunctionMapResponse(
                function_call_id=function_call_id,
                retry_policy=retry_policy,
                function_call_jwt=function_call_jwt,
                pipelined_inputs=response_inputs,
                sync_client_retries_enabled=self.sync_client_retries_enabled,
                max_inputs_outstanding=1000,
            )
        )

    async def FunctionRetryInputs(self, stream):
        request: api_pb2.FunctionRetryInputsRequest = await stream.recv_message()
        function_id, function_call_id = decode_function_call_jwt(request.function_call_jwt)
        function_call_inputs = self.function_call_inputs.setdefault(function_call_id, [])
        input_jwts = []
        for item in request.inputs:
            if item.input.WhichOneof("args_oneof") == "args":
                args, kwargs = deserialize(item.input.args, None)
            else:
                args, kwargs = deserialize(self.blobs[item.input.args_blob_id], None)
            self.n_inputs += 1
            idx, input_id, function_call_id, _, _ = decode_input_jwt(item.input_jwt)
            input_jwts.append(encode_input_jwt(idx, input_id, function_call_id, self.next_entry_id(), item.retry_count))
            function_call_inputs.append(((idx, input_id, item.retry_count), (args, kwargs)))
            self.function_call_inputs_update_event.set()
        await stream.send_message(api_pb2.FunctionRetryInputsResponse(input_jwts=input_jwts))

    async def FunctionPutInputs(self, stream):
        if self.fail_put_inputs_with_grpc_error and self.fail_put_inputs_with_grpc_error.count > 0:
            self.fail_put_inputs_with_grpc_error.count = self.fail_put_inputs_with_grpc_error.count - 1
            raise GRPCError(self.fail_put_inputs_with_grpc_error.grpc_error)
        if self.fail_put_inputs_with_stream_terminated_error > 0:
            self.fail_put_inputs_with_stream_terminated_error = self.fail_put_inputs_with_stream_terminated_error - 1
            await stream.cancel()
        request: api_pb2.FunctionPutInputsRequest = await stream.recv_message()
        response_items = []

        for item in request.inputs:
            input_id = f"in-{self.n_inputs}"
            self.n_inputs += 1
            response_items.append(
                api_pb2.FunctionPutInputsResponseItem(
                    input_id=input_id,
                    idx=item.idx,
                    input_jwt=encode_input_jwt(item.idx, input_id, request.function_call_id, self.next_entry_id(), 0),
                )
            )
            self.add_function_call_input(request.function_call_id, item, input_id, 0)
        if self.slow_put_inputs:
            await asyncio.sleep(0.001)
        await stream.send_message(api_pb2.FunctionPutInputsResponse(inputs=response_items))

    def add_function_call_input(self, function_call_id, item: api_pb2.FunctionPutInputsItem, input_id, retry_count):
        if item.input.WhichOneof("args_oneof") == "args":
            args, kwargs = deserialize(item.input.args, None)
        else:
            args, kwargs = deserialize(self.blobs[item.input.args_blob_id], None)
        function_call_inputs = self.function_call_inputs.setdefault(function_call_id, [])
        function_call_inputs.append(((item.idx, input_id, retry_count), (args, kwargs)))
        self.function_call_inputs_update_event.set()

    async def FunctionGetOutputs(self, stream):
        request: api_pb2.FunctionGetOutputsRequest = await stream.recv_message()
        if request.clear_on_success:
            self.cleared_function_calls.add(request.function_call_id)

        fc_inputs = self.function_call_inputs.setdefault(request.function_call_id, [])
        if fc_inputs and not self.function_is_running:
            popidx = len(fc_inputs) // 2  # simulate that results don't always come in order
            (idx, input_id, retry_count), (args, kwargs) = fc_inputs.pop(popidx)
            output_exc = None
            try:
                res = self._function_body(*args, **kwargs)

                if inspect.iscoroutine(res):
                    result = await res
                    result_data_format = api_pb2.DATA_FORMAT_PICKLE
                elif inspect.isgenerator(res):
                    count = 0
                    for item in res:
                        count += 1
                        await self.fc_data_out[request.function_call_id].put(
                            api_pb2.DataChunk(
                                data_format=api_pb2.DATA_FORMAT_PICKLE,
                                data=serialize_data_format(item, api_pb2.DATA_FORMAT_PICKLE),
                                index=count,
                            )
                        )
                    result = api_pb2.GeneratorDone(items_total=count)
                    result_data_format = api_pb2.DATA_FORMAT_GENERATOR_DONE
                else:
                    result = res
                    result_data_format = api_pb2.DATA_FORMAT_PICKLE
            except Exception as exc:
                serialized_exc = cloudpickle.dumps(exc)
                result = api_pb2.GenericResult(
                    status=self.failure_status,
                    data=serialized_exc,
                    exception=repr(exc),
                    traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                )
                output_exc = api_pb2.FunctionGetOutputsItem(
                    input_id=input_id,
                    idx=idx,
                    result=result,
                    data_format=api_pb2.DATA_FORMAT_PICKLE,
                    retry_count=retry_count,
                )

            if output_exc:
                output = output_exc
            else:
                serialized_data = serialize_data_format(result, result_data_format)
                if self.use_blob_outputs:
                    blob_id = await self.next_blob_id()
                    self.blobs[blob_id] = serialized_data
                    data_kwargs = {
                        "data_blob_id": blob_id,
                    }
                else:
                    data_kwargs = {"data": serialized_data}
                output = api_pb2.FunctionGetOutputsItem(
                    input_id=input_id,
                    idx=idx,
                    result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS, **data_kwargs),
                    data_format=result_data_format,
                    retry_count=retry_count,
                )

            await stream.send_message(api_pb2.FunctionGetOutputsResponse(outputs=[output]))
        else:
            # wait for there to be at least one input, since that will allow a subsequent call to
            # get the associated output using the above branch
            if len(fc_inputs):
                await stream.send_message(
                    api_pb2.FunctionGetOutputsResponse(outputs=[], num_unfinished_inputs=len(fc_inputs))
                )
            else:
                try:
                    await asyncio.wait_for(self.function_call_inputs_update_event.wait(), timeout=request.timeout)
                except asyncio.TimeoutError:
                    pass
                self.function_call_inputs_update_event.clear()
                await stream.send_message(
                    api_pb2.FunctionGetOutputsResponse(outputs=[], num_unfinished_inputs=len(fc_inputs))
                )

    async def FunctionGetSerialized(self, stream):
        await stream.send_message(
            api_pb2.FunctionGetSerializedResponse(
                function_serialized=self.function_serialized,
                class_serialized=self.class_serialized,
            )
        )

    async def FunctionCallCancel(self, stream):
        req = await stream.recv_message()
        self.cancelled_calls.append(req.function_call_id)
        await stream.send_message(Empty())

    async def FunctionCallGetDataIn(self, stream):
        req: api_pb2.FunctionCallGetDataRequest = await stream.recv_message()
        while True:
            chunk = await self.fc_data_in[req.function_call_id].get()
            await stream.send_message(chunk)

    async def FunctionCallGetDataOut(self, stream):
        req: api_pb2.FunctionCallGetDataRequest = await stream.recv_message()

        if len(self.fail_get_data_out) > 0:
            status_code = self.fail_get_data_out.pop()
            raise GRPCError(status_code, "foobar")

        while True:
            chunk = await self.fc_data_out[req.function_call_id].get()
            await stream.send_message(chunk)

    async def FunctionCallPutDataOut(self, stream):
        req: api_pb2.FunctionCallPutDataRequest = await stream.recv_message()
        for chunk in req.data_chunks:
            await self.fc_data_out[req.function_call_id].put(chunk)
        await stream.send_message(Empty())

    async def FunctionUpdateSchedulingParams(self, stream):
        req: api_pb2.FunctionUpdateSchedulingParamsRequest = await stream.recv_message()

        # update function definition
        fn_definition = self.app_functions[req.function_id]
        assert isinstance(fn_definition, api_pb2.Function)
        # Note that this doesn't mock the full server logic very well
        fn_definition.warm_pool_size = req.warm_pool_size_override
        fn_definition.autoscaler_settings.MergeFrom(req.settings)

        # Hacky that we're modifying the function definition directly
        # In the server we track autoscaler updates separately
        fn_definition.warm_pool_size = fn_definition.autoscaler_settings.min_containers
        fn_definition.concurrency_limit = fn_definition.autoscaler_settings.max_containers
        fn_definition._experimental_buffer_containers = fn_definition.autoscaler_settings.buffer_containers
        fn_definition.task_idle_timeout_secs = fn_definition.autoscaler_settings.scaledown_window

        await stream.send_message(api_pb2.FunctionUpdateSchedulingParamsResponse())

    ### Image

    async def ImageGetOrCreate(self, stream):
        request: api_pb2.ImageGetOrCreateRequest = await stream.recv_message()
        for image_id, image in self.images.items():
            if request.image.SerializeToString() == image.SerializeToString():
                await stream.send_message(
                    api_pb2.ImageGetOrCreateResponse(
                        image_id=image_id,
                        metadata=api_pb2.ImageMetadata(image_builder_version=self.image_builder_versions[image_id]),
                    )
                )
                return
        idx = len(self.images) + 1
        image_id = f"im-{idx}"

        self.images[image_id] = request.image
        self.image_build_function_ids[image_id] = request.build_function_id
        self.image_builder_versions[image_id] = request.builder_version
        if request.force_build:
            self.force_built_images.append(image_id)
        await stream.send_message(
            api_pb2.ImageGetOrCreateResponse(
                image_id=image_id,
                metadata=api_pb2.ImageMetadata(image_builder_version=request.builder_version),
            )
        )

    async def ImageJoinStreaming(self, stream):
        req = await stream.recv_message()

        if self.image_join_sleep_duration is not None:
            await asyncio.sleep(self.image_join_sleep_duration)

        task_log_1 = api_pb2.TaskLogs(data="build starting\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_INFO)
        task_log_2 = api_pb2.TaskLogs(
            task_progress=api_pb2.TaskProgress(
                len=1, pos=0, progress_type=api_pb2.IMAGE_SNAPSHOT_UPLOAD, description="xyz"
            )
        )
        task_log_3 = api_pb2.TaskLogs(data="build finished\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_INFO)
        await stream.send_message(api_pb2.ImageJoinStreamingResponse(task_logs=[task_log_1, task_log_2, task_log_3]))
        await stream.send_message(
            api_pb2.ImageJoinStreamingResponse(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS),
                metadata=api_pb2.ImageMetadata(
                    image_builder_version=self.image_builder_versions.get(req.image_id),
                ),
            )
        )

    ### Mount

    async def MountPutFile(self, stream):
        request: api_pb2.MountPutFileRequest = await stream.recv_message()
        if request.WhichOneof("data_oneof") is not None:
            if request.data.startswith(b"large"):
                # Useful for simulating a slow upload, e.g. to test our checks for mid-deploy modifications
                await asyncio.sleep(2)
            self.files_sha2data[request.sha256_hex] = {"data": request.data, "data_blob_id": request.data_blob_id}
            self.n_mount_files += 1
            await stream.send_message(api_pb2.MountPutFileResponse(exists=True))
        else:
            await stream.send_message(api_pb2.MountPutFileResponse(exists=False))

    async def MountGetOrCreate(self, stream):
        request: api_pb2.MountGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.namespace)
        if request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED:
            if k not in self.deployed_mounts:
                raise GRPCError(Status.NOT_FOUND, f"Mount {k} not found")
            mount_id = self.deployed_mounts[k]
            await stream.send_message(
                api_pb2.MountGetOrCreateResponse(
                    mount_id=mount_id,
                    handle_metadata=api_pb2.MountHandleMetadata(content_checksum_sha256_hex="deadbeef"),
                )
            )
            return
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS:
            self.n_mounts += 1
            mount_id = f"mo-{self.n_mounts}"
            self.deployed_mounts[k] = mount_id
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP:
            self.n_mounts += 1
            mount_id = f"mo-{self.n_mounts}"
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL:
            self.n_mounts += 1
            mount_id = f"mo-{self.n_mounts}"

        else:
            raise Exception("unsupported creation type")

        mount_content = self.mount_contents[mount_id] = {}
        for file in request.files:
            mount_content[file.filename] = self.files_name2sha[file.filename] = file.sha256_hex

        await stream.send_message(
            api_pb2.MountGetOrCreateResponse(
                mount_id=mount_id, handle_metadata=api_pb2.MountHandleMetadata(content_checksum_sha256_hex="deadbeef")
            )
        )

    ### Proxy

    async def ProxyGetOrCreate(self, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.ProxyGetOrCreateResponse(proxy_id="pr-123"))

    async def ProxyGet(self, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.ProxyGetResponse(proxy=api_pb2.Proxy(proxy_id="pr-123")))

    ### Queue

    async def QueueClear(self, stream):
        request: api_pb2.QueueClearRequest = await stream.recv_message()
        if request.all_partitions:
            self.queue = {b"": []}
        else:
            if request.partition_key in self.queue:
                self.queue[request.partition_key] = []
        await stream.send_message(Empty())

    async def QueueGetOrCreate(self, stream):
        request: api_pb2.QueueGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.environment_name)
        if k in self.deployed_queues:
            queue_id = self.deployed_queues[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
            self.n_queues += 1
            queue_id = f"qu-{self.n_queues}"
            self.deployed_queues[k] = queue_id
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL:
            self.n_queues += 1
            queue_id = f"qu-{self.n_queues}"
        else:
            raise GRPCError(Status.NOT_FOUND, f"Queue {k} not found")
        creation_info = api_pb2.CreationInfo(created_by=self.default_username)
        metadata = api_pb2.QueueMetadata(name=request.deployment_name, creation_info=creation_info)
        await stream.send_message(api_pb2.QueueGetOrCreateResponse(queue_id=queue_id, metadata=metadata))

    async def QueueDelete(self, stream):
        request: api_pb2.QueueDeleteRequest = await stream.recv_message()
        self.deployed_queues = {k: v for k, v in self.deployed_queues.items() if v != request.queue_id}
        await stream.send_message(Empty())

    async def QueueHeartbeat(self, stream):
        await stream.recv_message()
        self.n_queue_heartbeats += 1
        await stream.send_message(Empty())

    async def QueuePut(self, stream):
        request: api_pb2.QueuePutRequest = await stream.recv_message()
        if sum(map(len, self.queue.values())) >= self.queue_max_len:
            raise GRPCError(Status.RESOURCE_EXHAUSTED, f"Hit servicer's max len for Queues: {self.queue_max_len}")
        q = self.queue.setdefault(request.partition_key, [])
        q += request.values
        await stream.send_message(Empty())

    async def QueueGet(self, stream):
        request: api_pb2.QueueGetRequest = await stream.recv_message()
        q = self.queue.get(request.partition_key, [])
        if len(q) > 0:
            values = [q.pop(0)]
        else:
            values = []
            await asyncio.sleep(request.timeout)
        await stream.send_message(api_pb2.QueueGetResponse(values=values))

    async def QueueLen(self, stream):
        request = await stream.recv_message()
        if request.total:
            value = sum(map(len, self.queue.values()))
        else:
            q = self.queue.get(request.partition_key, [])
            value = len(q)
        await stream.send_message(api_pb2.QueueLenResponse(len=value))

    async def QueueList(self, stream):
        # TODO Note that the actual self.queue holding the data assumes we have a single queue
        # So there is a mismatch and I am not implementing a mock for the num_partitions / total_size
        queues = [api_pb2.QueueListResponse.QueueInfo(name=name, created_at=1) for name, _ in self.deployed_queues]
        await stream.send_message(api_pb2.QueueListResponse(queues=queues))

    async def QueueNextItems(self, stream):
        request: api_pb2.QueueNextItemsRequest = await stream.recv_message()
        next_item_idx = int(request.last_entry_id) + 1 if request.last_entry_id else 0
        q = self.queue.get(request.partition_key, [])
        if next_item_idx < len(q):
            item = api_pb2.QueueItem(value=q[next_item_idx], entry_id=f"{next_item_idx}")
            await stream.send_message(api_pb2.QueueNextItemsResponse(items=[item]))
        else:
            if request.item_poll_timeout > 0:
                await asyncio.sleep(0.1)
            await stream.send_message(api_pb2.QueueNextItemsResponse(items=[]))

    ### Sandbox

    async def SandboxCreate(self, stream):
        request: api_pb2.SandboxCreateRequest = await stream.recv_message()
        self.sandbox = await asyncio.subprocess.create_subprocess_exec(
            *(request.definition.entrypoint_args or ["sleep", f"{48 * 3600}"]),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
        )

        self.sandbox_app_id = request.app_id
        self.sandbox_defs.append(request.definition)

        await stream.send_message(api_pb2.SandboxCreateResponse(sandbox_id="sb-123"))

    async def SandboxGetLogs(self, stream):
        request: api_pb2.SandboxGetLogsRequest = await stream.recv_message()
        f: asyncio.StreamReader
        if request.file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            # Blocking read until EOF is returned.
            f = self.sandbox.stdout
        else:
            f = self.sandbox.stderr

        async for message in f:
            await stream.send_message(
                api_pb2.TaskLogsBatch(
                    items=[api_pb2.TaskLogs(data=message.decode("utf-8"), file_descriptor=request.file_descriptor)]
                )
            )

        await stream.send_message(api_pb2.TaskLogsBatch(eof=True))

    async def SandboxWait(self, stream):
        request: api_pb2.SandboxWaitRequest = await stream.recv_message()
        try:
            await asyncio.wait_for(self.sandbox.wait(), request.timeout)
        except asyncio.TimeoutError:
            pass

        if self.sandbox.returncode is None:
            # This happens when request.timeout is 0 and the sandbox hasn't completed.
            await stream.send_message(api_pb2.SandboxWaitResponse())
            return
        elif self.sandbox.returncode != 0:
            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE, exitcode=self.sandbox.returncode
            )
        else:
            result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
        self.sandbox_result = result
        await stream.send_message(api_pb2.SandboxWaitResponse(result=result))

    async def SandboxList(self, stream):
        request: api_pb2.SandboxListRequest = await stream.recv_message()
        if self.sandbox.returncode or request.before_timestamp == 1:
            await stream.send_message(api_pb2.SandboxListResponse(sandboxes=[]))
            return

        if request.app_id and request.app_id != self.sandbox_app_id:
            await stream.send_message(api_pb2.SandboxListResponse(sandboxes=[]))
            return

        for tag in request.tags:
            if self.sandbox_tags.get(tag.tag_name) != tag.tag_value:
                await stream.send_message(api_pb2.SandboxListResponse(sandboxes=[]))
                return

        await stream.send_message(
            api_pb2.SandboxListResponse(
                sandboxes=[
                    api_pb2.SandboxInfo(
                        id="sb-123", created_at=1, task_info=api_pb2.TaskInfo(result=self.sandbox_result)
                    )
                ]
            )
        )

    async def SandboxTagsSet(self, stream):
        request: api_pb2.SandboxTagsSetRequest = await stream.recv_message()
        self.sandbox_tags = {tag.tag_name: tag.tag_value for tag in request.tags}
        await stream.send_message(Empty())

    async def SandboxTerminate(self, stream):
        try:
            self.sandbox.terminate()
        except ProcessLookupError:
            pass
        await stream.send_message(api_pb2.SandboxTerminateResponse())

    async def SandboxSnapshot(self, stream):
        _request: api_pb2.SandboxSnapshotRequest = await stream.recv_message()
        await stream.send_message(api_pb2.SandboxSnapshotResponse(snapshot_id="sn-123"))

    async def SandboxSnapshotFs(self, stream):
        _request: api_pb2.SandboxSnapshotFsRequest = await stream.recv_message()
        await stream.send_message(
            api_pb2.SandboxSnapshotFsResponse(
                image_id="im-123",
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS),
            )
        )

    async def SandboxSnapshotWait(self, stream):
        _request: api_pb2.SandboxSnapshotWaitRequest = await stream.recv_message()
        await stream.send_message(
            api_pb2.SandboxSnapshotWaitResponse(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
            )
        )

    async def SandboxRestore(self, stream):
        _request: api_pb2.SandboxRestoreRequest = await stream.recv_message()
        await stream.send_message(api_pb2.SandboxRestoreResponse(sandbox_id="sb-123"))

    async def SandboxGetTaskId(self, stream):
        # only used for `modal shell` / `modal container exec`
        _request: api_pb2.SandboxGetTaskIdRequest = await stream.recv_message()
        await stream.send_message(api_pb2.SandboxGetTaskIdResponse(task_id="modal_container_exec"))

    async def SandboxStdinWrite(self, stream):
        request: api_pb2.SandboxStdinWriteRequest = await stream.recv_message()

        if self.sandbox.returncode is not None:
            raise GRPCError(Status.FAILED_PRECONDITION, "Sandbox has already terminated")

        self.sandbox.stdin.write(request.input)
        await self.sandbox.stdin.drain()

        if request.eof:
            self.sandbox.stdin.close()
        await stream.send_message(api_pb2.SandboxStdinWriteResponse())

    ### Secret

    async def SecretDelete(self, stream):
        request: api_pb2.SecretDeleteRequest = await stream.recv_message()
        self.deployed_secrets = {k: v for k, v in self.deployed_secrets.items() if v != request.secret_id}
        self.secrets.pop(request.secret_id)
        await stream.send_message(Empty())

    async def SecretGetOrCreate(self, stream):
        request: api_pb2.SecretGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.environment_name)
        if request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP:
            secret_id = "st-" + str(len(self.secrets))
            self.secrets[secret_id] = request.env_dict
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL:
            secret_id = "st-" + str(len(self.secrets))
            self.secrets[secret_id] = request.env_dict
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS:
            if k in self.deployed_secrets:
                raise GRPCError(Status.ALREADY_EXISTS, f"Secret {k} already exists")
            secret_id = None
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_OVERWRITE_IF_EXISTS:
            secret_id = None
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED:
            if k not in self.deployed_secrets:
                raise GRPCError(Status.NOT_FOUND, f"Secret {k} not found")
            secret_id = self.deployed_secrets[k]
        else:
            raise Exception("unsupported creation type")

        if secret_id is None:  # Create one
            secret_id = "st-" + str(len(self.secrets))
            self.secrets[secret_id] = request.env_dict
            self.deployed_secrets[k] = secret_id

        creation_info = api_pb2.CreationInfo(created_by=self.default_username)
        metadata = api_pb2.SecretMetadata(name=request.deployment_name, creation_info=creation_info)
        await stream.send_message(api_pb2.SecretGetOrCreateResponse(secret_id=secret_id, metadata=metadata))

    async def SecretList(self, stream):
        await stream.recv_message()
        # Note: being lazy and not implementing the env filtering
        items = [api_pb2.SecretListItem(label=name) for name, env in self.deployed_secrets]
        await stream.send_message(api_pb2.SecretListResponse(items=items))

    ### Snapshot

    async def SandboxSnapshotGet(self, stream):
        _request: api_pb2.SandboxSnapshotGetRequest = await stream.recv_message()
        await stream.send_message(api_pb2.SandboxSnapshotGetResponse(snapshot_id="sn-123"))

    ### Network File System (née Shared volume)

    async def SharedVolumeDelete(self, stream):
        req: api_pb2.SharedVolumeDeleteRequest = await stream.recv_message()
        self.nfs_files.pop(req.shared_volume_id)
        self.deployed_nfss = {k: vol_id for k, vol_id in self.deployed_nfss.items() if vol_id != req.shared_volume_id}
        await stream.send_message(Empty())

    async def SharedVolumeGetOrCreate(self, stream):
        request: api_pb2.SharedVolumeGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.environment_name)
        if request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED:
            if k not in self.deployed_nfss:
                if k in self.deployed_volumes:
                    raise GRPCError(Status.NOT_FOUND, "App has wrong entity vo")
                raise GRPCError(Status.NOT_FOUND, f"NFS {k} not found")
            nfs_id = self.deployed_nfss[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL:
            nfs_id = f"sv-{len(self.nfs_files)}"
            self.nfs_files[nfs_id] = {}
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
            if k not in self.deployed_nfss:
                nfs_id = f"sv-{len(self.nfs_files)}"
                self.nfs_files[nfs_id] = {}
                self.deployed_nfss[k] = nfs_id
            nfs_id = self.deployed_nfss[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS:
            if k in self.deployed_nfss:
                raise GRPCError(Status.ALREADY_EXISTS, f"NFS {k} already exists")
            nfs_id = f"sv-{len(self.nfs_files)}"
            self.nfs_files[nfs_id] = {}
            self.deployed_nfss[k] = nfs_id
        else:
            raise GRPCError(Status.INVALID_ARGUMENT, "unsupported object creation type")

        await stream.send_message(api_pb2.SharedVolumeGetOrCreateResponse(shared_volume_id=nfs_id))

    async def SharedVolumeHeartbeat(self, stream):
        await stream.recv_message()
        self.n_nfs_heartbeats += 1
        await stream.send_message(Empty())

    async def SharedVolumeList(self, stream):
        req = await stream.recv_message()
        items = []
        for (name, env_name), volume_id in self.deployed_nfss.items():
            if env_name != req.environment_name:
                continue
            items.append(api_pb2.SharedVolumeListItem(label=name, shared_volume_id=volume_id, created_at=1))
        resp = api_pb2.SharedVolumeListResponse(items=items, environment_name=req.environment_name)
        await stream.send_message(resp)

    async def SharedVolumePutFile(self, stream):
        req = await stream.recv_message()
        self.nfs_files[req.shared_volume_id][req.path] = req
        await stream.send_message(api_pb2.SharedVolumePutFileResponse(exists=True))

    async def SharedVolumeGetFile(self, stream):
        req = await stream.recv_message()
        put_req = self.nfs_files.get(req.shared_volume_id, {}).get(req.path)
        if not put_req:
            raise GRPCError(Status.NOT_FOUND, f"No such file: {req.path}")
        if put_req.data_blob_id:
            await stream.send_message(api_pb2.SharedVolumeGetFileResponse(data_blob_id=put_req.data_blob_id))
        else:
            await stream.send_message(api_pb2.SharedVolumeGetFileResponse(data=put_req.data))

    async def SharedVolumeListFilesStream(self, stream):
        req: api_pb2.SharedVolumeListFilesRequest = await stream.recv_message()
        for path in self.nfs_files[req.shared_volume_id].keys():
            entry = api_pb2.FileEntry(path=path, type=api_pb2.FileEntry.FileType.FILE)
            response = api_pb2.SharedVolumeListFilesResponse(entries=[entry])
            if req.path == "**" or req.path == "/" or req.path == path:  # hack
                await stream.send_message(response)

    ### Task

    async def TaskCurrentInputs(self, stream: grpclib.server.Stream[Empty, api_pb2.TaskCurrentInputsResponse]) -> None:
        await stream.send_message(api_pb2.TaskCurrentInputsResponse(input_ids=[]))  # dummy implementation

    async def TaskResult(self, stream):
        request: api_pb2.TaskResultRequest = await stream.recv_message()
        if self.task_result is None:
            self.task_result = request.result
        await stream.send_message(Empty())

    ### Token flow

    async def TokenFlowCreate(self, stream):
        request: api_pb2.TokenFlowCreateRequest = await stream.recv_message()
        self.token_flow_localhost_port = request.localhost_port
        await stream.send_message(
            api_pb2.TokenFlowCreateResponse(token_flow_id="tc-123", web_url="https://localhost/xyz/abc")
        )

    async def TokenFlowWait(self, stream):
        await stream.send_message(
            api_pb2.TokenFlowWaitResponse(
                token_id="abc",
                token_secret="xyz",
            )
        )

    async def WorkspaceNameLookup(self, stream):
        await stream.send_message(api_pb2.WorkspaceNameLookupResponse(username="test-username"))

    ### Tunnel

    async def TunnelStart(self, stream):
        request: api_pb2.TunnelStartRequest = await stream.recv_message()
        port = request.port
        await stream.send_message(api_pb2.TunnelStartResponse(host=f"{port}.modal.test", port=443))

    async def TunnelStop(self, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.TunnelStopResponse(exists=True))

    ### Volume

    async def VolumeGetOrCreate(self, stream):
        request: api_pb2.VolumeGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.environment_name)
        if request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED:
            if k not in self.deployed_volumes:
                raise GRPCError(Status.NOT_FOUND, f"Volume {k} not found")
            volume_id = self.deployed_volumes[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL:
            volume_id = f"vo-{len(self.volumes)}"
            self.volumes[volume_id] = Volume(version=request.version, files={})
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
            if k not in self.deployed_volumes:
                volume_id = f"vo-{len(self.volumes)}"
                self.volumes[volume_id] = Volume(version=request.version, files={})
                self.deployed_volumes[k] = volume_id
            volume_id = self.deployed_volumes[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS:
            if k in self.deployed_volumes:
                raise GRPCError(Status.ALREADY_EXISTS, f"Volume {k} already exists")
            volume_id = f"vo-{len(self.volumes)}"
            self.volumes[volume_id] = Volume(version=request.version, files={})
            self.deployed_volumes[k] = volume_id
        else:
            raise GRPCError(Status.INVALID_ARGUMENT, "unsupported object creation type")

        creation_info = api_pb2.CreationInfo(created_by=self.default_username)
        metadata = api_pb2.VolumeMetadata(
            name=request.deployment_name, creation_info=creation_info, version=request.version
        )
        response = api_pb2.VolumeGetOrCreateResponse(volume_id=volume_id, version=request.version, metadata=metadata)
        await stream.send_message(response)

    async def VolumeList(self, stream):
        req = await stream.recv_message()
        items = []
        for (name, env_name), volume_id in self.deployed_volumes.items():
            if env_name != req.environment_name:
                continue
            items.append(api_pb2.VolumeListItem(label=name, volume_id=volume_id, created_at=1))
        resp = api_pb2.VolumeListResponse(items=items, environment_name=req.environment_name)
        await stream.send_message(resp)

    async def VolumeHeartbeat(self, stream):
        await stream.recv_message()
        self.n_vol_heartbeats += 1
        await stream.send_message(Empty())

    async def VolumeCommit(self, stream):
        req = await stream.recv_message()
        self.requests.append(req)
        if not req.volume_id.startswith("vo-"):
            raise GRPCError(Status.NOT_FOUND, f"invalid volume ID {req.volume_id}")
        self.volume_commits[req.volume_id] += 1
        await stream.send_message(api_pb2.VolumeCommitResponse(skip_reload=False))

    async def VolumeDelete(self, stream):
        req: api_pb2.VolumeDeleteRequest = await stream.recv_message()
        self.volumes.pop(req.volume_id)
        self.deployed_volumes = {k: vol_id for k, vol_id in self.deployed_volumes.items() if vol_id != req.volume_id}
        await stream.send_message(Empty())

    async def VolumeReload(self, stream):
        req = await stream.recv_message()
        self.requests.append(req)
        self.volume_reloads[req.volume_id] += 1
        await stream.send_message(Empty())

    async def VolumeGetFile(self, stream):
        req = await stream.recv_message()
        if req.path not in self.volumes[req.volume_id].files:
            raise GRPCError(Status.NOT_FOUND, "File not found")
        vol_file = self.volumes[req.volume_id].files[req.path]
        if vol_file.data_blob_id:
            await stream.send_message(api_pb2.VolumeGetFileResponse(data_blob_id=vol_file.data_blob_id))
        else:
            size = len(vol_file.data)
            if req.start or req.len:
                start = req.start
                len_ = req.len or len(vol_file.data)
                await stream.send_message(
                    api_pb2.VolumeGetFileResponse(data=vol_file.data[start : start + len_], size=size)
                )
            else:
                await stream.send_message(api_pb2.VolumeGetFileResponse(data=vol_file.data, size=size))

    async def VolumeGetFile2(self, stream):
        req = await stream.recv_message()
        if req.path not in self.volumes[req.volume_id].files:
            raise GRPCError(Status.NOT_FOUND, "File not found")
        vol_file = self.volumes[req.volume_id].files[req.path]
        get_urls = []

        def ceildiv(a: int, b: int) -> int:
            return -(a // -b)

        if vol_file.data_blob_id:
            file_size = len(self.blobs[vol_file.data_blob_id])
        else:
            file_size = len(vol_file.data)

        total_start = req.start
        total_end = req.start + (req.len or file_size)

        def slice_block(block_idx: int):
            # Crude port of internal `blocks_for_byte_slice` algorithm:
            slice_start = total_start % BLOCK_SIZE if block_idx == 0 else 0
            slice_end = (
                (((total_end - 1) % BLOCK_SIZE) + 1 if total_end > 0 else 0)
                if block_idx == (block_end - block_start - 1)
                else BLOCK_SIZE
            )
            return slice_start, slice_end

        # Check if file was created using VolumePutFiles instead of VolumePutFiles2, which is supported since
        # VolumePutFiles2 should support volumefs1 volumes now.
        if vol_file.block_hashes is None:
            block_start = min(total_start, file_size) // BLOCK_SIZE
            block_end = ceildiv(min(total_end, file_size), BLOCK_SIZE)
            assert vol_file.data_sha256_hex is not None

            for idx in range(block_start, block_end):
                start, end = slice_block(idx)
                length = end - start
                get_urls.append(
                    f"{self.blob_host}/block/test-get-request:v1:{vol_file.data_sha256_hex}:{idx}:{start}:{length}"
                )
        else:
            block_start = min(total_start // BLOCK_SIZE, len(vol_file.block_hashes))
            block_end = min(ceildiv(total_end, BLOCK_SIZE), len(vol_file.block_hashes))
            assert vol_file.data_blob_id is None

            for idx, block_hash in enumerate(vol_file.block_hashes[block_start:block_end]):
                start, end = slice_block(idx)
                length = end - start
                get_urls.append(f"{self.blob_host}/block/test-get-request:v2:{block_hash.hex()}:{start}:{length}")

        response = api_pb2.VolumeGetFile2Response(
            get_urls=get_urls, size=len(vol_file.data), start=total_start, len=total_end - total_start
        )
        await stream.send_message(response)

    async def VolumeRemoveFile(self, stream):
        req = await stream.recv_message()
        self._volume_remove_file(req)
        await stream.send_message(Empty())

    async def VolumeRemoveFile2(self, stream):
        req = await stream.recv_message()
        self._volume_remove_file(req)
        await stream.send_message(Empty())

    def _volume_remove_file(self, req: Union[api_pb2.VolumeRemoveFileRequest, api_pb2.VolumeRemoveFile2Request]):
        if req.path not in self.volumes[req.volume_id].files:
            raise GRPCError(Status.NOT_FOUND, "File not found")
        del self.volumes[req.volume_id].files[req.path]

    async def VolumeRename(self, stream):
        req = await stream.recv_message()
        for key, vol_id in self.deployed_volumes.items():
            if vol_id == req.volume_id:
                break
        self.deployed_volumes[(req.name, *key[1:])] = self.deployed_volumes.pop(key)
        await stream.send_message(Empty())

    async def VolumeListFiles(self, stream):
        req = await stream.recv_message()
        await self._volume_list_files(stream, req, lambda entries: api_pb2.VolumeListFilesResponse(entries=entries))

    async def VolumeListFiles2(self, stream):
        req = await stream.recv_message()
        await self._volume_list_files(stream, req, lambda entries: api_pb2.VolumeListFiles2Response(entries=entries))

    async def _volume_list_files(
        self,
        stream,
        req: Union[api_pb2.VolumeListFilesRequest, api_pb2.VolumeListFiles2Request],
        make_resp: Callable[
            [list[api_pb2.FileEntry]], Union[api_pb2.VolumeListFilesResponse, api_pb2.VolumeListFiles2Response]
        ],
    ):
        path = req.path if req.path else "/"
        if path.startswith("/"):
            path = path[1:]
        if path.endswith("/"):
            path = path[:-1]

        found_file = False  # empty directory detection is not handled here!
        for k, vol_file in self.volumes[req.volume_id].files.items():
            if not path or k == path or (k.startswith(path + "/") and (req.recursive or "/" not in k[len(path) + 1 :])):
                entry = api_pb2.FileEntry(path=k, type=api_pb2.FileEntry.FileType.FILE, size=len(vol_file.data))
                await stream.send_message(make_resp([entry]))
                found_file = True

        if path and not found_file:
            raise GRPCError(Status.NOT_FOUND, "No such file")

    async def VolumePutFiles(self, stream):
        req = await stream.recv_message()
        for file in req.files:
            blob_data = self.files_sha2data[file.sha256_hex]

            if file.filename in self.volumes[req.volume_id].files and req.disallow_overwrite_existing_files:
                raise GRPCError(
                    Status.ALREADY_EXISTS,
                    (
                        f"{file.filename}: already exists "
                        f"(disallow_overwrite_existing_files={req.disallow_overwrite_existing_files}"
                    ),
                )

            self.volumes[req.volume_id].files[file.filename] = VolumeFile(
                data=blob_data["data"],
                data_blob_id=blob_data["data_blob_id"],
                data_sha256_hex=file.sha256_hex,
                mode=file.mode,
            )
        await stream.send_message(Empty())

    async def VolumePutFiles2(self, stream):
        req = await stream.recv_message()
        missing_blocks = []
        files_to_create = {}

        for file_index, file in enumerate(req.files):
            if file.path in self.volumes[req.volume_id].files and req.disallow_overwrite_existing_files:
                raise GRPCError(
                    Status.ALREADY_EXISTS,
                    (
                        f"{file.path}: already exists "
                        f"(disallow_overwrite_existing_files={req.disallow_overwrite_existing_files}"
                    ),
                )

            blocks = []
            block_hashes = []
            file_missing_blocks = []

            for block_index, block in enumerate(file.blocks):
                block_hashes.append(block.contents_sha256)
                actual_block_id = block.contents_sha256.hex()
                block_data = self.blocks.get(actual_block_id)

                # TODO(dflemstr): here, we assume that all blocks that the user uploads are always new; we could check
                #  if the block blob already exists, and not generate a MissingBlock for those block blobs, but then it
                #  would get trickier to validate the put_url response loop here...
                put_response = block.put_response
                if put_response:
                    prefix = b"test-put-response:"
                    expected_block_id = put_response[len(prefix) :].decode("utf-8")
                    valid_put_response = put_response.startswith(prefix) and expected_block_id == actual_block_id
                else:
                    valid_put_response = False

                if block_data is not None and valid_put_response:
                    # If this is not the last block, it needs to have size BLOCK_SIZE
                    if block_index + 1 < len(file.blocks):
                        assert len(block_data) == BLOCK_SIZE
                    # If this is the last block, it must be at most BLOCK_SIZE
                    if block_index + 1 == len(file.blocks):
                        assert len(block_data) <= BLOCK_SIZE
                    blocks.append(block_data)
                else:
                    missing_block = api_pb2.VolumePutFiles2Response.MissingBlock(
                        file_index=file_index,
                        block_index=block_index,
                        put_url=f"{self.blob_host}/block/test-put-request",
                    )
                    file_missing_blocks.append(missing_block)

            if file_missing_blocks:
                missing_blocks.extend(file_missing_blocks)
            else:
                files_to_create[file.path] = VolumeFile(
                    data=b"".join(blocks),
                    data_blob_id=None,
                    mode=file.mode,
                    block_hashes=block_hashes,
                )

        if not missing_blocks:
            self.volumes[req.volume_id].files.update(files_to_create)

        await stream.send_message(api_pb2.VolumePutFiles2Response(missing_blocks=missing_blocks))

    async def VolumeCopyFiles(self, stream):
        req = await stream.recv_message()
        self._copy_files(req)
        await stream.send_message(Empty())

    async def VolumeCopyFiles2(self, stream):
        req = await stream.recv_message()
        self._copy_files(req)
        await stream.send_message(Empty())

    def _copy_files(self, req):
        for src_path in req.src_paths:
            if src_path not in self.volumes[req.volume_id].files:
                raise GRPCError(Status.NOT_FOUND, f"Source file not found: {src_path}")
            src_file = self.volumes[req.volume_id].files[src_path]
            if len(req.src_paths) > 1:
                # check to make sure dst is a directory
                if req.dst_path.endswith(("/", "\\")) or not os.path.splitext(os.path.basename(req.dst_path))[1]:
                    dst_path = os.path.join(req.dst_path, os.path.basename(src_path))
                else:
                    raise GRPCError(Status.INVALID_ARGUMENT, f"{dst_path} is not a directory.")
            else:
                dst_path = req.dst_path
            self.volumes[req.volume_id].files[dst_path] = src_file

    async def AttemptStart(self, stream):
        request: api_pb2.AttemptStartRequest = await stream.recv_message()
        fn_definition = self.app_functions.get(request.function_id)
        retry_policy = fn_definition.retry_policy if fn_definition else None
        # TODO(ryan): implement attempt token logic
        await stream.send_message(
            api_pb2.AttemptStartResponse(attempt_token="bogus_attempt_token", retry_policy=retry_policy)
        )

    async def AttemptAwait(self, stream):
        # TODO(ryan): Eventually we may want to invoke the user's function and return a result.
        # For now we just return a dummy response

        # To test client retries, tests can configure outputs to fail some number of times.
        status = api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        if self.attempt_await_failures_remaining > 0:
            status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE
            self.attempt_await_failures_remaining = self.attempt_await_failures_remaining - 1

        await stream.send_message(
            api_pb2.AttemptAwaitResponse(
                output=api_pb2.FunctionGetOutputsItem(
                    input_id="in-1",
                    idx=0,
                    result=api_pb2.GenericResult(
                        status=status,
                        data=serialize_data_format("attempt_await_bogus_response", api_pb2.DATA_FORMAT_PICKLE),
                    ),
                    data_format=api_pb2.DATA_FORMAT_PICKLE,
                    retry_count=0,
                )
            )
        )

    async def AttemptRetry(self, stream):
        await stream.send_message(api_pb2.AttemptRetryResponse(attempt_token="bogus_retry_token"))

    async def MapStartOrContinue(self, stream):
        request: api_pb2.MapStartOrContinueRequest = await stream.recv_message()
        # print(f"MapStartOrContinue: request: {request}")

        # If function_call_id is provided, this is a continue request, otherwise it's a start
        if request.function_call_id:
            function_call_id = request.function_call_id
        else:
            self.fcidx += 1
            function_call_id = f"fc-{self.fcidx}"
            self.function_id_for_function_call[function_call_id] = request.function_id

        # Process inputs and store them for MapAwait to pick up later
        attempt_tokens = []
        for index, item in enumerate(request.items):
            retry_count = 0
            if item.attempt_token != "":
                retry_count = int(item.attempt_token.split(":")[1]) + 1
            attempt_tokens.append(f"bogus-attempt-token-{item.input.idx}:{retry_count}")
            # Store inputs for MapAwait to process
            input_id = f"in-{self.n_inputs}"
            self.n_inputs += 1
            self.add_function_call_input(function_call_id, item.input, input_id, retry_count)

        # Get retry policy from function definition if available
        fn_definition = self.app_functions.get(request.function_id)
        retry_policy = fn_definition.retry_policy if fn_definition else None

        response = api_pb2.MapStartOrContinueResponse(
            function_id=request.function_id,
            function_call_id=function_call_id,
            max_inputs_outstanding=1000,
            attempt_tokens=attempt_tokens,
            retry_policy=retry_policy,
        )

        # Inhereted variable name from python code PutInputs
        if self.slow_put_inputs:
            await asyncio.sleep(0.001)
        # print(f"MapStartOrContinue: response: {response}")
        await stream.send_message(response)

    async def MapAwait(self, stream):
        request: api_pb2.MapAwaitRequest = await stream.recv_message()

        # Check if we have any function call inputs for this function call
        fc_inputs = self.function_call_inputs.setdefault(request.function_call_id, [])
        outputs = []

        if fc_inputs and not self.function_is_running:
            popidx = len(fc_inputs) // 2  # simulate that results don't always come in order
            (idx, input_id, retry_count), (args, kwargs) = fc_inputs.pop(popidx)
            output_exc = None
            try:
                res = self._function_body(*args, **kwargs)

                if inspect.iscoroutine(res):
                    result = await res
                    result_data_format = api_pb2.DATA_FORMAT_PICKLE
                elif inspect.isgenerator(res):
                    count = 0
                    for item in res:
                        count += 1
                        await self.fc_data_out[request.function_call_id].put(
                            api_pb2.DataChunk(
                                data_format=api_pb2.DATA_FORMAT_PICKLE,
                                data=serialize_data_format(item, api_pb2.DATA_FORMAT_PICKLE),
                                index=count,
                            )
                        )
                    result = api_pb2.GeneratorDone(items_total=count)
                    result_data_format = api_pb2.DATA_FORMAT_GENERATOR_DONE
                else:
                    result = res
                    result_data_format = api_pb2.DATA_FORMAT_PICKLE
            except Exception as exc:
                serialized_exc = cloudpickle.dumps(exc)
                result = api_pb2.GenericResult(
                    status=self.failure_status,
                    data=serialized_exc,
                    exception=repr(exc),
                    traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                )
                output_exc = api_pb2.FunctionGetOutputsItem(
                    input_id=input_id,
                    idx=idx,
                    result=result,
                    data_format=api_pb2.DATA_FORMAT_PICKLE,
                    retry_count=retry_count,
                )

            if output_exc:
                outputs.append(output_exc)
            else:
                serialized_data = serialize_data_format(result, result_data_format)
                if self.use_blob_outputs:
                    blob_id = await self.next_blob_id()
                    self.blobs[blob_id] = serialized_data
                    data_kwargs = {
                        "data_blob_id": blob_id,
                    }
                else:
                    data_kwargs = {"data": serialized_data}
                outputs.append(
                    api_pb2.FunctionGetOutputsItem(
                        input_id=input_id,
                        idx=idx,
                        result=api_pb2.GenericResult(
                            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS, **data_kwargs
                        ),
                        data_format=result_data_format,
                        retry_count=retry_count,
                    )
                )

        else:
            # wait for there to be at least one input, since that will allow a subsequent call to
            # get the associated output using the above branch
            if not len(fc_inputs):
                try:
                    await asyncio.wait_for(self.function_call_inputs_update_event.wait(), timeout=request.timeout)
                except asyncio.TimeoutError:
                    pass
                self.function_call_inputs_update_event.clear()

        response = api_pb2.MapAwaitResponse(outputs=outputs, last_entry_id=str(len(outputs)))
        await stream.send_message(response)

    async def MapCheckInputs(self, stream):
        request: api_pb2.MapCheckInputsRequest = await stream.recv_message()

        # For testing purposes, assume no inputs are lost unless specifically configured
        # Return False (not lost) for all attempt tokens
        lost = [False] * len(request.attempt_tokens)

        await stream.send_message(api_pb2.MapCheckInputsResponse(lost=lost))


@pytest.fixture
def blob_server():
    blobs = {}
    blob_parts: dict[str, dict[int, bytes]] = defaultdict(dict)
    blocks = {}
    files_sha2data: dict[str, dict] = {}

    async def upload(request):
        blob_id = request.query["blob_id"]
        content = await request.content.read()
        if content == b"FAILURE":
            return aiohttp.web.Response(status=500)
        content_md5 = hashlib.md5(content).hexdigest()
        etag = f'"{content_md5}"'
        if "part_number" in request.query:
            part_number = int(request.query["part_number"])
            blob_parts[blob_id][part_number] = content
        else:
            blobs[blob_id] = content
        return aiohttp.web.Response(text="Hello, world", headers={"ETag": etag})

    async def complete_multipart(request):
        blob_id = request.query["blob_id"]
        blob_nums = range(min(blob_parts[blob_id].keys()), max(blob_parts[blob_id].keys()) + 1)
        content = b""
        part_hashes = b""
        for num in blob_nums:
            part_content = blob_parts[blob_id][num]
            content += part_content
            part_hashes += hashlib.md5(part_content).digest()

        content_md5 = hashlib.md5(part_hashes).hexdigest()
        etag = f'"{content_md5}-{len(blob_parts[blob_id])}"'
        blobs[blob_id] = content
        return aiohttp.web.Response(text=f"<etag>{etag}</etag>")

    async def download(request):
        blob_id = request.query["blob_id"]
        if blob_id == "bl-failure":
            return aiohttp.web.Response(status=500)
        return aiohttp.web.Response(body=blobs[blob_id])

    async def put_block(request):
        token = request.match_info["token"]
        if token != "test-put-request":
            return aiohttp.web.Response(status=400, text="bad token")

        content = await request.content.read()
        if content == b"FAILURE":
            return aiohttp.web.Response(status=500, text="simulated server error")

        if len(content) > BLOCK_SIZE:
            return aiohttp.web.Response(status=413, text="block too big")

        block_id = hashlib.sha256(content).hexdigest()
        blocks[block_id] = content
        return aiohttp.web.Response(text=f"test-put-response:{block_id}")

    async def get_block(request):
        token = request.match_info["token"]

        magic, version, *rest = token.split(":")
        if magic != "test-get-request":
            return aiohttp.web.Response(status=400, text="bad token")

        if version == "v1":
            file_sha256_hex, block_idx, start, length = rest
            start = BLOCK_SIZE * int(block_idx) + int(start)
            length = int(length)
            file_data = files_sha2data[file_sha256_hex]
            blob_id = file_data["data_blob_id"]
            if blob_id:
                body = blobs[blob_id][start : start + length]
            else:
                body = file_data["data"][start : start + length]
        elif version == "v2":
            block_id, start, length = rest
            start = int(start)
            length = int(length)
            body = blocks[block_id][start : start + length]
        else:
            return aiohttp.web.Response(status=404)

        return aiohttp.web.Response(body=body)

    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.put("/upload", upload)])
    app.add_routes([aiohttp.web.get("/download", download)])
    app.add_routes([aiohttp.web.post("/complete_multipart", complete_multipart)])

    # API used for volume version 2 blocks:
    app.add_routes([aiohttp.web.get("/block/{token}", get_block)])
    app.add_routes([aiohttp.web.put("/block/{token}", put_block)])

    started = threading.Event()
    stop_server = threading.Event()

    host = None

    def run_server_other_thread():
        loop = asyncio.new_event_loop()

        async def async_main():
            nonlocal host
            async with run_temporary_http_server(app) as _host:
                host = _host
                started.set()
                await loop.run_in_executor(None, stop_server.wait)

        loop.run_until_complete(async_main())

        # clean up event loop
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

    # run server on separate thread to not lock up the server event loop in case of blocking calls in tests
    thread = threading.Thread(target=run_server_other_thread)
    thread.start()
    started.wait()
    yield host, blobs, blocks, files_sha2data
    stop_server.set()
    thread.join()


@pytest_asyncio.fixture(scope="function")
def temporary_sock_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield os.path.join(tmpdirname, "servicer.sock")


@contextlib.asynccontextmanager
async def run_server(servicer, host=None, port=None, path=None):
    server = None

    async def _start_servicer():
        nonlocal server
        server = grpclib.server.Server([servicer])
        listen(server, RecvRequest, servicer.recv_request)
        await server.start(host=host, port=port, path=path)

    async def _stop_servicer():
        servicer.container_heartbeat_abort.set()
        server.close()
        # This is the proper way to close down the asyncio server,
        # but it causes our tests to hang on 3.12+ because client connections
        # for clients created through _Client.from_env don't get closed until
        # asyncio event loop shutdown. Commenting out but perhaps revisit if we
        # refactor the way that _Client cleanup happens.
        # await server.wait_closed()

    start_servicer = synchronize_api(_start_servicer)
    stop_servicer = synchronize_api(_stop_servicer)

    await start_servicer.aio()
    try:
        yield
    finally:
        await stop_servicer.aio()


@pytest.fixture(scope="function")
def credentials():
    token_id = "ak-" + str(uuid.uuid4())
    token_secret = "as-" + str(uuid.uuid4())
    return (token_id, token_secret)


@pytest_asyncio.fixture(scope="function")
async def servicer(blob_server, temporary_sock_path, credentials):
    port = find_free_port()

    blob_host, blobs, blocks, files_sha2data = blob_server
    servicer = MockClientServicer(blob_host, blobs, blocks, files_sha2data, credentials, port)  # type: ignore

    if platform.system() != "Windows":
        async with run_server(servicer, host="0.0.0.0", port=port):
            async with run_server(servicer, path=temporary_sock_path):
                servicer.client_addr = f"http://127.0.0.1:{port}"
                servicer.container_addr = f"unix://{temporary_sock_path}"
                yield servicer
    else:
        # Use a regular TCP socket for the container connection
        container_port = find_free_port()
        async with run_server(servicer, host="0.0.0.0", port=port):
            async with run_server(servicer, host="0.0.0.0", port=container_port):
                servicer.client_addr = f"http://127.0.0.1:{port}"
                servicer.container_addr = f"http://127.0.0.1:{container_port}"
                yield servicer


@pytest_asyncio.fixture(scope="function")
async def client(servicer, credentials):
    with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def container_client(servicer):
    async with Client(servicer.container_addr, api_pb2.CLIENT_TYPE_CONTAINER, None) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def server_url_env(servicer, monkeypatch):
    monkeypatch.setenv("MODAL_SERVER_URL", servicer.client_addr)
    yield


@pytest_asyncio.fixture(scope="function")
async def token_env(servicer, monkeypatch, credentials):
    token_id, token_secret = credentials
    monkeypatch.setenv("MODAL_TOKEN_ID", token_id)
    monkeypatch.setenv("MODAL_TOKEN_SECRET", token_secret)
    yield


@pytest_asyncio.fixture(scope="function")
async def container_env(servicer, monkeypatch):
    monkeypatch.setenv("MODAL_SERVER_URL", servicer.container_addr)
    monkeypatch.setenv("MODAL_TASK_ID", "ta-123")
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    yield


@pytest_asyncio.fixture(scope="function", autouse=True)
async def reset_default_client():
    Client.set_env_client(None)


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
            yield root_dir
        finally:
            os.chdir(cwd)
            shutil.rmtree(root_dir, ignore_errors=True)

    return mock_dir


@pytest.fixture(autouse=True)
def reset_container_app():
    try:
        yield
    finally:
        _ContainerIOManager._reset_singleton()
        _App._reset_container_app()


@pytest.fixture
def repo_root(request):
    return Path(request.config.rootdir)


@pytest.fixture(scope="module")
def test_dir(request):
    """Absolute path to directory containing test file."""
    return Path(__file__).parent


@pytest.fixture(scope="function")
def modal_config():
    """Return a context manager with a temporary modal.toml file"""

    @contextlib.contextmanager
    def mock_modal_toml(contents: str = "", show_on_error: bool = False):
        # Some of the cli tests run within within the main process
        # so we need to modify the config singletons to pick up any changes
        orig_config_path_env = os.environ.get("MODAL_CONFIG_PATH")
        orig_config_path = config.user_config_path
        orig_profile = config._profile
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".toml", mode="w") as t:
                t.write(textwrap.dedent(contents.strip("\n")))
            os.environ["MODAL_CONFIG_PATH"] = t.name
            config.user_config_path = t.name
            config._user_config = config._read_user_config()
            config._profile = config._config_active_profile()
            yield t.name
        except Exception:
            if show_on_error:
                with open(t.name) as f:
                    print(f"Test config file contents:\n\n{f.read()}", file=sys.stderr)
            raise
        finally:
            if orig_config_path_env:
                os.environ["MODAL_CONFIG_PATH"] = orig_config_path_env
            else:
                del os.environ["MODAL_CONFIG_PATH"]
            config.user_config_path = orig_config_path
            config._user_config = config._read_user_config()
            config._profile = orig_profile
            os.remove(t.name)

    return mock_modal_toml


@pytest.fixture
def supports_dir(test_dir):
    return test_dir / Path("supports")


@pytest_asyncio.fixture
async def set_env_client(client):
    try:
        Client.set_env_client(client)
        yield
    finally:
        Client.set_env_client(None)


@pytest.fixture
def no_rich(monkeypatch):
    normal_import = __import__

    def import_fail_for_rich(name: str, *args, **kwargs) -> ModuleType:
        if name.startswith("rich"):
            raise ModuleNotFoundError("No module named 'rich'")
        else:
            return normal_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_fail_for_rich)
    yield


@pytest.fixture()
def supports_on_path(supports_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(supports_dir))


def encode_input_jwt(idx: int, input_id: str, function_call_id: str, entry_id: str, retry_count: int) -> str:
    """
    Creates fake input jwt token.
    """
    assert str(idx) and input_id and function_call_id and entry_id
    return DecodedJwt.encode_without_signature(
        {
            "idx": idx,
            "input_id": input_id,
            "function_call_id": function_call_id,
            "entry_id": entry_id,
            "retry_count": retry_count,
        }
    )


def decode_input_jwt(input_jwt: str) -> tuple[int, str, str, str, int]:
    """
    Decodes fake input jwt. Returns idx, input_id, function_call_id, entry_id, retry_count.
    """
    decoded = DecodedJwt.decode_without_verification(input_jwt)
    return (
        decoded.payload["idx"],
        decoded.payload["input_id"],
        decoded.payload["function_call_id"],
        decoded.payload["entry_id"],
        decoded.payload["retry_count"],
    )


def encode_function_call_jwt(function_id: str, function_call_id: str) -> str:
    """
    Creates fake function call jwt.
    """
    assert function_id and function_call_id
    return DecodedJwt.encode_without_signature({"function_id": function_id, "function_call_id": function_call_id})


def decode_function_call_jwt(function_call_jwt: str) -> tuple[str, str]:
    """
    Decodes fake function call jwt. Returns function_id, function_call_id.
    """
    decoded = DecodedJwt.decode_without_verification(function_call_jwt)
    return (decoded.payload["function_id"], decoded.payload["function_call_id"])


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        yield
