# Copyright Modal Labs 2024
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import inspect
import os
import pytest
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Optional

import aiohttp.web
import aiohttp.web_runner
import grpclib.server
import pkg_resources
import pytest_asyncio
from google.protobuf.empty_pb2 import Empty
from grpclib import GRPCError, Status

import modal._serialization
from modal import __version__, config
from modal._serialization import serialize_data_format
from modal._utils.async_utils import asyncify, synchronize_api
from modal._utils.grpc_testing import patch_mock_servicer
from modal._utils.grpc_utils import find_free_port
from modal._utils.http_utils import run_temporary_http_server
from modal._vendor import cloudpickle
from modal.app import _ContainerApp
from modal.client import Client
from modal.mount import client_mount_name
from modal_proto import api_grpc, api_pb2


@dataclasses.dataclass
class VolumeFile:
    data: bytes
    data_blob_id: str
    mode: int


@patch_mock_servicer
class MockClientServicer(api_grpc.ModalClientBase):
    # TODO(erikbern): add more annotations
    container_inputs: list[api_pb2.FunctionGetInputsResponse]
    container_outputs: list[api_pb2.FunctionPutOutputsRequest]
    fc_data_in: defaultdict[str, asyncio.Queue[api_pb2.DataChunk]]
    fc_data_out: defaultdict[str, asyncio.Queue[api_pb2.DataChunk]]

    def __init__(self, blob_host, blobs):
        self.put_outputs_barrier = threading.Barrier(
            1, timeout=10
        )  # set to non-1 to get lock-step of output pushing within a test
        self.get_inputs_barrier = threading.Barrier(
            1, timeout=10
        )  # set to non-1 to get lock-step of input releases within a test

        self.app_state_history = defaultdict(list)
        self.app_heartbeats: Dict[str, int] = defaultdict(int)
        self.container_checkpoint_requests = 0
        self.n_blobs = 0
        self.blob_host = blob_host
        self.blobs = blobs  # shared dict
        self.requests = []
        self.done = False
        self.rate_limit_sleep_duration = None
        self.fail_get_inputs = False
        self.slow_put_inputs = False
        self.container_inputs = []
        self.container_outputs = []
        self.fc_data_in = defaultdict(lambda: asyncio.Queue())  # unbounded
        self.fc_data_out = defaultdict(lambda: asyncio.Queue())  # unbounded
        self.queue = []
        self.deployed_apps = {
            client_mount_name(): "ap-x",
        }
        self.app_objects = {}
        self.app_single_objects = {}
        self.app_unindexed_objects = {
            "ap-1": ["im-1", "vo-1"],
        }
        self.n_inputs = 0
        self.n_queues = 0
        self.n_mounts = 0
        self.n_mount_files = 0
        self.mount_contents = {}
        self.files_name2sha = {}
        self.files_sha2data = {}
        self.function_id_for_function_call = {}
        self.client_calls = {}
        self.function_is_running = False
        self.n_functions = 0
        self.n_schedules = 0
        self.function2schedule = {}
        self.function_create_error = False
        self.heartbeat_status_code = None
        self.n_apps = 0
        self.classes = {}

        self.task_result = None

        self.nfs_files: Dict[str, Dict[str, api_pb2.SharedVolumePutFileRequest]] = defaultdict(dict)
        self.volume_files: Dict[str, Dict[str, VolumeFile]] = defaultdict(dict)
        self.images = {}
        self.image_build_function_ids = {}
        self.force_built_images = []
        self.fail_blob_create = []
        self.blob_create_metadata = None
        self.blob_multipart_threshold = 10_000_000

        self.precreated_functions = set()
        self.app_functions = {}
        self.fcidx = 0

        self.function_serialized = None
        self.class_serialized = None

        self.client_hello_metadata = None

        self.dicts = {}
        self.secrets = {}

        self.deployed_dicts = {}
        self.deployed_mounts = {
            (client_mount_name(), api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL): "mo-123",
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

        self.volume_counter = 0
        # Volume-id -> commit/reload count
        self.volume_commits: Dict[str, int] = defaultdict(lambda: 0)
        self.volume_reloads: Dict[str, int] = defaultdict(lambda: 0)

        self.sandbox_defs = []
        self.sandbox: subprocess.Popen = None
        self.sandbox_result: Optional[api_pb2.GenericResult] = None

        self.token_flow_localhost_port = None
        self.queue_max_len = 1_00

        self.container_heartbeat_abort = threading.Event()

        @self.function_body
        def default_function_body(*args, **kwargs):
            return sum(arg**2 for arg in args) + sum(value**2 for key, value in kwargs.items())

    def function_body(self, func):
        """Decorator for setting the function that will be called for any FunctionGetOutputs calls"""
        self._function_body = func
        return func

    def container_heartbeat_return_now(self, response: api_pb2.ContainerHeartbeatResponse):
        self.container_heartbeat_response = response
        self.container_heartbeat_abort.set()

    def get_function_metadata(self, object_id: str) -> api_pb2.FunctionHandleMetadata:
        definition: api_pb2.Function = self.app_functions[object_id]
        return api_pb2.FunctionHandleMetadata(
            function_name=definition.function_name,
            function_type=definition.function_type,
            web_url=definition.web_url,
            is_method=definition.is_method,
        )

    def get_class_metadata(self, object_id: str) -> api_pb2.ClassHandleMetadata:
        class_handle_metadata = api_pb2.ClassHandleMetadata()
        for f_name, f_id in self.classes[object_id].items():
            function_handle_metadata = self.get_function_metadata(f_id)
            class_handle_metadata.methods.append(
                api_pb2.ClassMethod(
                    function_name=f_name, function_id=f_id, function_handle_metadata=function_handle_metadata
                )
            )
        return class_handle_metadata

    def get_object_metadata(self, object_id) -> api_pb2.Object:
        if object_id.startswith("fu-"):
            res = api_pb2.Object(function_handle_metadata=self.get_function_metadata(object_id))

        elif object_id.startswith("cs-"):
            res = api_pb2.Object(class_handle_metadata=self.get_class_metadata(object_id))

        elif object_id.startswith("mo-"):
            mount_handle_metadata = api_pb2.MountHandleMetadata(content_checksum_sha256_hex="abc123")
            res = api_pb2.Object(mount_handle_metadata=mount_handle_metadata)

        elif object_id.startswith("sb-"):
            sandbox_handle_metadata = api_pb2.SandboxHandleMetadata(result=self.sandbox_result)
            res = api_pb2.Object(sandbox_handle_metadata=sandbox_handle_metadata)

        else:
            res = api_pb2.Object()

        res.object_id = object_id
        return res

    ### App

    async def AppCreate(self, stream):
        request: api_pb2.AppCreateRequest = await stream.recv_message()
        self.requests.append(request)
        self.n_apps += 1
        app_id = f"ap-{self.n_apps}"
        self.app_state_history[app_id].append(api_pb2.APP_STATE_INITIALIZING)
        await stream.send_message(
            api_pb2.AppCreateResponse(app_id=app_id, app_logs_url="https://modaltest.com/apps/ap-123")
        )

    async def AppClientDisconnect(self, stream):
        request: api_pb2.AppClientDisconnectRequest = await stream.recv_message()
        self.requests.append(request)
        self.done = True
        self.app_client_disconnect_count += 1
        state_history = self.app_state_history[request.app_id]
        if state_history[-1] not in [api_pb2.APP_STATE_DETACHED, api_pb2.APP_STATE_DEPLOYED]:
            state_history.append(api_pb2.APP_STATE_STOPPED)
        await stream.send_message(Empty())

    async def AppGetLogs(self, stream):
        request: api_pb2.AppGetLogsRequest = await stream.recv_message()
        if not request.last_entry_id:
            # Just count initial requests
            self.app_get_logs_initial_count += 1
            last_entry_id = "1"
        else:
            last_entry_id = str(int(request.last_entry_id) + 1)
        await asyncio.sleep(0.5)
        log = api_pb2.TaskLogs(data=f"hello, world ({last_entry_id})\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id=last_entry_id, items=[log]))
        if self.done:
            await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    async def AppGetObjects(self, stream):
        request: api_pb2.AppGetObjectsRequest = await stream.recv_message()
        object_ids = self.app_objects.get(request.app_id, {})
        objects = list(object_ids.items())
        if request.include_unindexed:
            unindexed_object_ids = self.app_unindexed_objects.get(request.app_id, [])
            objects += [(None, object_id) for object_id in unindexed_object_ids]
        items = [
            api_pb2.AppGetObjectsItem(tag=tag, object=self.get_object_metadata(object_id)) for tag, object_id in objects
        ]
        await stream.send_message(api_pb2.AppGetObjectsResponse(items=items))

    async def AppSetObjects(self, stream):
        request: api_pb2.AppSetObjectsRequest = await stream.recv_message()
        self.app_objects[request.app_id] = dict(request.indexed_object_ids)
        self.app_unindexed_objects[request.app_id] = list(request.unindexed_object_ids)
        if request.single_object_id:
            self.app_single_objects[request.app_id] = request.single_object_id
        self.app_set_objects_count += 1
        if request.new_app_state:
            self.app_state_history[request.app_id].append(request.new_app_state)
        await stream.send_message(Empty())

    async def AppDeploy(self, stream):
        request: api_pb2.AppDeployRequest = await stream.recv_message()
        self.deployed_apps[request.name] = request.app_id
        self.app_state_history[request.app_id].append(api_pb2.APP_STATE_DEPLOYED)
        await stream.send_message(api_pb2.AppDeployResponse(url="http://test.modal.com/foo/bar"))

    async def AppGetByDeploymentName(self, stream):
        request: api_pb2.AppGetByDeploymentNameRequest = await stream.recv_message()
        await stream.send_message(api_pb2.AppGetByDeploymentNameResponse(app_id=self.deployed_apps.get(request.name)))

    async def AppHeartbeat(self, stream):
        request: api_pb2.AppHeartbeatRequest = await stream.recv_message()
        self.requests.append(request)
        self.app_heartbeats[request.app_id] += 1
        await stream.send_message(Empty())

    ### Checkpoint

    async def ContainerCheckpoint(self, stream):
        request: api_pb2.ContainerCheckpointRequest = await stream.recv_message()
        self.requests.append(request)
        self.container_checkpoint_requests += 1
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
            self.n_blobs += 1
            blob_id = f"bl-{self.n_blobs}"
            num_parts = (req.content_length + self.blob_multipart_threshold - 1) // self.blob_multipart_threshold
            upload_urls = []
            for part_number in range(num_parts):
                upload_url = f"{self.blob_host}/upload?blob_id={blob_id}&part_number={part_number}"
                upload_urls.append(upload_url)

            await stream.send_message(
                api_pb2.BlobCreateResponse(
                    blob_id=blob_id,
                    multipart=api_pb2.MultiPartUpload(
                        part_length=self.blob_multipart_threshold,
                        upload_urls=upload_urls,
                        completion_url=f"{self.blob_host}/complete_multipart?blob_id={blob_id}",
                    ),
                )
            )
        else:
            self.n_blobs += 1
            blob_id = f"bl-{self.n_blobs}"
            upload_url = f"{self.blob_host}/upload?blob_id={blob_id}"
            await stream.send_message(api_pb2.BlobCreateResponse(blob_id=blob_id, upload_url=upload_url))

    async def BlobGet(self, stream):
        request: api_pb2.BlobGetRequest = await stream.recv_message()
        download_url = f"{self.blob_host}/download?blob_id={request.blob_id}"
        await stream.send_message(api_pb2.BlobGetResponse(download_url=download_url))

    ### Class

    async def ClassCreate(self, stream):
        request: api_pb2.ClassCreateRequest = await stream.recv_message()
        assert request.app_id
        methods: dict[str, str] = {method.function_name: method.function_id for method in request.methods}
        class_id = "cs-" + str(len(self.classes))
        self.classes[class_id] = methods
        await stream.send_message(
            api_pb2.ClassCreateResponse(class_id=class_id, handle_metadata=self.get_class_metadata(class_id))
        )

    async def ClassGet(self, stream):
        request: api_pb2.ClassGetRequest = await stream.recv_message()
        app_id = self.deployed_apps.get(request.app_name)
        app_objects = self.app_objects[app_id]
        object_id = app_objects.get(request.object_tag)
        if object_id is None:
            raise GRPCError(Status.NOT_FOUND, f"can't find object {request.object_tag}")
        await stream.send_message(
            api_pb2.ClassGetResponse(class_id=object_id, handle_metadata=self.get_class_metadata(object_id))
        )

    ### Client

    async def ClientHello(self, stream):
        request: Empty = await stream.recv_message()
        self.requests.append(request)
        self.client_create_metadata = stream.metadata
        client_version = stream.metadata["x-modal-client-version"]
        assert stream.user_agent.startswith(f"modal-client/{__version__} ")
        if stream.metadata.get("x-modal-token-id") == "bad":
            raise GRPCError(Status.UNAUTHENTICATED, "bad bad bad")
        elif client_version == "timeout":
            await asyncio.sleep(60)
            await stream.send_message(api_pb2.ClientHelloResponse())
        elif client_version == "unauthenticated":
            raise GRPCError(Status.UNAUTHENTICATED, "failed authentication")
        elif client_version == "deprecated":
            await stream.send_message(api_pb2.ClientHelloResponse(warning="SUPER OLD"))
        elif pkg_resources.parse_version(client_version) < pkg_resources.parse_version(__version__):
            raise GRPCError(Status.FAILED_PRECONDITION, "Old client")
        else:
            await stream.send_message(api_pb2.ClientHelloResponse())

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
        _request: api_pb2.ContainerExecRequest = await stream.recv_message()
        await stream.send_message(api_pb2.ContainerExecResponse(exec_id="container_exec_id"))

    async def ContainerExecGetOutput(self, stream):
        _request: api_pb2.ContainerExecGetOutputRequest = await stream.recv_message()
        await stream.send_message(
            api_pb2.RuntimeOutputBatch(
                items=[
                    api_pb2.RuntimeOutputMessage(
                        file_descriptor=api_pb2.FileDescriptor.FILE_DESCRIPTOR_STDOUT, message="Hello World"
                    )
                ]
            )
        )
        await stream.send_message(api_pb2.RuntimeOutputBatch(exit_code=0))

    ### Dict

    async def DictCreate(self, stream):
        request: api_pb2.DictCreateRequest = await stream.recv_message()
        if request.existing_dict_id:
            dict_id = request.existing_dict_id
        else:
            dict_id = f"di-{len(self.dicts)}"
            self.dicts[dict_id] = {}
        await stream.send_message(api_pb2.DictCreateResponse(dict_id=dict_id))

    async def DictGetOrCreate(self, stream):
        request: api_pb2.DictGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.namespace, request.environment_name)
        if k in self.deployed_dicts:
            dict_id = self.deployed_dicts[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
            dict_id = f"di-{len(self.dicts)}"
            self.dicts[dict_id] = {}
            self.deployed_dicts[k] = dict_id
        else:
            raise GRPCError(Status.NOT_FOUND, "Queue not found")
        await stream.send_message(api_pb2.DictGetOrCreateResponse(dict_id=dict_id))

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

    async def DictUpdate(self, stream):
        request: api_pb2.DictUpdateRequest = await stream.recv_message()
        for update in request.updates:
            self.dicts[request.dict_id][update.key] = update.value
        await stream.send_message(api_pb2.DictUpdateResponse())

    ### Function

    async def FunctionBindParams(self, stream):
        request: api_pb2.FunctionBindParamsRequest = await stream.recv_message()
        assert request.function_id
        assert request.serialized_params
        self.n_functions += 1
        function_id = f"fu-{self.n_functions}"

        await stream.send_message(api_pb2.FunctionBindParamsResponse(bound_function_id=function_id))

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

    async def FunctionGetInputs(self, stream):
        self.get_inputs_barrier.wait()
        request: api_pb2.FunctionGetInputsRequest = await stream.recv_message()
        assert request.function_id
        if self.fail_get_inputs:
            raise GRPCError(Status.INTERNAL)
        elif self.rate_limit_sleep_duration is not None:
            s = self.rate_limit_sleep_duration
            self.rate_limit_sleep_duration = None
            await stream.send_message(api_pb2.FunctionGetInputsResponse(rate_limit_sleep_duration=s))
        elif not self.container_inputs:
            await asyncio.sleep(1.0)
            await stream.send_message(api_pb2.FunctionGetInputsResponse(inputs=[]))
        else:
            await stream.send_message(self.container_inputs.pop(0))

    async def FunctionPutOutputs(self, stream):
        self.put_outputs_barrier.wait()
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
        await stream.send_message(
            api_pb2.FunctionPrecreateResponse(
                function_id=function_id,
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    function_name=req.function_name,
                    function_type=req.function_type,
                    web_url=web_url,
                ),
            )
        )

    async def FunctionCreate(self, stream):
        request: api_pb2.FunctionCreateRequest = await stream.recv_message()
        if self.function_create_error:
            raise GRPCError(Status.INTERNAL, "Function create failed")
        if request.existing_function_id:
            function_id = request.existing_function_id
        else:
            self.n_functions += 1
            function_id = f"fu-{self.n_functions}"
        if request.schedule:
            self.function2schedule[function_id] = request.schedule
        function = api_pb2.Function()
        function.CopyFrom(request.function)
        if function.webhook_config.type:
            function.web_url = "http://xyz.internal"

        self.app_functions[function_id] = function
        await stream.send_message(
            api_pb2.FunctionCreateResponse(
                function_id=function_id,
                function=function,
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    function_name=function.function_name,
                    function_type=function.function_type,
                    web_url=function.web_url,
                ),
            )
        )

    async def FunctionGet(self, stream):
        request: api_pb2.FunctionGetRequest = await stream.recv_message()
        app_id = self.deployed_apps.get(request.app_name)
        app_objects = self.app_objects[app_id]
        object_id = app_objects.get(request.object_tag)
        if object_id is None:
            raise GRPCError(Status.NOT_FOUND, f"can't find object {request.object_tag}")
        await stream.send_message(
            api_pb2.FunctionGetResponse(function_id=object_id, handle_metadata=self.get_function_metadata(object_id))
        )

    async def FunctionMap(self, stream):
        self.fcidx += 1
        request: api_pb2.FunctionMapRequest = await stream.recv_message()
        function_call_id = f"fc-{self.fcidx}"
        self.function_id_for_function_call[function_call_id] = request.function_id
        await stream.send_message(api_pb2.FunctionMapResponse(function_call_id=function_call_id))

    async def FunctionPutInputs(self, stream):
        request: api_pb2.FunctionPutInputsRequest = await stream.recv_message()
        response_items = []
        function_call_inputs = self.client_calls.setdefault(request.function_call_id, [])
        for item in request.inputs:
            args, kwargs = modal._serialization.deserialize(item.input.args, None) if item.input.args else ((), {})
            input_id = f"in-{self.n_inputs}"
            self.n_inputs += 1
            response_items.append(api_pb2.FunctionPutInputsResponseItem(input_id=input_id, idx=item.idx))
            function_call_inputs.append(((item.idx, input_id), (args, kwargs)))
        if self.slow_put_inputs:
            await asyncio.sleep(0.001)
        await stream.send_message(api_pb2.FunctionPutInputsResponse(inputs=response_items))

    async def FunctionGetOutputs(self, stream):
        request: api_pb2.FunctionGetOutputsRequest = await stream.recv_message()
        if request.clear_on_success:
            self.cleared_function_calls.add(request.function_call_id)

        client_calls = self.client_calls.get(request.function_call_id, [])
        if client_calls and not self.function_is_running:
            popidx = len(client_calls) // 2  # simulate that results don't always come in order
            (idx, input_id), (args, kwargs) = client_calls.pop(popidx)
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
                    status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                    data=serialized_exc,
                    exception=repr(exc),
                    traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                )
                output_exc = api_pb2.FunctionGetOutputsItem(
                    input_id=input_id, idx=idx, result=result, gen_index=0, data_format=api_pb2.DATA_FORMAT_PICKLE
                )

            if output_exc:
                output = output_exc
            else:
                output = api_pb2.FunctionGetOutputsItem(
                    input_id=input_id,
                    idx=idx,
                    result=api_pb2.GenericResult(
                        status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                        data=serialize_data_format(result, result_data_format),
                    ),
                    data_format=result_data_format,
                )

            await stream.send_message(api_pb2.FunctionGetOutputsResponse(outputs=[output]))
        else:
            await stream.send_message(api_pb2.FunctionGetOutputsResponse(outputs=[]))

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
        while True:
            chunk = await self.fc_data_out[req.function_call_id].get()
            await stream.send_message(chunk)

    async def FunctionCallPutDataOut(self, stream):
        req: api_pb2.FunctionCallPutDataRequest = await stream.recv_message()
        for chunk in req.data_chunks:
            await self.fc_data_out[req.function_call_id].put(chunk)
        await stream.send_message(Empty())

    ### Image

    async def ImageGetOrCreate(self, stream):
        request: api_pb2.ImageGetOrCreateRequest = await stream.recv_message()
        idx = len(self.images) + 1
        image_id = f"im-{idx}"

        self.images[image_id] = request.image
        self.image_build_function_ids[image_id] = request.build_function_id
        if request.force_build:
            self.force_built_images.append(image_id)
        await stream.send_message(api_pb2.ImageGetOrCreateResponse(image_id=image_id))

    async def ImageJoinStreaming(self, stream):
        await stream.recv_message()
        task_log_1 = api_pb2.TaskLogs(data="hello, world\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_INFO)
        task_log_2 = api_pb2.TaskLogs(
            task_progress=api_pb2.TaskProgress(
                len=1, pos=0, progress_type=api_pb2.IMAGE_SNAPSHOT_UPLOAD, description="xyz"
            )
        )
        await stream.send_message(api_pb2.ImageJoinStreamingResponse(task_logs=[task_log_1, task_log_2]))
        await stream.send_message(
            api_pb2.ImageJoinStreamingResponse(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
            )
        )

    ### Mount

    async def MountPutFile(self, stream):
        request: api_pb2.MountPutFileRequest = await stream.recv_message()
        if request.WhichOneof("data_oneof") is not None:
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
                raise GRPCError(Status.NOT_FOUND, "Mount not found")
            mount_id = self.deployed_mounts[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS:
            self.n_mounts += 1
            mount_id = f"mo-{self.n_mounts}"
            self.deployed_mounts[k] = mount_id
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP:
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

    ### Queue

    async def QueueCreate(self, stream):
        request: api_pb2.QueueCreateRequest = await stream.recv_message()
        if request.existing_queue_id:
            queue_id = request.existing_queue_id
        else:
            self.n_queues += 1
            queue_id = f"qu-{self.n_queues}"
        await stream.send_message(api_pb2.QueueCreateResponse(queue_id=queue_id))

    async def QueueGetOrCreate(self, stream):
        request: api_pb2.QueueGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.namespace, request.environment_name)
        if k in self.deployed_queues:
            queue_id = self.deployed_queues[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
            self.n_queues += 1
            queue_id = f"qu-{self.n_queues}"
            self.deployed_queues[k] = queue_id
        else:
            raise GRPCError(Status.NOT_FOUND, "Queue not found")
        await stream.send_message(api_pb2.QueueGetOrCreateResponse(queue_id=queue_id))

    async def QueuePut(self, stream):
        request: api_pb2.QueuePutRequest = await stream.recv_message()
        if len(self.queue) >= self.queue_max_len:
            raise GRPCError(Status.RESOURCE_EXHAUSTED, f"Hit servicer's max len for Queues: {self.queue_max_len}")
        self.queue += request.values
        await stream.send_message(Empty())

    async def QueueGet(self, stream):
        await stream.recv_message()
        if len(self.queue) > 0:
            values = [self.queue.pop(0)]
        else:
            values = []
        await stream.send_message(api_pb2.QueueGetResponse(values=values))

    async def QueueLen(self, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.QueueLenResponse(len=len(self.queue)))

    ### Sandbox

    async def SandboxCreate(self, stream):
        request: api_pb2.SandboxCreateRequest = await stream.recv_message()
        # Not using asyncio.subprocess here for Python 3.7 compatibility.
        self.sandbox = subprocess.Popen(
            request.definition.entrypoint_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        self.sandbox_defs.append(request.definition)
        await stream.send_message(api_pb2.SandboxCreateResponse(sandbox_id="sb-123"))

    async def SandboxGetLogs(self, stream):
        request: api_pb2.SandboxGetLogsRequest = await stream.recv_message()
        if request.file_descriptor == api_pb2.FILE_DESCRIPTOR_STDOUT:
            data = self.sandbox.stdout.read()
        else:
            data = self.sandbox.stderr.read()
        await stream.send_message(
            api_pb2.TaskLogsBatch(
                items=[api_pb2.TaskLogs(data=data.decode("utf-8"), file_descriptor=request.file_descriptor)]
            )
        )
        await stream.send_message(api_pb2.TaskLogsBatch(eof=True))

    async def SandboxWait(self, stream):
        request: api_pb2.SandboxWaitRequest = await stream.recv_message()
        try:
            self.sandbox.wait(timeout=request.timeout)
        except subprocess.TimeoutExpired:
            await stream.send_message(api_pb2.SandboxWaitResponse())
            return

        if self.sandbox.returncode != 0:
            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE, exitcode=self.sandbox.returncode
            )
        else:
            result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
        self.sandbox_result = result
        await stream.send_message(api_pb2.SandboxWaitResponse(result=result))

    async def SandboxTerminate(self, stream):
        self.sandbox.terminate()
        await stream.send_message(api_pb2.SandboxTerminateResponse())

    async def SandboxGetTaskId(self, stream):
        # only used for `modal shell` / `modal container exec`
        _request: api_pb2.SandboxGetTaskIdRequest = await stream.recv_message()
        await stream.send_message(api_pb2.SandboxGetTaskIdResponse(task_id="modal_container_exec"))

    ### Secret

    async def SecretGetOrCreate(self, stream):
        request: api_pb2.SecretGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.namespace, request.environment_name)
        if request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP:
            secret_id = "st-" + str(len(self.secrets))
            self.secrets[secret_id] = request.env_dict
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS:
            if k in self.deployed_secrets:
                raise GRPCError(Status.ALREADY_EXISTS, "Already exists")
            secret_id = None
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_OVERWRITE_IF_EXISTS:
            secret_id = None
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED:
            if k not in self.deployed_secrets:
                raise GRPCError(Status.NOT_FOUND, "No such secret")
            secret_id = self.deployed_secrets[k]
        else:
            raise Exception("unsupported creation type")

        if secret_id is None:  # Create one
            secret_id = "st-" + str(len(self.secrets))
            self.secrets[secret_id] = request.env_dict
            self.deployed_secrets[k] = secret_id

        await stream.send_message(api_pb2.SecretGetOrCreateResponse(secret_id=secret_id))

    async def SecretList(self, stream):
        await stream.recv_message()
        items = [api_pb2.SecretListItem(label=f"dummy-secret-{i}") for i, _ in enumerate(self.secrets)]
        await stream.send_message(api_pb2.SecretListResponse(items=items))

    ### Network File System (née Shared volume)

    async def SharedVolumeCreate(self, stream):
        nfs_id = f"sv-{len(self.nfs_files)}"
        self.nfs_files[nfs_id] = {}
        await stream.send_message(api_pb2.SharedVolumeCreateResponse(shared_volume_id=nfs_id))

    async def SharedVolumeGetOrCreate(self, stream):
        request: api_pb2.SharedVolumeGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.namespace, request.environment_name)
        if request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED:
            if k not in self.deployed_nfss:
                raise GRPCError(Status.NOT_FOUND, "NFS not found")
            nfs_id = self.deployed_nfss[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
            if k not in self.deployed_nfss:
                nfs_id = f"sv-{len(self.nfs_files)}"
                self.nfs_files[nfs_id] = {}
                self.deployed_nfss[k] = nfs_id
            nfs_id = self.deployed_nfss[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS:
            if k in self.deployed_nfss:
                raise GRPCError(Status.ALREADY_EXISTS, "NFS already exists")
            nfs_id = f"sv-{len(self.nfs_files)}"
            self.nfs_files[nfs_id] = {}
            self.deployed_nfss[k] = nfs_id
        else:
            raise GRPCError(Status.INVALID_ARGUMENT, "unsupported object creation type")

        await stream.send_message(api_pb2.SharedVolumeGetOrCreateResponse(shared_volume_id=nfs_id))

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

    ### Task

    async def TaskCurrentInputs(
        self, stream: "grpclib.server.Stream[Empty, api_pb2.TaskCurrentInputsResponse]"
    ) -> None:
        await stream.send_message(api_pb2.TaskCurrentInputsResponse(input_ids=[]))  # dummy implementation

    async def TaskResult(self, stream):
        request: api_pb2.TaskResultRequest = await stream.recv_message()
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
        await stream.send_message(
            api_pb2.WorkspaceNameLookupResponse(workspace_name="test-workspace", username="test-username")
        )

    ### Tunnel

    async def TunnelStart(self, stream):
        request: api_pb2.TunnelStartRequest = await stream.recv_message()
        port = request.port
        await stream.send_message(api_pb2.TunnelStartResponse(host=f"{port}.modal.test", port=443))

    async def TunnelStop(self, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.TunnelStopResponse(exists=True))

    ### Volume

    async def VolumeCreate(self, stream):
        req = await stream.recv_message()
        self.requests.append(req)
        self.volume_counter += 1
        volume_id = f"vo-{self.volume_counter}"
        self.volume_files[volume_id] = {}
        await stream.send_message(api_pb2.VolumeCreateResponse(volume_id=volume_id))

    async def VolumeGetOrCreate(self, stream):
        request: api_pb2.VolumeGetOrCreateRequest = await stream.recv_message()
        k = (request.deployment_name, request.namespace, request.environment_name)
        if request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED:
            if k not in self.deployed_volumes:
                raise GRPCError(Status.NOT_FOUND, "Volume not found")
            volume_id = self.deployed_volumes[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING:
            if k not in self.deployed_volumes:
                volume_id = f"vo-{len(self.volume_files)}"
                self.volume_files[volume_id] = {}
                self.deployed_volumes[k] = volume_id
            volume_id = self.deployed_volumes[k]
        elif request.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS:
            if k in self.deployed_volumes:
                raise GRPCError(Status.ALREADY_EXISTS, "Volume already exists")
            volume_id = f"vo-{len(self.volume_files)}"
            self.volume_files[volume_id] = {}
            self.deployed_volumes[k] = volume_id
        else:
            raise GRPCError(Status.INVALID_ARGUMENT, "unsupported object creation type")

        await stream.send_message(api_pb2.VolumeGetOrCreateResponse(volume_id=volume_id))

    async def VolumeCommit(self, stream):
        req = await stream.recv_message()
        self.requests.append(req)
        if not req.volume_id.startswith("vo-"):
            raise GRPCError(Status.NOT_FOUND, f"invalid volume ID {req.volume_id}")
        self.volume_commits[req.volume_id] += 1
        await stream.send_message(api_pb2.VolumeCommitResponse(skip_reload=False))

    async def VolumeReload(self, stream):
        req = await stream.recv_message()
        self.requests.append(req)
        self.volume_reloads[req.volume_id] += 1
        await stream.send_message(Empty())

    async def VolumeGetFile(self, stream):
        req = await stream.recv_message()
        if req.path.decode("utf-8") not in self.volume_files[req.volume_id]:
            raise GRPCError(Status.NOT_FOUND, "File not found")
        vol_file = self.volume_files[req.volume_id][req.path.decode("utf-8")]
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

    async def VolumeRemoveFile(self, stream):
        req = await stream.recv_message()
        if req.path.decode("utf-8") not in self.volume_files[req.volume_id]:
            raise GRPCError(Status.INVALID_ARGUMENT, "File not found")
        del self.volume_files[req.volume_id][req.path.decode("utf-8")]
        await stream.send_message(Empty())

    async def VolumeListFiles(self, stream):
        req = await stream.recv_message()
        if req.path != "**":
            raise NotImplementedError("Only '**' listing is supported.")
        for k, vol_file in self.volume_files[req.volume_id].items():
            entries = [
                api_pb2.VolumeListFilesEntry(
                    path=k, type=api_pb2.VolumeListFilesEntry.FileType.FILE, size=len(vol_file.data)
                )
            ]
            await stream.send_message(api_pb2.VolumeListFilesResponse(entries=entries))

    async def VolumePutFiles(self, stream):
        req = await stream.recv_message()
        for file in req.files:
            blob_data = self.files_sha2data[file.sha256_hex]

            if file.filename in self.volume_files[req.volume_id] and req.disallow_overwrite_existing_files:
                raise GRPCError(
                    Status.ALREADY_EXISTS,
                    f"{file.filename}: already exists (disallow_overwrite_existing_files={req.disallow_overwrite_existing_files}",
                )

            self.volume_files[req.volume_id][file.filename] = VolumeFile(
                data=blob_data["data"],
                data_blob_id=blob_data["data_blob_id"],
                mode=file.mode,
            )
        await stream.send_message(Empty())

    async def VolumeCopyFiles(self, stream):
        req = await stream.recv_message()
        for src_path in req.src_paths:
            if src_path.decode("utf-8") not in self.volume_files[req.volume_id]:
                raise GRPCError(Status.NOT_FOUND, f"Source file not found: {src_path}")
            src_file = self.volume_files[req.volume_id][src_path.decode("utf-8")]
            if len(req.src_paths) > 1:
                # check to make sure dst is a directory
                if (
                    req.dst_path.decode("utf-8").endswith(("/", "\\"))
                    or not os.path.splitext(os.path.basename(req.dst_path))[1]
                ):
                    dst_path = os.path.join(req.dst_path, os.path.basename(src_path))
                else:
                    raise GRPCError(Status.INVALID_ARGUMENT, f"{dst_path} is not a directory.")
            else:
                dst_path = req.dst_path
            self.volume_files[req.volume_id][dst_path.decode("utf-8")] = src_file
        await stream.send_message(Empty())


@pytest_asyncio.fixture
async def blob_server():
    blobs = {}
    blob_parts: Dict[str, Dict[int, bytes]] = defaultdict(dict)

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

    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.put("/upload", upload)])
    app.add_routes([aiohttp.web.get("/download", download)])
    app.add_routes([aiohttp.web.post("/complete_multipart", complete_multipart)])

    async with run_temporary_http_server(app) as host:
        yield host, blobs


@pytest_asyncio.fixture(scope="function")
async def servicer_factory(blob_server):
    @contextlib.asynccontextmanager
    async def create_server(host=None, port=None, path=None):
        blob_host, blobs = blob_server
        servicer = MockClientServicer(blob_host, blobs)  # type: ignore
        server = None

        async def _start_servicer():
            nonlocal server
            server = grpclib.server.Server([servicer])
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
            yield servicer
        finally:
            await stop_servicer.aio()

    yield create_server


@pytest_asyncio.fixture(scope="function")
async def servicer(servicer_factory):
    port = find_free_port()
    async with servicer_factory(host="0.0.0.0", port=port) as servicer:
        servicer.remote_addr = f"http://127.0.0.1:{port}"
        yield servicer


@pytest_asyncio.fixture(scope="function")
async def unix_servicer(servicer_factory):
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, "servicer.sock")
        async with servicer_factory(path=path) as servicer:
            servicer.remote_addr = f"unix://{path}"
            yield servicer


@pytest_asyncio.fixture(scope="function")
async def client(servicer):
    with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def container_client(unix_servicer):
    async with Client(unix_servicer.remote_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret")) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def server_url_env(servicer, monkeypatch):
    monkeypatch.setenv("MODAL_SERVER_URL", servicer.remote_addr)
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
        _ContainerApp._reset_container()


@pytest.fixture(scope="module")
def test_dir(request):
    """Absolute path to directory containing test file."""
    root_dir = Path(request.config.rootdir)
    test_dir = Path(os.getenv("PYTEST_CURRENT_TEST")).parent
    return root_dir / test_dir


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


@pytest.fixture()
def modal_test_support_dir(request):
    # TODO: merge this with test/supports dir?
    root_dir = Path(request.config.rootdir)
    return root_dir / "modal_test_support"
