# Copyright Modal Labs 2022
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import os
import pytest
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

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
from modal.client import Client
from modal.image import _dockerhub_python_version
from modal.mount import client_mount_name
from modal_proto import api_grpc, api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_testing import patch_mock_servicer
from modal_utils.grpc_utils import find_free_port
from modal_utils.http_utils import run_temporary_http_server


def function_definition_to_handle_metadata(definition: Optional[api_pb2.Function]) -> api_pb2.FunctionHandleMetadata:
    if definition is None:
        return None
    return api_pb2.FunctionHandleMetadata(
        function_name=definition.function_name,
        function_type=definition.function_type,
        web_url=definition.web_url,
    )


@patch_mock_servicer
class MockClientServicer(api_grpc.ModalClientBase):
    # TODO(erikbern): add more annotations
    container_inputs: list[api_pb2.FunctionGetInputsResponse]
    container_outputs: list[api_pb2.FunctionPutOutputsRequest]

    def __init__(self, blob_host, blobs):
        self.app_state_history = defaultdict(list)
        self.app_heartbeats: Dict[str, int] = defaultdict(int)
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
        self.n_inputs = 0
        self.n_queues = 0
        self.n_mounts = 0
        self.n_mount_files = 0
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

        self.task_result = None

        self.nfs_files: Dict[str, Dict[str, api_pb2.SharedVolumePutFileRequest]] = defaultdict(dict)
        self.images = {}
        self.image_build_function_ids = {}
        self.force_built_images = []
        self.fail_blob_create = []
        self.blob_create_metadata = None
        self.blob_multipart_threshold = 10_000_000

        self.precreated_functions = set()
        self.app_functions = {}
        self.fcidx = 0
        self.secrets = {}

        self.function_serialized = None
        self.class_serialized = None

        self.client_hello_metadata = None

        self.dicts = {}

        self.cleared_function_calls = set()

        self.cancelled_calls = []

        self.app_client_disconnect_count = 0
        self.app_get_logs_initial_count = 0
        self.app_set_objects_count = 0

        self.function2params = {}

        self.volume_counter = 0
        # Volume-id -> commit/reload count
        self.volume_commits: Dict[str, int] = defaultdict(lambda: 0)
        self.volume_reloads: Dict[str, int] = defaultdict(lambda: 0)

        self.sandbox: subprocess.Popen = None

        @self.function_body
        def default_function_body(*args, **kwargs):
            return sum(arg**2 for arg in args) + sum(value**2 for key, value in kwargs.items())

    def function_body(self, func):
        """Decorator for setting the function that will be called for any FunctionGetOutputs calls"""
        self._function_body = func
        return func

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
        items = [
            api_pb2.AppGetObjectsItem(
                tag=tag,
                object_id=object_id,
                function=function_definition_to_handle_metadata(self.app_functions.get(object_id)),
            )
            for tag, object_id in object_ids.items()
        ]
        await stream.send_message(api_pb2.AppGetObjectsResponse(items=items))

    async def AppSetObjects(self, stream):
        request: api_pb2.AppSetObjectsRequest = await stream.recv_message()
        self.app_objects[request.app_id] = dict(request.indexed_object_ids)
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

    async def AppLookupObject(self, stream):
        request: api_pb2.AppLookupObjectRequest = await stream.recv_message()
        if not request.object_id:
            app_id = self.deployed_apps.get(request.app_name)
            if app_id is None:
                raise GRPCError(Status.NOT_FOUND, f"can't find app {request.app_name}")
            app_objects = self.app_objects[app_id]
            if request.object_tag:
                object_id = app_objects.get(request.object_tag)
                if object_id is None:
                    raise GRPCError(Status.NOT_FOUND, f"can't find object {request.object_tag}")
            else:
                (object_id,) = list(app_objects.values())
        else:
            object_id = request.object_id

        if request.app_name:
            assert request.object_entity
            if object_id:
                assert object_id.startswith(request.object_entity)

        function = function_definition_to_handle_metadata(self.app_functions.get(object_id))
        await stream.send_message(api_pb2.AppLookupObjectResponse(object_id=object_id, function=function))

    async def AppHeartbeat(self, stream):
        request: api_pb2.AppHeartbeatRequest = await stream.recv_message()
        self.requests.append(request)
        self.app_heartbeats[request.app_id] += 1
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
        await stream.send_message(Empty())

    ### Dict

    async def DictCreate(self, stream):
        dict_id = f"di-{len(self.dicts)}"
        self.dicts[dict_id] = {}
        await stream.send_message(api_pb2.DictCreateResponse(dict_id=dict_id))

    async def DictGet(self, stream):
        request: api_pb2.DictGetRequest = await stream.recv_message()
        d = self.dicts[request.dict_id]
        await stream.send_message(api_pb2.DictGetResponse(value=d.get(request.key), found=bool(request.key in d)))

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
        self.function2params[self.n_functions] = request.serialized_params
        function_id = f"fu-{self.n_functions}"

        await stream.send_message(api_pb2.FunctionBindParamsResponse(bound_function_id=function_id))

    async def FunctionGetInputs(self, stream):
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

    async def FunctionMap(self, stream):
        self.fcidx += 1
        request: api_pb2.FunctionMapRequest = await stream.recv_message()
        function_call_id = f"fc-{self.fcidx}"
        self.function_id_for_function_call[function_call_id] = request.function_id
        await stream.send_message(api_pb2.FunctionMapResponse(function_call_id=function_call_id))

    async def FunctionPutInputs(self, stream):
        request: api_pb2.FunctionPutInputsRequest = await stream.recv_message()
        response_items = []
        function_calls = self.client_calls.setdefault(request.function_call_id, [])
        for item in request.inputs:
            args, kwargs = cloudpickle.loads(item.input.args) if item.input.args else ((), {})
            input_id = f"in-{self.n_inputs}"
            self.n_inputs += 1
            response_items.append(api_pb2.FunctionPutInputsResponseItem(input_id=input_id, idx=item.idx))
            function_calls.append(((item.idx, input_id), (args, kwargs)))
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
            results = []
            output_exc = None
            try:
                res = self._function_body(*args, **kwargs)

                if inspect.iscoroutine(res):
                    results = [await res]
                elif inspect.isgenerator(res):
                    for item in res:
                        results.append(item)
                else:
                    results = [res]
            except Exception as exc:
                serialized_exc = cloudpickle.dumps(exc)
                result = api_pb2.GenericResult(
                    status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                    data=serialized_exc,
                    exception=repr(exc),
                    traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                )
                output_exc = api_pb2.FunctionGetOutputsItem(input_id=input_id, idx=idx, result=result, gen_index=0)

            outputs = []
            for index, value in enumerate(results):
                gen_status = api_pb2.GenericResult.GENERATOR_STATUS_UNSPECIFIED
                if inspect.isgenerator(res):
                    gen_status = api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE

                result = api_pb2.GenericResult(
                    status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                    data=cloudpickle.dumps(value),
                    gen_status=gen_status,
                )
                item = api_pb2.FunctionGetOutputsItem(
                    input_id=input_id,
                    idx=idx,
                    result=result,
                    gen_index=index,
                )
                outputs.append(item)

            if output_exc:
                outputs.append(output_exc)
            elif inspect.isgenerator(res):
                finish_item = api_pb2.FunctionGetOutputsItem(
                    input_id=input_id,
                    idx=idx,
                    result=api_pb2.GenericResult(
                        status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                        gen_status=api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE,
                    ),
                )
                outputs.append(finish_item)

            await stream.send_message(api_pb2.FunctionGetOutputsResponse(outputs=outputs))
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

    ### Image

    async def ImageGetOrCreate(self, stream):
        request: api_pb2.ImageGetOrCreateRequest = await stream.recv_message()
        idx = len(self.images)

        self.images[idx] = request.image
        self.image_build_function_ids[idx] = request.build_function_id
        if request.force_build:
            self.force_built_images.append(idx)
        await stream.send_message(api_pb2.ImageGetOrCreateResponse(image_id=f"im-{idx}"))

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

    async def MountBuild(self, stream):
        request: api_pb2.MountBuildRequest = await stream.recv_message()
        for file in request.files:
            self.files_name2sha[file.filename] = file.sha256_hex
        self.n_mounts += 1
        await stream.send_message(api_pb2.MountBuildResponse(mount_id="mo-123"))

    ### Queue

    async def QueueCreate(self, stream):
        request: api_pb2.QueueCreateRequest = await stream.recv_message()
        if request.existing_queue_id:
            queue_id = request.existing_queue_id
        else:
            self.n_queues += 1
            queue_id = f"qu-{self.n_queues}"
        await stream.send_message(api_pb2.QueueCreateResponse(queue_id=queue_id))

    async def QueuePut(self, stream):
        request: api_pb2.QueuePutRequest = await stream.recv_message()
        self.queue += request.values
        await stream.send_message(Empty())

    async def QueueGet(self, stream):
        await stream.recv_message()
        if len(self.queue) > 0:
            values = [self.queue.pop(0)]
        else:
            values = []
        await stream.send_message(api_pb2.QueueGetResponse(values=values))

    ### Sandbox

    async def SandboxCreate(self, stream):
        request: api_pb2.SandboxCreateRequest = await stream.recv_message()
        # Not using asyncio.subprocess here for Python 3.7 compatibility.
        self.sandbox = subprocess.Popen(
            request.definition.entrypoint_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
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
        self.sandbox.wait()
        if self.sandbox.returncode != 0:
            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE, exitcode=self.sandbox.returncode
            )
        else:
            result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
        await stream.send_message(api_pb2.SandboxWaitResponse(result=result))

    ### Secret

    async def SecretCreate(self, stream):
        msg: api_pb2.SecretCreateRequest = await stream.recv_message()
        secret_id = "st-" + str(len(self.secrets))
        self.secrets[secret_id] = msg
        await stream.send_message(api_pb2.SecretCreateResponse(secret_id=secret_id))

    async def SecretList(self, stream):
        await stream.recv_message()
        items = [api_pb2.SecretListItem(label=f"dummy-secret-{i}") for i, _ in enumerate(self.secrets)]
        await stream.send_message(api_pb2.SecretListResponse(items=items))

    ### Shared volume

    async def SharedVolumeCreate(self, stream):
        nfs_id = f"sv-{len(self.nfs_files)}"
        self.nfs_files[nfs_id] = {}
        await stream.send_message(api_pb2.SharedVolumeCreateResponse(shared_volume_id=nfs_id))

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

    async def TaskResult(self, stream):
        request: api_pb2.TaskResultRequest = await stream.recv_message()
        self.task_result = request.result
        await stream.send_message(Empty())

    ### Token flow

    async def TokenFlowCreate(self, stream):
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

    async def VolumeCreate(self, stream):
        await stream.recv_message()
        self.volume_counter += 1
        await stream.send_message(api_pb2.VolumeCreateResponse(volume_id=f"vo-{self.volume_counter}"))

    async def VolumeCommit(self, stream):
        req = await stream.recv_message()
        self.volume_commits[req.volume_id] += 1
        await stream.send_message(Empty())

    async def VolumeReload(self, stream):
        req = await stream.recv_message()
        self.volume_reloads[req.volume_id] += 1
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
            server.close()
            await server.wait_closed()

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
        _App._reset_container()


@pytest.fixture(scope="module")
def test_dir(request):
    """Absolute path to directory containing test file."""
    root_dir = Path(request.config.rootdir)
    test_dir = Path(os.getenv("PYTEST_CURRENT_TEST")).parent
    return root_dir / test_dir
