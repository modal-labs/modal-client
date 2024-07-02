# Copyright Modal Labs 2023
import asyncio
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple
from unittest import mock

from modal import Client
from modal._container_entrypoint import UserException, main
from modal._serialization import (
    serialize,
)
from modal.app import _App
from modal_proto import api_pb2


def deploy_app_externally(
    servicer,
    file_or_module: str,
    app_variable: Optional[str] = None,
    deployment_name="Deployment",
    cwd=None,
    env={},
    capture_output=True,
) -> Optional[str]:
    # deploys an app from another interpreter to prevent leaking state from client into a container process
    # (apart from what goes through the servicer) also has the advantage that no modules imported by the
    # test files themselves will be added to sys.modules and included in mounts etc.
    windows_support: dict[str, str] = {}

    if sys.platform == "win32":
        windows_support = {
            **os.environ.copy(),
            **{"PYTHONUTF8": "1"},
        }  # windows apparently needs a bunch of env vars to start python...

    env = {**windows_support, "MODAL_SERVER_URL": servicer.client_addr, "MODAL_ENVIRONMENT": "main", **env}
    if cwd is None:
        cwd = pathlib.Path(__file__).parent.parent

    app_ref = file_or_module if app_variable is None else f"{file_or_module}::{app_variable}"

    p = subprocess.Popen(
        [sys.executable, "-m", "modal.cli.entry_point", "deploy", app_ref, "--name", deployment_name],
        cwd=cwd,
        env=env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE if capture_output else None,
    )
    stdout_b, stderr_b = p.communicate()
    stdout_s, stderr_s = (b.decode() if b is not None else None for b in (stdout_b, stderr_b))
    if p.returncode != 0:
        print(f"Deploying app failed!\n### stdout ###\n{stdout_s}\n### stderr ###\n{stderr_s}")
        raise Exception("Test helper failed to deploy app")
    return stdout_s


@dataclasses.dataclass
class ContainerResult:
    client: Client
    items: List[api_pb2.FunctionPutOutputsItem]
    data_chunks: List[api_pb2.DataChunk]
    task_result: api_pb2.GenericResult


def _run_container(
    servicer,
    module_name,
    function_name,
    fail_get_inputs=False,
    inputs=None,
    function_type=api_pb2.Function.FUNCTION_TYPE_FUNCTION,
    webhook_type=api_pb2.WEBHOOK_TYPE_UNSPECIFIED,
    definition_type=api_pb2.Function.DEFINITION_TYPE_FILE,
    app_name: str = "",
    is_builder_function: bool = False,
    allow_concurrent_inputs: Optional[int] = None,
    serialized_params: Optional[bytes] = None,
    is_checkpointing_function: bool = False,
    deps: List[str] = ["im-1"],
    volume_mounts: Optional[List[api_pb2.VolumeMount]] = None,
    is_auto_snapshot: bool = False,
    max_inputs: Optional[int] = None,
    is_class: bool = False,
) -> ContainerResult:
    container_args = _container_args(
        module_name,
        function_name,
        function_type,
        webhook_type,
        definition_type,
        app_name,
        is_builder_function,
        allow_concurrent_inputs,
        serialized_params,
        is_checkpointing_function,
        deps,
        volume_mounts,
        is_auto_snapshot,
        max_inputs,
        is_class=is_class,
    )
    with Client(servicer.container_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret")) as client:
        if inputs is None:
            servicer.container_inputs = _get_inputs()
        else:
            servicer.container_inputs = inputs
        function_call_id = servicer.container_inputs[0].inputs[0].function_call_id
        servicer.fail_get_inputs = fail_get_inputs

        if module_name in sys.modules:
            # Drop the module from sys.modules since some function code relies on the
            # assumption that that the app is created before the user code is imported.
            # This is really only an issue for tests.
            sys.modules.pop(module_name)

        env = os.environ.copy()
        temp_restore_file_path = tempfile.NamedTemporaryFile()
        if is_checkpointing_function:
            # State file is written to allow for a restore to happen.
            tmp_file_name = temp_restore_file_path.name
            with pathlib.Path(tmp_file_name).open("w") as target:
                json.dump({}, target)
            env["MODAL_RESTORE_STATE_PATH"] = tmp_file_name

            # Override server URL to reproduce restore behavior.
            env["MODAL_SERVER_URL"] = servicer.container_addr

        # reset _App tracking state between runs
        _App._all_apps.clear()

        try:
            with mock.patch.dict(os.environ, env):
                main(container_args, client)
        except UserException:
            # Handle it gracefully
            pass
        finally:
            temp_restore_file_path.close()

        # Flatten outputs
        items = _flatten_outputs(servicer.container_outputs)

        # Get data chunks
        data_chunks: List[api_pb2.DataChunk] = []
        if function_call_id in servicer.fc_data_out:
            try:
                while True:
                    chunk = servicer.fc_data_out[function_call_id].get_nowait()
                    data_chunks.append(chunk)
            except asyncio.QueueEmpty:
                pass

        return ContainerResult(client, items, data_chunks, servicer.task_result)


def _container_args(
    module_name,
    function_name,
    function_type=api_pb2.Function.FUNCTION_TYPE_FUNCTION,
    webhook_type=api_pb2.WEBHOOK_TYPE_UNSPECIFIED,
    definition_type=api_pb2.Function.DEFINITION_TYPE_FILE,
    app_name: str = "",
    is_builder_function: bool = False,
    allow_concurrent_inputs: Optional[int] = None,
    serialized_params: Optional[bytes] = None,
    is_checkpointing_function: bool = False,
    deps: List[str] = ["im-1"],
    volume_mounts: Optional[List[api_pb2.VolumeMount]] = None,
    is_auto_snapshot: bool = False,
    max_inputs: Optional[int] = None,
    is_class: bool = False,
):
    if webhook_type:
        webhook_config = api_pb2.WebhookConfig(
            type=webhook_type,
            method="GET",
            async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
        )
    else:
        webhook_config = None

    function_def = api_pb2.Function(
        module_name=module_name,
        function_name=function_name,
        function_type=function_type,
        volume_mounts=volume_mounts,
        webhook_config=webhook_config,
        definition_type=definition_type,
        app_name=app_name or "",
        is_builder_function=is_builder_function,
        is_auto_snapshot=is_auto_snapshot,
        allow_concurrent_inputs=allow_concurrent_inputs,
        is_checkpointing_function=is_checkpointing_function,
        object_dependencies=[api_pb2.ObjectDependency(object_id=object_id) for object_id in deps],
        max_inputs=max_inputs,
        is_class=is_class,
    )

    return api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        app_id="ap-1",
        function_def=function_def,
        serialized_params=serialized_params,
        checkpoint_id=f"ch-{uuid.uuid4()}",
    )


def _get_inputs(
    args: Tuple[Tuple, Dict] = ((42,), {}),
    n: int = 1,
    kill_switch=True,
    method_name: Optional[str] = None,
) -> List[api_pb2.FunctionGetInputsResponse]:
    input_pb = api_pb2.FunctionInput(
        args=serialize(args), data_format=api_pb2.DATA_FORMAT_PICKLE, method_name=method_name or ""
    )
    inputs = [
        *(
            api_pb2.FunctionGetInputsItem(input_id=f"in-xyz{i}", function_call_id="fc-123", input=input_pb)
            for i in range(n)
        ),
        *([api_pb2.FunctionGetInputsItem(kill_switch=True)] if kill_switch else []),
    ]
    return [api_pb2.FunctionGetInputsResponse(inputs=[x]) for x in inputs]


def _flatten_outputs(outputs) -> List[api_pb2.FunctionPutOutputsItem]:
    items: List[api_pb2.FunctionPutOutputsItem] = []
    for req in outputs:
        items += list(req.outputs)
    return items
