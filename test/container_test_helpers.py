# Copyright Modal Labs 2022

import dataclasses
import json
import os
import pathlib
import pytest
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Optional, Sequence
from unittest import mock

from modal import Client
from modal._container_entrypoint import UserException, main
from modal._serialization import (
    deserialize,
    deserialize_data_format,
    serialize,
    serialize_data_format,
)
from modal._utils.async_utils import synchronize_api
from modal._utils.blob_utils import (
    blob_download as _blob_download,
    blob_upload as _blob_upload,
)
from modal.app import _App
from modal_proto import api_pb2

from .helpers import deploy_app_externally

EXTRA_TOLERANCE_DELAY = 2.0 if sys.platform == "linux" else 5.0
FUNCTION_CALL_ID = "fc-123"
SLEEP_DELAY = 0.1
SLEEP_TIME = 0.1

DEFAULT_APP_LAYOUT_SENTINEL: Any = object()

blob_upload = synchronize_api(_blob_upload)
blob_download = synchronize_api(_blob_download)


@dataclasses.dataclass
class ContainerResult:
    client: Client
    items: list[api_pb2.FunctionPutOutputsItem]
    data_chunks: list[api_pb2.DataChunk]
    task_result: api_pb2.GenericResult


# =============================================================================
# Deployment Helpers
# =============================================================================
# Isolated Deployment: create an isolated servicer, deploys an app externally, and return deployed metadata
# Deploy App Externally: spawn python process to run `modal deploy` and return cli output


def isolated_deploy(module_name: str, app_variable_name: Optional[str] = None):
    """Package-scoped fixture that deploys an app using an ephemeral servicer instance

    Returns:
        tuple[dict[str, tuple[str, api_pb2.Function]], api_pb2.AppLayout]:
            The function definitions (by name) and app layout for the deployed app.
    """
    from test.conftest import blob_server_factory, servicer_factory

    # Create isolated servicer instance using our new factories
    with blob_server_factory() as blob_server:
        credentials = ("test-ak-" + str(uuid.uuid4()), "test-as-" + str(uuid.uuid4()))

        async def _deploy_and_get_functions():
            async with servicer_factory(blob_server, credentials) as servicer:
                deploy_app_externally(servicer, credentials, module_name, app_variable_name, capture_output=False)

                app_layout = servicer.app_get_layout("ap-1")
                functions_by_name = {}

                for func_id, func_def in servicer.app_functions.items():
                    function_name = func_def.function_name
                    functions_by_name[function_name] = (func_id, func_def)

                return functions_by_name, app_layout

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            function_definitions, app_layout = loop.run_until_complete(_deploy_and_get_functions())
        finally:
            loop.close()

    return function_definitions, app_layout


# =============================================================================
# Container Helpers
# =============================================================================
# _run_container: run container entrypoint inline with synthetic metadata created by `_container_args`
# _run_container_auto: run container entrypoint inline
#     with real deployed metadata (function_def and app_layout) from isolated deployment
# _run_container_process: run container entrypoint in a subprocess with synthetic metadata created by `_container_args`
# _run_container_process_auto: run container entrypoint in a subprocess
#     with real deployed metadata (function_def and app_layout) from isolated deployment


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
    max_concurrent_inputs: Optional[int] = None,
    target_concurrent_inputs: Optional[int] = None,
    batch_max_size: int = 0,
    batch_wait_ms: int = 0,
    serialized_params: Optional[bytes] = None,
    is_checkpointing_function: bool = False,
    deps: list[str] = ["im-1"],
    volume_mounts: Optional[list[api_pb2.VolumeMount]] = None,
    is_auto_snapshot: bool = False,
    max_inputs: Optional[int] = None,
    is_class: bool = False,
    class_parameter_info=api_pb2.ClassParameterInfo(
        format=api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_UNSPECIFIED, schema=[]
    ),
    app_layout=DEFAULT_APP_LAYOUT_SENTINEL,
    web_server_port: Optional[int] = None,
    web_server_startup_timeout: Optional[float] = None,
    function_serialized: Optional[bytes] = None,
    class_serialized: Optional[bytes] = None,
    supported_output_formats: Sequence["api_pb2.DataFormat.ValueType"] = [
        api_pb2.DATA_FORMAT_PICKLE,
        api_pb2.DATA_FORMAT_CBOR,
    ],
    method_definitions: dict[str, api_pb2.MethodDefinition] = {},
    is_server: bool = False,
) -> ContainerResult:
    container_args = _container_args(
        module_name=module_name,
        function_name=function_name,
        function_type=function_type,
        webhook_type=webhook_type,
        definition_type=definition_type,
        app_name=app_name,
        is_builder_function=is_builder_function,
        max_concurrent_inputs=max_concurrent_inputs,
        target_concurrent_inputs=target_concurrent_inputs,
        batch_max_size=batch_max_size,
        batch_wait_ms=batch_wait_ms,
        serialized_params=serialized_params,
        is_checkpointing_function=is_checkpointing_function,
        deps=deps,
        volume_mounts=volume_mounts,
        is_auto_snapshot=is_auto_snapshot,
        max_inputs=max_inputs,
        is_class=is_class,
        class_parameter_info=class_parameter_info,
        app_layout=app_layout,
        web_server_port=web_server_port,
        web_server_startup_timeout=web_server_startup_timeout,
        function_serialized=function_serialized,
        class_serialized=class_serialized,
        supported_output_formats=supported_output_formats,
        method_definitions=method_definitions,
        is_server=is_server,
    )
    with Client(servicer.container_addr, api_pb2.CLIENT_TYPE_CONTAINER, None) as client:
        if inputs is None:
            servicer.container_inputs = _get_inputs()
        else:
            servicer.container_inputs = inputs
        first_function_call_id = servicer.container_inputs[0].inputs[0].function_call_id
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
            env["MODAL_ENABLE_SNAP_RESTORE"] = "1"

        # These env vars are always present in containers
        env["MODAL_SERVER_URL"] = servicer.container_addr
        env["MODAL_TASK_ID"] = "ta-123"
        env["MODAL_IS_REMOTE"] = "1"

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

        data_chunks = servicer.get_data_chunks(first_function_call_id)

        return ContainerResult(client, items, data_chunks, servicer.task_result)


def _run_container_auto(
    servicer,
    function_name: str,
    deployed_support_function_definitions,
    *,
    inputs=None,
    serialized_params: bytes = b"",
    is_checkpointing_function: bool = False,
) -> ContainerResult:
    """
    Utility function that runs a function from test.supports.functions using predeployed
    function definitions instead of constructing them
    """
    functions_dict, app_layout = deployed_support_function_definitions
    function_id, function_def = functions_dict[function_name]
    function_def.is_checkpointing_function = is_checkpointing_function

    # Create container arguments using the predeployed function definition
    container_args = api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id=function_id,
        app_id="ap-1",  # Use the same app ID as the deployed functions
        function_def=function_def,
        checkpoint_id=f"ch-{uuid.uuid4()}",
        app_layout=app_layout,
        serialized_params=serialized_params,
    )

    # Use the same container execution logic as _run_container
    with Client(servicer.container_addr, api_pb2.CLIENT_TYPE_CONTAINER, None) as client:
        if inputs is None:
            servicer.container_inputs = _get_inputs()
        else:
            servicer.container_inputs = inputs
        first_function_call_id = servicer.container_inputs[0].inputs[0].function_call_id

        env = os.environ.copy()

        # These env vars are always present in containers
        env["MODAL_SERVER_URL"] = servicer.container_addr
        env["MODAL_TASK_ID"] = "ta-123"
        env["MODAL_IS_REMOTE"] = "1"

        if is_checkpointing_function:
            temp_restore_file_path = tempfile.NamedTemporaryFile()
            # State file is written to allow for a restore to happen.
            tmp_file_name = temp_restore_file_path.name
            with pathlib.Path(tmp_file_name).open("w") as target:
                json.dump({}, target)
            env["MODAL_RESTORE_STATE_PATH"] = tmp_file_name
            # Override server URL to reproduce restore behavior.
            env["MODAL_ENABLE_SNAP_RESTORE"] = "1"

        # reset _App tracking state between runs
        _App._all_apps.clear()

        try:
            with mock.patch.dict(os.environ, env):
                main(container_args, client)
        except UserException:
            # Handle it gracefully
            pass

        # Flatten outputs
        items = _flatten_outputs(servicer.container_outputs)

        data_chunks = servicer.get_data_chunks(first_function_call_id)

        return ContainerResult(client, items, data_chunks, servicer.task_result)


def _run_container_process(
    servicer,
    tmp_path,
    module_name,
    function_name,
    *,
    inputs: list[tuple[str, tuple, dict[str, Any]]],
    max_concurrent_inputs: Optional[int] = None,
    target_concurrent_inputs: Optional[int] = None,
    cls_params: tuple[tuple, dict[str, Any]] = ((), {}),
    _print=False,  # for debugging - print directly to stdout/stderr instead of pipeing
    env={},
    is_class=False,
    function_type: "api_pb2.Function.FunctionType.ValueType" = api_pb2.Function.FUNCTION_TYPE_FUNCTION,
    volume_mounts: Optional[list[api_pb2.VolumeMount]] = None,
    http_config: Optional[api_pb2.HTTPConfig] = None,
    is_server: bool = False,
) -> subprocess.Popen:
    container_args = _container_args(
        module_name,
        function_name,
        max_concurrent_inputs=max_concurrent_inputs,
        target_concurrent_inputs=target_concurrent_inputs,
        serialized_params=serialize(cls_params),
        is_class=is_class,
        function_type=function_type,
        volume_mounts=volume_mounts,
        http_config=http_config,
        is_server=is_server,
    )

    # These env vars are always present in containers
    env["MODAL_TASK_ID"] = "ta-123"
    env["MODAL_IS_REMOTE"] = "1"

    container_args_path = tmp_path / "container-arguments.bin"
    with container_args_path.open("wb") as f:
        f.write(container_args.SerializeToString())
    env["MODAL_CONTAINER_ARGUMENTS_PATH"] = str(container_args_path)

    servicer.container_inputs = _get_multi_inputs(inputs)

    return subprocess.Popen(
        [sys.executable, "-m", "modal._container_entrypoint"],
        env={**os.environ, **env},
        stdout=subprocess.PIPE if not _print else None,
        stderr=subprocess.PIPE if not _print else None,
    )


def _run_container_process_auto(
    servicer,
    tmp_path,
    function_name: str,
    deployed_support_function_definitions,
    *,
    inputs=None,
    serialized_params: bytes = b"",
    _print=False,
    env=None,
) -> subprocess.Popen:
    functions_dict, app_layout = deployed_support_function_definitions
    function_id, function_def = functions_dict[function_name]

    container_args = api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id=function_id,
        app_id="ap-1",
        function_def=function_def,
        checkpoint_id=f"ch-{uuid.uuid4()}",
        app_layout=app_layout,
        serialized_params=serialized_params,
    )

    container_args_path = tmp_path / "container-arguments.bin"
    with container_args_path.open("wb") as f:
        f.write(container_args.SerializeToString())

    if inputs is None:
        servicer.container_inputs = _get_multi_inputs([]) if function_def.is_class else _get_inputs()
    elif function_def.is_class or function_def.is_server:
        servicer.container_inputs = _get_multi_inputs(inputs)
    else:
        servicer.container_inputs = inputs

    if env is None:
        env = {}
    env["MODAL_TASK_ID"] = "ta-123"
    env["MODAL_IS_REMOTE"] = "1"
    env["MODAL_CONTAINER_ARGUMENTS_PATH"] = str(container_args_path)

    return subprocess.Popen(
        [sys.executable, "-m", "modal._container_entrypoint"],
        env={**os.environ, **env},
        stdout=subprocess.PIPE if not _print else None,
        stderr=subprocess.PIPE if not _print else None,
    )


# =============================================================================
# Batch Helpers
# =============================================================================
def _batch_function_test_helper(
    batch_func,
    servicer,
    args_list,
    expected_outputs,
    expected_status="success",
    batch_max_size=4,
):
    batch_wait_ms = 500
    inputs = _get_inputs_batched(args_list, batch_max_size)

    ret = _run_container(
        servicer,
        "test.supports.functions",
        batch_func,
        inputs=inputs,
        batch_max_size=batch_max_size,
        batch_wait_ms=batch_wait_ms,
    )
    if expected_status == "success":
        outputs = _unwrap_batch_scalar(ret, len(expected_outputs))
    else:
        outputs = _unwrap_batch_exception(ret, len(expected_outputs))
    assert outputs == expected_outputs


def _batch_function_test_helper_auto(
    batch_func,
    servicer,
    deployed_support_function_definitions,
    args_list,
    expected_outputs,
    expected_status="success",
    batch_max_size=4,
):
    inputs = _get_inputs_batched(args_list, batch_max_size)

    ret = _run_container_auto(
        servicer,
        batch_func,
        deployed_support_function_definitions,
        inputs=inputs,
    )
    if expected_status == "success":
        outputs = _unwrap_batch_scalar(ret, len(expected_outputs))
    else:
        outputs = _unwrap_batch_exception(ret, len(expected_outputs))
    assert outputs == expected_outputs


# =============================================================================
# Container Helpers
# =============================================================================


def _container_args(
    module_name,
    function_name,
    function_type=api_pb2.Function.FUNCTION_TYPE_FUNCTION,
    webhook_type=api_pb2.WEBHOOK_TYPE_UNSPECIFIED,
    definition_type=api_pb2.Function.DEFINITION_TYPE_FILE,
    app_name: str = "",
    is_builder_function: bool = False,
    max_concurrent_inputs: Optional[int] = None,
    target_concurrent_inputs: Optional[int] = None,
    batch_max_size: Optional[int] = None,
    batch_wait_ms: Optional[int] = None,
    serialized_params: Optional[bytes] = None,
    is_checkpointing_function: bool = False,
    deps: list[str] = ["im-1"],
    volume_mounts: Optional[list[api_pb2.VolumeMount]] = None,
    is_auto_snapshot: bool = False,
    max_inputs: Optional[int] = None,
    is_class: bool = False,
    class_parameter_info=api_pb2.ClassParameterInfo(
        format=api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_UNSPECIFIED, schema=[]
    ),
    app_id: str = "ap-1",
    app_layout: api_pb2.AppLayout = DEFAULT_APP_LAYOUT_SENTINEL,
    web_server_port: Optional[int] = None,
    web_server_startup_timeout: Optional[float] = None,
    function_serialized: Optional[bytes] = None,
    class_serialized: Optional[bytes] = None,
    supported_output_formats: Sequence["api_pb2.DataFormat.ValueType"] = [
        api_pb2.DATA_FORMAT_PICKLE,
        api_pb2.DATA_FORMAT_CBOR,
    ],
    method_definitions: dict[str, api_pb2.MethodDefinition] = {},
    http_config: Optional[api_pb2.HTTPConfig] = None,
):
    if app_layout is DEFAULT_APP_LAYOUT_SENTINEL:
        app_layout = api_pb2.AppLayout(
            objects=[
                api_pb2.Object(object_id="im-1"),
                api_pb2.Object(
                    object_id="fu-123",
                    function_handle_metadata=api_pb2.FunctionHandleMetadata(
                        function_name=function_name,
                    ),
                ),
            ],
            function_ids={function_name: "fu-123"},
        )
        if is_class:
            app_layout.objects.append(
                api_pb2.Object(object_id="cs-123", class_handle_metadata=api_pb2.ClassHandleMetadata())
            )
            app_layout.class_ids[function_name.removesuffix(".*")] = "cs-123"

    if webhook_type:
        webhook_config = api_pb2.WebhookConfig(
            type=webhook_type,
            method="GET",
            async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
            web_server_port=web_server_port,
            web_server_startup_timeout=web_server_startup_timeout,
        )
    else:
        webhook_config = None
    function_def = api_pb2.Function(
        module_name=module_name,
        function_name=function_name,
        implementation_name=function_name,
        function_type=function_type,
        volume_mounts=volume_mounts,
        webhook_config=webhook_config,
        definition_type=definition_type,
        app_name=app_name or "",
        is_builder_function=is_builder_function,
        is_auto_snapshot=is_auto_snapshot,
        target_concurrent_inputs=target_concurrent_inputs,
        max_concurrent_inputs=max_concurrent_inputs,
        batch_max_size=batch_max_size,
        batch_linger_ms=batch_wait_ms,
        is_checkpointing_function=is_checkpointing_function,
        object_dependencies=[api_pb2.ObjectDependency(object_id=object_id) for object_id in deps],
        max_inputs=max_inputs,
        is_class=is_class,
        class_parameter_info=class_parameter_info,
        function_serialized=function_serialized,
        class_serialized=class_serialized,
        supported_output_formats=supported_output_formats,
        method_definitions=method_definitions,
        http_config=http_config,
    )

    return api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        app_id=app_id,
        function_def=function_def,
        serialized_params=serialized_params,
        checkpoint_id=f"ch-{uuid.uuid4()}",
        app_layout=app_layout,
    )


# =============================================================================
# Input Helpers
# =============================================================================


def _get_inputs(
    args: tuple[tuple, dict] = ((42,), {}),
    n: int = 1,
    kill_switch=True,
    method_name: Optional[str] = None,
    upload_to_blob: bool = False,
    client: Optional[Client] = None,
    data_format: "api_pb2.DataFormat.ValueType" = api_pb2.DATA_FORMAT_PICKLE,
) -> list[api_pb2.FunctionGetInputsResponse]:
    if upload_to_blob:
        args_blob_id = blob_upload(serialize_data_format(args, data_format), client.stub)
        input_pb = api_pb2.FunctionInput(
            args_blob_id=args_blob_id, data_format=data_format, method_name=method_name or ""
        )
    else:
        input_pb = api_pb2.FunctionInput(
            args=serialize_data_format(args, data_format), data_format=data_format, method_name=method_name or ""
        )
    inputs = [
        *(
            api_pb2.FunctionGetInputsItem(input_id=f"in-xyz{i}", function_call_id="fc-123", input=input_pb)
            for i in range(n)
        ),
        *([api_pb2.FunctionGetInputsItem(kill_switch=True)] if kill_switch else []),
    ]
    return [api_pb2.FunctionGetInputsResponse(inputs=[x]) for x in inputs]


def _get_multi_inputs_with_methods(args: list[tuple[str, tuple, dict]] = []) -> list[api_pb2.FunctionGetInputsResponse]:
    responses = []
    for input_n, (method_name, *input_args) in enumerate(args):
        resp = api_pb2.FunctionGetInputsResponse(
            inputs=[
                api_pb2.FunctionGetInputsItem(
                    input_id=f"in-{input_n:03}",
                    input=api_pb2.FunctionInput(
                        args=serialize(input_args), method_name=method_name, data_format=api_pb2.DATA_FORMAT_PICKLE
                    ),
                )
            ]
        )
        responses.append(resp)

    return responses + [api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(kill_switch=True)])]


def _unwrap_scalar(ret: ContainerResult):
    assert len(ret.items) == 1
    assert ret.items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    return deserialize(ret.items[0].result.data, ret.client)


def _unwrap_blob_scalar(ret: ContainerResult, client: Client):
    assert len(ret.items) == 1
    assert ret.items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    data = blob_download(ret.items[0].result.data_blob_id, client.stub)
    return deserialize(data, ret.client)


def _unwrap_batch_scalar(ret: ContainerResult, batch_size):
    assert len(ret.items) == batch_size
    outputs = []
    for item in ret.items:
        assert item.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        outputs.append(deserialize(item.result.data, ret.client))
    assert len(outputs) == batch_size
    return outputs


def _unwrap_exception(ret: ContainerResult):
    assert len(ret.items) == 1
    assert ret.items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert "Traceback" in ret.items[0].result.traceback
    return deserialize(ret.items[0].result.data, ret.client)


def _unwrap_batch_exception(ret: ContainerResult, batch_size):
    assert len(ret.items) == batch_size
    outputs = []
    for item in ret.items:
        assert item.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
        assert "Traceback" in item.result.traceback
        outputs.append(item.result.exception)
    assert len(outputs) == batch_size
    return outputs


def _unwrap_generator(
    ret: ContainerResult, assert_data_format: Optional["api_pb2.DataFormat.ValueType"] = None
) -> tuple[list[Any], Optional[Exception]]:
    assert len(ret.items) == 1
    item = ret.items[0]

    values = []
    for chunk in ret.data_chunks:
        if assert_data_format is not None:
            assert chunk.data_format == assert_data_format
        values.append(deserialize_data_format(chunk.data, chunk.data_format, None))

    if item.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
        if item.data_format == api_pb2.DATA_FORMAT_PICKLE:
            assert assert_data_format is None or assert_data_format == api_pb2.DATA_FORMAT_PICKLE
            exc = deserialize_data_format(item.result.data, item.data_format, ret.client)
        else:
            exc = None  # no support for exceptions unless we use pickle at the moment
        return values, exc
    elif item.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
        assert item.data_format == api_pb2.DATA_FORMAT_GENERATOR_DONE
        done: api_pb2.GeneratorDone = deserialize_data_format(item.result.data, item.data_format, None)
        assert done.items_total == len(values)
        return values, None
    else:
        raise RuntimeError("unknown result type")


def _unwrap_asgi(ret: ContainerResult):
    values, exc = _unwrap_generator(ret)
    assert exc is None, "web endpoint raised exception"
    return values


def _get_web_inputs(path="/", method_name=""):
    scope = {
        "method": "GET",
        "type": "http",
        "path": path,
        "headers": {},
        "query_string": b"arg=space",
        "http_version": "2",
    }
    return _get_inputs(((scope,), {}), method_name=method_name)


def _get_inputs_batched_with_formats(
    args_list: list[tuple[tuple, dict]],
    data_formats: list["api_pb2.DataFormat.ValueType"],
    batch_max_size: int,
    kill_switch=True,
    method_name: Optional[str] = None,
):
    """Create batched inputs with different data formats per item."""
    assert len(args_list) == len(data_formats), "args_list and data_formats must have same length"
    input_pbs = []
    for args, data_format in zip(args_list, data_formats):
        serialized_args = serialize_data_format(args, data_format)
        input_pbs.append(
            api_pb2.FunctionInput(args=serialized_args, data_format=data_format, method_name=method_name or "")
        )
    inputs = [
        *(
            api_pb2.FunctionGetInputsItem(input_id=f"in-xyz{i}", function_call_id="fc-123", input=input_pb)
            for i, input_pb in enumerate(input_pbs)
        ),
        *([api_pb2.FunctionGetInputsItem(kill_switch=True)] if kill_switch else []),
    ]
    response_list = []
    current_batch: list[Any] = []
    while inputs:
        input = inputs.pop(0)
        if input.kill_switch:
            if len(current_batch) > 0:
                response_list.append(api_pb2.FunctionGetInputsResponse(inputs=current_batch))
            current_batch = [input]
            break
        if len(current_batch) > batch_max_size:
            response_list.append(api_pb2.FunctionGetInputsResponse(inputs=current_batch))
            current_batch = []
        current_batch.append(input)

    if len(current_batch) > 0:
        response_list.append(api_pb2.FunctionGetInputsResponse(inputs=current_batch))

    return response_list


def _get_inputs_batched(
    args_list: list[tuple[tuple, dict]],
    batch_max_size: int,
    kill_switch=True,
    method_name: Optional[str] = None,
):
    """Helper function to create batched inputs with PICKLE format for all items."""
    data_formats = [api_pb2.DATA_FORMAT_PICKLE] * len(args_list)
    return _get_inputs_batched_with_formats(args_list, data_formats, batch_max_size, kill_switch, method_name)


def _get_multi_inputs(args: list[tuple[str, tuple, dict]] = []) -> list[api_pb2.FunctionGetInputsResponse]:
    responses = []
    for input_n, (method_name, input_args, input_kwargs) in enumerate(args):
        resp = api_pb2.FunctionGetInputsResponse(
            inputs=[
                api_pb2.FunctionGetInputsItem(
                    function_call_id="fc-123",
                    input_id=f"in-{input_n:03}",
                    input=api_pb2.FunctionInput(
                        args=serialize((input_args, input_kwargs)),
                        method_name=method_name,
                        data_format=api_pb2.DATA_FORMAT_PICKLE,
                    ),
                )
            ]
        )
        responses.append(resp)

    return responses + [api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(kill_switch=True)])]


def _flatten_outputs(outputs) -> list[api_pb2.FunctionPutOutputsItem]:
    items: list[api_pb2.FunctionPutOutputsItem] = []
    for req in outputs:
        items += list(req.outputs)
    return items


def _unwrap_concurrent_input_outputs(n_inputs: int, n_parallel: int, ret: ContainerResult):
    # Ensure that outputs align with expectation of running concurrent inputs

    # Each group of n_parallel inputs should start together of each other
    # and different groups should start SLEEP_TIME apart.
    assert len(ret.items) == n_inputs
    for i in range(1, len(ret.items)):
        diff = ret.items[i].input_started_at - ret.items[i - 1].input_started_at
        expected_diff = SLEEP_TIME if i % n_parallel == 0 else 0
        assert diff == pytest.approx(expected_diff, abs=0.3)

    outputs = []
    for item in ret.items:
        assert item.output_created_at - item.input_started_at == pytest.approx(SLEEP_TIME, abs=0.3)
        assert item.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        outputs.append(deserialize(item.result.data, ret.client))
    return outputs
