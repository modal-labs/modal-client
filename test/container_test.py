# Copyright Modal Labs 2022

import asyncio
import base64
import dataclasses
import json
import os
import pathlib
import pickle
import pytest
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock

from grpclib import Status
from grpclib.exceptions import GRPCError

import modal
from modal import Client, Queue, Volume, is_local
from modal._container_entrypoint import UserException, main
from modal._serialization import (
    deserialize,
    deserialize_data_format,
    serialize,
    serialize_data_format,
)
from modal._utils import async_utils
from modal._utils.blob_utils import MAX_OBJECT_SIZE_BYTES
from modal.app import _App
from modal.exception import InvalidError
from modal.partial_function import enter, method
from modal_proto import api_pb2

from .helpers import deploy_app_externally
from .supports.skip import skip_github_non_linux

EXTRA_TOLERANCE_DELAY = 2.0 if sys.platform == "linux" else 5.0
FUNCTION_CALL_ID = "fc-123"
SLEEP_DELAY = 0.1


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


def _get_multi_inputs(args: List[Tuple[str, Tuple, Dict]] = []) -> List[api_pb2.FunctionGetInputsResponse]:
    responses = []
    for input_n, (method_name, input_args, input_kwargs) in enumerate(args):
        resp = api_pb2.FunctionGetInputsResponse(
            inputs=[
                api_pb2.FunctionGetInputsItem(
                    input_id=f"in-{input_n:03}",
                    input=api_pb2.FunctionInput(args=serialize((input_args, input_kwargs)), method_name=method_name),
                )
            ]
        )
        responses.append(resp)

    return responses + [api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(kill_switch=True)])]


@dataclasses.dataclass
class ContainerResult:
    client: Client
    items: List[api_pb2.FunctionPutOutputsItem]
    data_chunks: List[api_pb2.DataChunk]
    task_result: api_pb2.GenericResult


def _get_multi_inputs_with_methods(args: List[Tuple[str, Tuple, Dict]] = []) -> List[api_pb2.FunctionGetInputsResponse]:
    responses = []
    for input_n, (method_name, *input_args) in enumerate(args):
        resp = api_pb2.FunctionGetInputsResponse(
            inputs=[
                api_pb2.FunctionGetInputsItem(
                    input_id=f"in-{input_n:03}",
                    input=api_pb2.FunctionInput(args=serialize(input_args), method_name=method_name),
                )
            ]
        )
        responses.append(resp)

    return responses + [api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(kill_switch=True)])]


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
    class_parameter_info=api_pb2.ClassParameterInfo(
        format=api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_UNSPECIFIED, schema=[]
    ),
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
        class_parameter_info=class_parameter_info,
    )

    return api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        app_id="ap-1",
        function_def=function_def,
        serialized_params=serialized_params,
        checkpoint_id=f"ch-{uuid.uuid4()}",
    )


def _flatten_outputs(outputs) -> List[api_pb2.FunctionPutOutputsItem]:
    items: List[api_pb2.FunctionPutOutputsItem] = []
    for req in outputs:
        items += list(req.outputs)
    return items


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
    class_parameter_info=api_pb2.ClassParameterInfo(
        format=api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_UNSPECIFIED, schema=[]
    ),
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
        class_parameter_info=class_parameter_info,
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


def _unwrap_scalar(ret: ContainerResult):
    assert len(ret.items) == 1
    assert ret.items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    return deserialize(ret.items[0].result.data, ret.client)


def _unwrap_exception(ret: ContainerResult):
    assert len(ret.items) == 1
    assert ret.items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert "Traceback" in ret.items[0].result.traceback
    return ret.items[0].result.exception


def _unwrap_generator(ret: ContainerResult) -> Tuple[List[Any], Optional[Exception]]:
    assert len(ret.items) == 1
    item = ret.items[0]
    assert item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_UNSPECIFIED

    values: List[Any] = [deserialize_data_format(chunk.data, chunk.data_format, None) for chunk in ret.data_chunks]

    if item.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
        exc = deserialize(item.result.data, ret.client)
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


@skip_github_non_linux
def test_success(servicer, event_loop):
    t0 = time.time()
    ret = _run_container(servicer, "test.supports.functions", "square")
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY
    assert _unwrap_scalar(ret) == 42**2


@skip_github_non_linux
def test_generator_success(servicer, event_loop):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "gen_n",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )

    items, exc = _unwrap_generator(ret)
    assert items == [i**2 for i in range(42)]
    assert exc is None


@skip_github_non_linux
def test_generator_failure(servicer, capsys):
    inputs = _get_inputs(((10, 5), {}))
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "gen_n_fail_on_m",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
        inputs=inputs,
    )
    items, exc = _unwrap_generator(ret)
    assert items == [i**2 for i in range(5)]
    assert isinstance(exc, Exception)
    assert exc.args == ("bad",)
    assert 'raise Exception("bad")' in capsys.readouterr().err


@skip_github_non_linux
def test_async(servicer):
    t0 = time.time()
    ret = _run_container(servicer, "test.supports.functions", "square_async")
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY
    assert _unwrap_scalar(ret) == 42**2


@skip_github_non_linux
def test_failure(servicer, capsys):
    ret = _run_container(servicer, "test.supports.functions", "raises")
    assert _unwrap_exception(ret) == "Exception('Failure!')"
    assert 'raise Exception("Failure!")' in capsys.readouterr().err  # traceback


@skip_github_non_linux
def test_raises_base_exception(servicer, capsys):
    ret = _run_container(servicer, "test.supports.functions", "raises_sysexit")
    assert _unwrap_exception(ret) == "SystemExit(1)"
    assert "raise SystemExit(1)" in capsys.readouterr().err  # traceback


@skip_github_non_linux
def test_keyboardinterrupt(servicer):
    with pytest.raises(KeyboardInterrupt):
        _run_container(servicer, "test.supports.functions", "raises_keyboardinterrupt")


@skip_github_non_linux
def test_rate_limited(servicer, event_loop):
    t0 = time.time()
    servicer.rate_limit_sleep_duration = 0.25
    ret = _run_container(servicer, "test.supports.functions", "square")
    assert 0.25 <= time.time() - t0 < 0.25 + EXTRA_TOLERANCE_DELAY
    assert _unwrap_scalar(ret) == 42**2


@skip_github_non_linux
def test_grpc_failure(servicer, event_loop):
    # An error in "Modal code" should cause the entire container to fail
    with pytest.raises(GRPCError):
        _run_container(
            servicer,
            "test.supports.functions",
            "square",
            fail_get_inputs=True,
        )

    # assert servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    # assert "GRPCError" in servicer.task_result.exception


@skip_github_non_linux
def test_missing_main_conditional(servicer, capsys):
    _run_container(servicer, "test.supports.missing_main_conditional", "square")
    output = capsys.readouterr()
    assert "Can not run an app from within a container" in output.err

    assert servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert "modal run" in servicer.task_result.traceback

    exc = deserialize(servicer.task_result.data, None)
    assert isinstance(exc, InvalidError)


@skip_github_non_linux
def test_startup_failure(servicer, capsys):
    _run_container(servicer, "test.supports.startup_failure", "f")

    assert servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE

    exc = deserialize(servicer.task_result.data, None)
    assert isinstance(exc, ImportError)
    assert "ModuleNotFoundError: No module named 'nonexistent_package'" in capsys.readouterr().err


@skip_github_non_linux
def test_from_local_python_packages_inside_container(servicer):
    """`from_local_python_packages` shouldn't actually collect modules inside the container, because it's possible
    that there are modules that were present locally for the user that didn't get mounted into
    all the containers."""
    ret = _run_container(servicer, "test.supports.package_mount", "num_mounts")
    assert _unwrap_scalar(ret) == 0


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


# needs to be synchronized so the asyncio.Queue gets used from the same event loop as the servicer
@async_utils.synchronize_api
async def _put_web_body(servicer, body: bytes):
    asgi = {"type": "http.request", "body": body, "more_body": False}
    data = serialize_data_format(asgi, api_pb2.DATA_FORMAT_ASGI)

    q = servicer.fc_data_in.setdefault("fc-123", asyncio.Queue())
    q.put_nowait(api_pb2.DataChunk(data_format=api_pb2.DATA_FORMAT_ASGI, data=data, index=1))


@skip_github_non_linux
def test_webhook(servicer):
    inputs = _get_web_inputs()
    _put_web_body(servicer, b"")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "webhook",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
    )
    items = _unwrap_asgi(ret)

    # There should be one message for the header, one for the body, one for the EOF
    first_message, second_message = items  # _unwrap_asgi ignores the eof

    # Check the headers
    assert first_message["status"] == 200
    headers = dict(first_message["headers"])
    assert headers[b"content-type"] == b"application/json"

    # Check body
    assert json.loads(second_message["body"]) == {"hello": "space"}


@skip_github_non_linux
def test_webhook_setup_failure(servicer):
    inputs = _get_web_inputs()
    _put_web_body(servicer, b"")
    with servicer.intercept() as ctx:
        ret = _run_container(
            servicer,
            "test.supports.functions",
            "error_in_asgi_setup",
            inputs=inputs,
            webhook_type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
        )

    task_result_request: api_pb2.TaskResultRequest
    (task_result_request,) = ctx.get_requests("TaskResult")
    assert task_result_request.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert "Error while setting up asgi app" in ret.task_result.exception
    assert ret.items == []
    # TODO: We should send some kind of 500 error back to modal-http here when the container can't start up


@skip_github_non_linux
def test_serialized_function(servicer):
    def triple(x):
        return 3 * x

    servicer.function_serialized = serialize(triple)
    ret = _run_container(
        servicer,
        "foo.bar.baz",
        "f",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )
    assert _unwrap_scalar(ret) == 3 * 42


@skip_github_non_linux
def test_webhook_serialized(servicer):
    inputs = _get_web_inputs()
    _put_web_body(servicer, b"")

    # Store a serialized webhook function on the servicer
    def webhook(arg="world"):
        return f"Hello, {arg}"

    servicer.function_serialized = serialize(webhook)

    ret = _run_container(
        servicer,
        "foo.bar.baz",
        "f",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )

    _, second_message = _unwrap_asgi(ret)
    assert second_message["body"] == b'"Hello, space"'  # Note: JSON-encoded


@skip_github_non_linux
def test_function_returning_generator(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "fun_returning_gen",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )
    items, exc = _unwrap_generator(ret)
    assert len(items) == 42


@skip_github_non_linux
def test_asgi(servicer):
    inputs = _get_web_inputs(path="/foo")
    _put_web_body(servicer, b"")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "fastapi_app",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
    )

    # There should be one message for the header, and one for the body
    first_message, second_message = _unwrap_asgi(ret)

    # Check the headers
    assert first_message["status"] == 200
    headers = dict(first_message["headers"])
    assert headers[b"content-type"] == b"application/json"

    # Check body
    assert json.loads(second_message["body"]) == {"hello": "space"}


@skip_github_non_linux
def test_wsgi(servicer):
    inputs = _get_web_inputs(path="/")
    _put_web_body(servicer, b"my wsgi body")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "basic_wsgi_app",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_WSGI_APP,
    )

    # There should be one message for headers, one for the body, and one for the end-of-body.
    first_message, second_message, third_message = _unwrap_asgi(ret)

    # Check the headers
    assert first_message["status"] == 200
    headers = dict(first_message["headers"])
    assert headers[b"content-type"] == b"text/plain; charset=utf-8"

    # Check body
    assert second_message["body"] == b"got body: my wsgi body"
    assert second_message.get("more_body", False) is True
    assert third_message["body"] == b""
    assert third_message.get("more_body", False) is False


@skip_github_non_linux
def test_webhook_streaming_sync(servicer):
    inputs = _get_web_inputs()
    _put_web_body(servicer, b"")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "webhook_streaming",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )
    data = _unwrap_asgi(ret)
    bodies = [d["body"].decode() for d in data if d.get("body")]
    assert bodies == [f"{i}..." for i in range(10)]


@skip_github_non_linux
def test_webhook_streaming_async(servicer):
    inputs = _get_web_inputs()
    _put_web_body(servicer, b"")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "webhook_streaming_async",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )

    data = _unwrap_asgi(ret)
    bodies = [d["body"].decode() for d in data if d.get("body")]
    assert bodies == [f"{i}..." for i in range(10)]


@skip_github_non_linux
def test_cls_function(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "Cls.*",
        is_class=True,
        inputs=_get_inputs(method_name="f"),
    )
    assert _unwrap_scalar(ret) == 42 * 111


@skip_github_non_linux
def test_lifecycle_enter_sync(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "LifecycleCls.*",
        inputs=_get_inputs(((), {}), method_name="f_sync"),
        is_class=True,
    )
    assert _unwrap_scalar(ret) == ["enter_sync", "enter_async", "f_sync", "local"]


@skip_github_non_linux
def test_lifecycle_enter_async(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "LifecycleCls.*",
        inputs=_get_inputs(((), {}), method_name="f_async"),
        is_class=True,
    )
    assert _unwrap_scalar(ret) == ["enter_sync", "enter_async", "f_async", "local"]


@skip_github_non_linux
def test_param_cls_function(servicer):
    serialized_params = pickle.dumps(([111], {"y": "foo"}))
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "ParamCls.*",
        serialized_params=serialized_params,
        is_class=True,
        inputs=_get_inputs(method_name="f"),
    )
    assert _unwrap_scalar(ret) == "111 foo 42"


@skip_github_non_linux
def test_param_cls_function_strict_params(servicer):
    schema = [
        api_pb2.ClassParameterSpec(name="x", type=api_pb2.PARAM_TYPE_INT),
        api_pb2.ClassParameterSpec(name="y", type=api_pb2.PARAM_TYPE_STRING),
    ]
    serialized_params = modal._serialization.serialize_proto_params({"x": 111, "y": "foo"}, schema)
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "ParamCls.*",
        serialized_params=serialized_params,
        is_class=True,
        inputs=_get_inputs(method_name="f"),
        class_parameter_info=api_pb2.ClassParameterInfo(
            format=api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PROTO,
            schema=[
                api_pb2.ClassParameterSpec(name="x", type=api_pb2.PARAM_TYPE_INT),
                api_pb2.ClassParameterSpec(name="y", type=api_pb2.PARAM_TYPE_STRING),
            ],
        ),
    )
    assert _unwrap_scalar(ret) == "111 foo 42"


@skip_github_non_linux
def test_cls_web_endpoint(servicer):
    inputs = _get_web_inputs(method_name="web")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "Cls.*",
        inputs=inputs,
        is_class=True,
    )

    _, second_message = _unwrap_asgi(ret)
    assert json.loads(second_message["body"]) == {"ret": "space" * 111}


@skip_github_non_linux
def test_cls_web_asgi_construction(servicer):
    servicer.app_objects.setdefault("ap-1", {}).setdefault("square", "fu-2")
    servicer.app_functions["fu-2"] = api_pb2.FunctionHandleMetadata()

    inputs = _get_web_inputs(method_name="asgi_web")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "Cls.*",
        inputs=inputs,
        is_class=True,
    )

    _, second_message = _unwrap_asgi(ret)
    return_dict = json.loads(second_message["body"])
    assert return_dict == {
        "arg": "space",
        "at_construction": 111,  # @enter should have run when the asgi app constructor is called
        "at_runtime": 111,
        "other_hydrated": True,
    }


@skip_github_non_linux
def test_serialized_cls(servicer):
    class Cls:
        @enter()
        def enter(self):
            self.power = 5

        @method()
        def method(self, x):
            return x**self.power

    servicer.class_serialized = serialize(Cls)
    servicer.function_serialized = serialize(
        {"method": Cls.__dict__["method"]}
    )  # can't use Cls.method because of descriptor protocol that returns Function instead of PartialFunction
    ret = _run_container(
        servicer,
        "module.doesnt.matter",
        "function.doesnt.matter",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
        is_class=True,
        inputs=_get_inputs(method_name="method"),
    )
    assert _unwrap_scalar(ret) == 42**5


@skip_github_non_linux
def test_cls_generator(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "Cls.*",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
        is_class=True,
        inputs=_get_inputs(method_name="generator"),
    )
    items, exc = _unwrap_generator(ret)
    assert items == [42**3]
    assert exc is None


@skip_github_non_linux
def test_checkpointing_cls_function(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "SnapshottingCls.*",
        inputs=_get_inputs((("D",), {}), method_name="f"),
        is_checkpointing_function=True,
        is_class=True,
    )
    assert any(isinstance(request, api_pb2.ContainerCheckpointRequest) for request in servicer.requests)
    for request in servicer.requests:
        if isinstance(request, api_pb2.ContainerCheckpointRequest):
            assert request.checkpoint_id
    assert _unwrap_scalar(ret) == "ABCD"


@skip_github_non_linux
def test_cls_enter_uses_event_loop(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "EventLoopCls.*",
        inputs=_get_inputs(((), {}), method_name="f"),
        is_class=True,
    )
    assert _unwrap_scalar(ret) == True


@skip_github_non_linux
def test_container_heartbeats(servicer):
    _run_container(servicer, "test.supports.functions", "square")
    assert any(isinstance(request, api_pb2.ContainerHeartbeatRequest) for request in servicer.requests)

    _run_container(servicer, "test.supports.functions", "snapshotting_square")
    assert any(isinstance(request, api_pb2.ContainerHeartbeatRequest) for request in servicer.requests)


@skip_github_non_linux
def test_cli(servicer):
    # This tests the container being invoked as a subprocess (the if __name__ == "__main__" block)

    # Build up payload we pass through sys args
    function_def = api_pb2.Function(
        module_name="test.supports.functions",
        function_name="square",
        function_type=api_pb2.Function.FUNCTION_TYPE_FUNCTION,
        definition_type=api_pb2.Function.DEFINITION_TYPE_FILE,
        object_dependencies=[api_pb2.ObjectDependency(object_id="im-123")],
    )
    container_args = api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        app_id="ap-123",
        function_def=function_def,
    )
    data_base64: str = base64.b64encode(container_args.SerializeToString()).decode("ascii")

    # Needed for function hydration
    servicer.app_objects["ap-123"] = {"": "im-123"}

    # Inputs that will be consumed by the container
    servicer.container_inputs = _get_inputs()

    # Launch subprocess
    env = {"MODAL_SERVER_URL": servicer.container_addr}
    lib_dir = pathlib.Path(__file__).parent.parent
    args: List[str] = [sys.executable, "-m", "modal._container_entrypoint", data_base64]
    ret = subprocess.run(args, cwd=lib_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = ret.stdout.decode()
    stderr = ret.stderr.decode()
    if ret.returncode != 0:
        raise Exception(f"Failed with {ret.returncode} stdout: {stdout} stderr: {stderr}")

    if sys.version_info[:2] != (3, 8):  # Skip on Python 3.8 as we'll have PendingDeprecationError messages
        assert stdout == ""
        assert stderr == ""


@skip_github_non_linux
def test_function_sibling_hydration(servicer):
    deploy_app_externally(servicer, "test.supports.functions", "app", capture_output=False)
    ret = _run_container(servicer, "test.supports.functions", "check_sibling_hydration")
    assert _unwrap_scalar(ret) is None


@skip_github_non_linux
def test_multiapp(servicer, caplog):
    deploy_app_externally(servicer, "test.supports.multiapp", "a")
    ret = _run_container(servicer, "test.supports.multiapp", "a_func")
    assert _unwrap_scalar(ret) is None
    assert len(caplog.messages) == 0
    # Note that the app can be inferred from the function, even though there are multiple
    # apps present in the file


@skip_github_non_linux
def test_multiapp_privately_decorated(servicer, caplog):
    # function handle does not override the original function, so we can't find the app
    # and the two apps are not named
    ret = _run_container(servicer, "test.supports.multiapp_privately_decorated", "foo")
    assert _unwrap_scalar(ret) == 1
    assert "You have more than one unnamed app." in caplog.text


@skip_github_non_linux
def test_multiapp_privately_decorated_named_app(servicer, caplog):
    # function handle does not override the original function, so we can't find the app
    # but we can use the names of the apps to determine the active app
    ret = _run_container(
        servicer,
        "test.supports.multiapp_privately_decorated_named_app",
        "foo",
        app_name="dummy",
    )
    assert _unwrap_scalar(ret) == 1
    assert len(caplog.messages) == 0  # no warnings, since target app is named


@skip_github_non_linux
def test_multiapp_same_name_warning(servicer, caplog, capsys):
    # function handle does not override the original function, so we can't find the app
    # two apps with the same name - warn since we won't know which one to hydrate
    ret = _run_container(
        servicer,
        "test.supports.multiapp_same_name",
        "foo",
        app_name="dummy",
    )
    assert _unwrap_scalar(ret) == 1
    assert "You have more than one app with the same name ('dummy')" in caplog.text
    capsys.readouterr()


@skip_github_non_linux
def test_multiapp_serialized_func(servicer, caplog):
    # serialized functions shouldn't warn about multiple/not finding apps, since
    # they shouldn't load the module to begin with
    def dummy(x):
        return x

    servicer.function_serialized = serialize(dummy)
    ret = _run_container(
        servicer,
        "test.supports.multiapp_serialized_func",
        "foo",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )
    assert _unwrap_scalar(ret) == 42
    assert len(caplog.messages) == 0


@skip_github_non_linux
def test_image_run_function_no_warn(servicer, caplog):
    # builder functions currently aren't tied to any modal app,
    # so they shouldn't need to warn if they can't determine which app to use
    ret = _run_container(
        servicer,
        "test.supports.image_run_function",
        "builder_function",
        inputs=_get_inputs(((), {})),
        is_builder_function=True,
    )
    assert _unwrap_scalar(ret) is None
    assert len(caplog.messages) == 0


SLEEP_TIME = 0.7


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


@skip_github_non_linux
@pytest.mark.timeout(5)
def test_concurrent_inputs_sync_function(servicer):
    n_inputs = 18
    n_parallel = 6

    t0 = time.time()
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "sleep_700_sync",
        inputs=_get_inputs(n=n_inputs),
        allow_concurrent_inputs=n_parallel,
    )

    expected_execution = n_inputs / n_parallel * SLEEP_TIME
    assert expected_execution <= time.time() - t0 < expected_execution + EXTRA_TOLERANCE_DELAY
    outputs = _unwrap_concurrent_input_outputs(n_inputs, n_parallel, ret)
    for i, (squared, input_id, function_call_id) in enumerate(outputs):
        assert squared == 42**2
        assert input_id and input_id != outputs[i - 1][1]
        assert function_call_id and function_call_id == outputs[i - 1][2]


@skip_github_non_linux
def test_concurrent_inputs_async_function(servicer):
    n_inputs = 18
    n_parallel = 6

    t0 = time.time()
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "sleep_700_async",
        inputs=_get_inputs(n=n_inputs),
        allow_concurrent_inputs=n_parallel,
    )

    expected_execution = n_inputs / n_parallel * SLEEP_TIME
    assert expected_execution <= time.time() - t0 < expected_execution + EXTRA_TOLERANCE_DELAY
    outputs = _unwrap_concurrent_input_outputs(n_inputs, n_parallel, ret)
    for i, (squared, input_id, function_call_id) in enumerate(outputs):
        assert squared == 42**2
        assert input_id and input_id != outputs[i - 1][1]
        assert function_call_id and function_call_id == outputs[i - 1][2]


@skip_github_non_linux
def test_unassociated_function(servicer):
    ret = _run_container(servicer, "test.supports.functions", "unassociated_function")
    assert _unwrap_scalar(ret) == 58


@skip_github_non_linux
def test_param_cls_function_calling_local(servicer):
    serialized_params = pickle.dumps(([111], {"y": "foo"}))
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "ParamCls.*",
        serialized_params=serialized_params,
        inputs=_get_inputs(method_name="g"),
        is_class=True,
    )
    assert _unwrap_scalar(ret) == "111 foo 42"


@skip_github_non_linux
def test_derived_cls(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "DerivedCls.*",
        inputs=_get_inputs(((3,), {}), method_name="run"),
        is_class=True,
    )
    assert _unwrap_scalar(ret) == 6


@skip_github_non_linux
def test_call_function_that_calls_function(servicer):
    deploy_app_externally(servicer, "test.supports.functions", "app")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "cube",
        inputs=_get_inputs(((42,), {})),
    )
    assert _unwrap_scalar(ret) == 42**3


@skip_github_non_linux
def test_call_function_that_calls_method(servicer, set_env_client):
    # TODO (elias): Remove set_env_client fixture dependency - shouldn't need an env client here?
    deploy_app_externally(servicer, "test.supports.functions", "app")
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "function_calling_method",
        inputs=_get_inputs(((42, "abc", 123), {})),
    )
    assert _unwrap_scalar(ret) == 123**2  # servicer's implementation of function calling


@skip_github_non_linux
def test_checkpoint_and_restore_success(servicer):
    """Functions send a checkpointing request and continue to execute normally,
    simulating a restore operation."""
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "square",
        is_checkpointing_function=True,
    )
    assert any(isinstance(request, api_pb2.ContainerCheckpointRequest) for request in servicer.requests)
    for request in servicer.requests:
        if isinstance(request, api_pb2.ContainerCheckpointRequest):
            assert request.checkpoint_id

    assert _unwrap_scalar(ret) == 42**2


@skip_github_non_linux
def test_volume_commit_on_exit(servicer):
    volume_mounts = [
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-123", allow_background_commits=True),
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-456", allow_background_commits=True),
    ]
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "square",
        volume_mounts=volume_mounts,
    )
    volume_commit_rpcs = [r for r in servicer.requests if isinstance(r, api_pb2.VolumeCommitRequest)]
    assert volume_commit_rpcs
    assert {"vo-123", "vo-456"} == set(r.volume_id for r in volume_commit_rpcs)
    assert _unwrap_scalar(ret) == 42**2


@skip_github_non_linux
def test_volume_commit_on_error(servicer, capsys):
    volume_mounts = [
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-foo", allow_background_commits=True),
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-bar", allow_background_commits=True),
    ]
    _run_container(
        servicer,
        "test.supports.functions",
        "raises",
        volume_mounts=volume_mounts,
    )
    volume_commit_rpcs = [r for r in servicer.requests if isinstance(r, api_pb2.VolumeCommitRequest)]
    assert {"vo-foo", "vo-bar"} == set(r.volume_id for r in volume_commit_rpcs)
    assert 'raise Exception("Failure!")' in capsys.readouterr().err


@skip_github_non_linux
def test_no_volume_commit_on_exit(servicer):
    volume_mounts = [api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-999", allow_background_commits=False)]
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "square",
        volume_mounts=volume_mounts,
    )
    volume_commit_rpcs = [r for r in servicer.requests if isinstance(r, api_pb2.VolumeCommitRequest)]
    assert not volume_commit_rpcs  # No volume commit on exit for legacy volumes
    assert _unwrap_scalar(ret) == 42**2


@skip_github_non_linux
def test_volume_commit_on_exit_doesnt_fail_container(servicer):
    volume_mounts = [
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-999", allow_background_commits=True),
        api_pb2.VolumeMount(
            mount_path="/var/foo",
            volume_id="BAD-ID-FOR-VOL",
            allow_background_commits=True,
        ),
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vol-111", allow_background_commits=True),
    ]
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "square",
        volume_mounts=volume_mounts,
    )
    volume_commit_rpcs = [r for r in servicer.requests if isinstance(r, api_pb2.VolumeCommitRequest)]
    assert len(volume_commit_rpcs) == 3
    assert _unwrap_scalar(ret) == 42**2


@skip_github_non_linux
def test_build_decorator_cls(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        # note: builder functions are still run as standalone functions from their class service function
        "BuildCls.build1",
        inputs=_get_inputs(((), {})),
        is_builder_function=True,
        is_auto_snapshot=True,
    )
    assert _unwrap_scalar(ret) == 101
    # TODO: this is GENERIC_STATUS_FAILURE when `@exit` fails,
    # but why is it not set when `@exit` is successful?
    # assert ret.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert ret.task_result is None


@skip_github_non_linux
def test_multiple_build_decorator_cls(servicer):
    ret = _run_container(
        servicer,
        "test.supports.functions",
        # note: builder functions are still run as standalone functions from their class service function
        "BuildCls.build2",
        inputs=_get_inputs(((), {})),
        is_builder_function=True,
        is_auto_snapshot=True,
    )
    assert _unwrap_scalar(ret) == 1001
    assert ret.task_result is None


@skip_github_non_linux
@pytest.mark.timeout(10.0)
def test_function_io_doesnt_inspect_args_or_return_values(monkeypatch, servicer):
    synchronizer = async_utils.synchronizer

    # set up spys to track synchronicity calls to _translate_scalar_in/out
    translate_in_spy = MagicMock(wraps=synchronizer._translate_scalar_in)
    monkeypatch.setattr(synchronizer, "_translate_scalar_in", translate_in_spy)
    translate_out_spy = MagicMock(wraps=synchronizer._translate_scalar_out)
    monkeypatch.setattr(synchronizer, "_translate_scalar_out", translate_out_spy)

    # don't do blobbing for this test
    monkeypatch.setattr("modal._container_io_manager.MAX_OBJECT_SIZE_BYTES", 1e100)

    large_data_list = list(range(int(1e6)))  # large data set

    t0 = time.perf_counter()
    # pr = cProfile.Profile()
    # pr.enable()
    _run_container(
        servicer,
        "test.supports.functions",
        "ident",
        inputs=_get_inputs(((large_data_list,), {})),
    )
    # pr.disable()
    # pr.print_stats()
    duration = time.perf_counter() - t0
    assert duration < 5.0  # TODO (elias): might be able to get this down significantly more by improving serialization

    # function_io_manager.serialize(large_data_list)
    in_translations = []
    out_translations = []
    for call in translate_in_spy.call_args_list:
        in_translations += list(call.args)
    for call in translate_out_spy.call_args_list:
        out_translations += list(call.args)

    assert len(in_translations) < 1000  # typically 136 or something
    assert len(out_translations) < 2000


def _run_container_process(
    servicer,
    module_name,
    function_name,
    *,
    inputs: List[Tuple[str, Tuple, Dict[str, Any]]],
    allow_concurrent_inputs: Optional[int] = None,
    cls_params: Tuple[Tuple, Dict[str, Any]] = ((), {}),
    print=False,  # for debugging - print directly to stdout/stderr instead of pipeing
    env={},
    is_class=False,
) -> subprocess.Popen:
    container_args = _container_args(
        module_name,
        function_name,
        allow_concurrent_inputs=allow_concurrent_inputs,
        serialized_params=serialize(cls_params),
        is_class=is_class,
    )
    encoded_container_args = base64.b64encode(container_args.SerializeToString())
    servicer.container_inputs = _get_multi_inputs(inputs)
    return subprocess.Popen(
        [sys.executable, "-m", "modal._container_entrypoint", encoded_container_args],
        env={**os.environ, **env},
        stdout=subprocess.PIPE if not print else None,
        stderr=subprocess.PIPE if not print else None,
    )


@skip_github_non_linux
@pytest.mark.usefixtures("server_url_env")
@pytest.mark.parametrize(
    ["function_name", "input_args", "cancelled_input_ids", "expected_container_output", "live_cancellations"],
    [
        # the 10 second inputs here are to be cancelled:
        ("delay", [0.01, 20, 0.02], ["in-001"], [0.01, 0.02], 1),  # cancel second input
        ("delay_async", [0.01, 20, 0.02], ["in-001"], [0.01, 0.02], 1),  # async variant
        # cancel first input, but it has already been processed, so all three should come through:
        ("delay", [0.01, 0.5, 0.03], ["in-000"], [0.01, 0.5, 0.03], 0),
        ("delay_async", [0.01, 0.5, 0.03], ["in-000"], [0.01, 0.5, 0.03], 0),
    ],
)
def test_cancellation_aborts_current_input_on_match(
    servicer, function_name, input_args, cancelled_input_ids, expected_container_output, live_cancellations
):
    # NOTE: for a cancellation to actually happen in this test, it needs to be
    #    triggered while the relevant input is being processed. A future input
    #    would not be cancelled, since those are expected to be handled by
    #    the backend
    with servicer.input_lockstep() as input_lock:
        container_process = _run_container_process(
            servicer,
            "test.supports.functions",
            function_name,
            inputs=[("", (arg,), {}) for arg in input_args],
        )
        time.sleep(1)
        input_lock.wait()
        input_lock.wait()
        # second input has been sent to container here
    time.sleep(0.05)  # give it a little time to start processing

    # now let container receive container heartbeat indicating there is a cancellation
    t0 = time.monotonic()
    num_prior_outputs = len(_flatten_outputs(servicer.container_outputs))
    assert num_prior_outputs == 1  # the second input shouldn't have completed yet

    servicer.container_heartbeat_return_now(
        api_pb2.ContainerHeartbeatResponse(cancel_input_event=api_pb2.CancelInputEvent(input_ids=cancelled_input_ids))
    )
    stdout, stderr = container_process.communicate()
    assert stderr.decode().count("was cancelled by a user request") == live_cancellations
    assert "Traceback" not in stderr.decode()
    assert container_process.returncode == 0  # wait for container to exit
    duration = time.monotonic() - t0  # time from heartbeat to container exit

    items = _flatten_outputs(servicer.container_outputs)
    assert len(items) == len(expected_container_output)
    data = [deserialize(i.result.data, client=None) for i in items]
    assert data == expected_container_output
    # should never run for ~20s, which is what the input would take if the sleep isn't interrupted
    assert duration < 10  # should typically be < 1s, but for some reason in gh actions, it takes a really long time!


@skip_github_non_linux
@pytest.mark.usefixtures("server_url_env")
@pytest.mark.parametrize(
    ["function_name"],
    [("delay",), ("delay_async",)],
)
def test_cancellation_stops_task_with_concurrent_inputs(servicer, function_name):
    with servicer.input_lockstep() as input_lock:
        container_process = _run_container_process(
            servicer,
            "test.supports.functions",
            function_name,
            inputs=[("", (20,), {})] * 2,  # two inputs
            allow_concurrent_inputs=2,
        )
        input_lock.wait()
        input_lock.wait()

    time.sleep(0.05)  # let the container get and start processing the input
    servicer.container_heartbeat_return_now(
        api_pb2.ContainerHeartbeatResponse(cancel_input_event=api_pb2.CancelInputEvent(input_ids=["in-000", "in-001"]))
    )
    # container should exit soon!
    exit_code = container_process.wait(5)
    assert (
        len(servicer.container_outputs) == 0
    )  # should not fail the outputs, as they would have been cancelled in backend already
    assert "Traceback" not in container_process.stderr.read().decode("utf8")
    assert exit_code == 0  # container should exit gracefully


@skip_github_non_linux
@pytest.mark.usefixtures("server_url_env")
def test_lifecycle_full(servicer):
    # Sync and async container lifecycle methods on a sync function.
    container_process = _run_container_process(
        servicer,
        "test.supports.functions",
        "LifecycleCls.*",
        inputs=[("f_sync", (), {})],
        cls_params=((True,), {}),
        is_class=True,
    )
    stdout, _ = container_process.communicate(timeout=5)
    assert container_process.returncode == 0
    assert "[events:enter_sync,enter_async,f_sync,local,exit_sync,exit_async]" in stdout.decode()

    # Sync and async container lifecycle methods on an async function.
    container_process = _run_container_process(
        servicer,
        "test.supports.functions",
        "LifecycleCls.*",
        inputs=[("f_async", (), {})],
        cls_params=((True,), {}),
        is_class=True,
    )
    stdout, _ = container_process.communicate(timeout=5)
    assert container_process.returncode == 0
    assert "[events:enter_sync,enter_async,f_async,local,exit_sync,exit_async]" in stdout.decode()


## modal.experimental functionality ##


@skip_github_non_linux
def test_stop_fetching_inputs(servicer):
    ret = _run_container(
        servicer,
        "test.supports.experimental",
        "StopFetching.*",
        inputs=_get_inputs(((42,), {}), n=4, kill_switch=False, method_name="after_two"),
        is_class=True,
    )

    assert len(ret.items) == 2
    assert ret.items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS


@skip_github_non_linux
def test_container_heartbeat_survives_grpc_deadlines(servicer, caplog, monkeypatch):
    monkeypatch.setattr("modal._container_io_manager.HEARTBEAT_INTERVAL", 0.01)
    num_heartbeats = 0

    async def heartbeat_responder(servicer, stream):
        nonlocal num_heartbeats
        num_heartbeats += 1
        await stream.recv_message()
        raise GRPCError(Status.DEADLINE_EXCEEDED)

    with servicer.intercept() as ctx:
        ctx.set_responder("ContainerHeartbeat", heartbeat_responder)
        ret = _run_container(
            servicer,
            "test.supports.functions",
            "delay",
            inputs=_get_inputs(((2,), {})),
        )
        assert ret.task_result is None  # should not cause a failure result
    loop_iteration_failures = caplog.text.count("Heartbeat attempt failed")
    assert "Traceback" not in caplog.text  # should not print a full traceback - don't scare users!
    assert (
        loop_iteration_failures > 1
    )  # one occurence per failing `retry_transient_errors()`, so fewer than the number of failing requests!
    assert loop_iteration_failures < num_heartbeats
    assert num_heartbeats > 4  # more than the default number of retries per heartbeat attempt + 1


@skip_github_non_linux
def test_container_heartbeat_survives_local_exceptions(servicer, caplog, monkeypatch):
    numcalls = 0

    async def custom_heartbeater(self):
        nonlocal numcalls
        numcalls += 1
        raise Exception("oops")

    monkeypatch.setattr("modal._container_io_manager.HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(
        "modal._container_io_manager._ContainerIOManager._heartbeat_handle_cancellations", custom_heartbeater
    )

    ret = _run_container(
        servicer,
        "test.supports.functions",
        "delay",
        inputs=_get_inputs(((0.5,), {})),
    )
    assert ret.task_result is None  # should not cause a failure result
    loop_iteration_failures = caplog.text.count("Heartbeat attempt failed")
    assert loop_iteration_failures > 5
    assert "error=Exception('oops')" in caplog.text
    assert "Traceback" not in caplog.text  # should not print a full traceback - don't scare users!


@skip_github_non_linux
@pytest.mark.usefixtures("server_url_env")
def test_container_doesnt_send_large_exceptions(servicer):
    # Tests that large exception messages (>2mb are trimmed)
    ret = _run_container(
        servicer,
        "test.supports.functions",
        "raise_large_unicode_exception",
        inputs=_get_inputs(((), {})),
    )

    assert len(ret.items) == 1
    assert len(ret.items[0].SerializeToString()) < MAX_OBJECT_SIZE_BYTES * 1.5
    assert ret.items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert "UnicodeDecodeError" in ret.items[0].result.exception
    assert servicer.task_result is None  # should not cause a failure result


@skip_github_non_linux
@pytest.mark.usefixtures("server_url_env")
def test_sigint_termination_input_concurrent(servicer):
    # Sync and async container lifecycle methods on a sync function.
    with servicer.input_lockstep() as input_barrier:
        container_process = _run_container_process(
            servicer,
            "test.supports.functions",
            "LifecycleCls.*",
            inputs=[("delay", (10,), {})] * 3,
            cls_params=((), {"print_at_exit": True}),
            allow_concurrent_inputs=2,
            is_class=True,
        )
        input_barrier.wait()  # get one input
        input_barrier.wait()  # get one input
        time.sleep(0.5)
        # container won't be able to fetch next input
        signal_time = time.monotonic()
        os.kill(container_process.pid, signal.SIGINT)

    stdout, stderr = container_process.communicate(timeout=5)
    stop_duration = time.monotonic() - signal_time
    assert len(servicer.container_outputs) == 0
    assert (
        container_process.returncode == 0
    )  # container should catch and indicate successful termination by exiting cleanly when possible
    assert "[events:enter_sync,enter_async,delay,delay,exit_sync,exit_async]" in stdout.decode()
    assert "Traceback" not in stderr.decode()
    assert "Traceback" not in stdout.decode()
    assert stop_duration < 2.0  # if this would be ~4.5s, then the input isn't getting terminated
    assert servicer.task_result is None


@skip_github_non_linux
@pytest.mark.usefixtures("server_url_env")
@pytest.mark.parametrize("method", ["delay", "delay_async"])
def test_sigint_termination_input(servicer, method):
    # Sync and async container lifecycle methods on a sync function.
    with servicer.input_lockstep() as input_barrier:
        container_process = _run_container_process(
            servicer,
            "test.supports.functions",
            "LifecycleCls.*",
            inputs=[(method, (5,), {})],
            cls_params=((), {"print_at_exit": True}),
            is_class=True,
        )
        input_barrier.wait()  # get input
        time.sleep(0.5)
        signal_time = time.monotonic()
        os.kill(container_process.pid, signal.SIGINT)

    stdout, stderr = container_process.communicate(timeout=5)
    stop_duration = time.monotonic() - signal_time
    assert len(servicer.container_outputs) == 0
    assert (
        container_process.returncode == 0
    )  # container should catch and indicate successful termination by exiting cleanly when possible
    assert f"[events:enter_sync,enter_async,{method},exit_sync,exit_async]" in stdout.decode()
    assert "Traceback" not in stderr.decode()
    assert stop_duration < 2.0  # if this would be ~4.5s, then the input isn't getting terminated
    assert servicer.task_result is None


@skip_github_non_linux
@pytest.mark.usefixtures("server_url_env")
@pytest.mark.parametrize("enter_type", ["sync_enter", "async_enter"])
@pytest.mark.parametrize("method", ["delay", "delay_async"])
def test_sigint_termination_enter_handler(servicer, method, enter_type):
    # Sync and async container lifecycle methods on a sync function.
    container_process = _run_container_process(
        servicer,
        "test.supports.functions",
        "LifecycleCls.*",
        inputs=[(method, (5,), {})],
        cls_params=((), {"print_at_exit": True, f"{enter_type}_duration": 10}),
        is_class=True,
    )
    time.sleep(1)  # should be enough to start the enter method
    signal_time = time.monotonic()
    os.kill(container_process.pid, signal.SIGINT)
    stdout, stderr = container_process.communicate(timeout=5)
    stop_duration = time.monotonic() - signal_time
    assert len(servicer.container_outputs) == 0
    assert container_process.returncode == 0
    if enter_type == "sync_enter":
        assert "[events:enter_sync]" in stdout.decode()
    else:
        # enter_sync should run in 0s, and then we interrupt during the async enter
        assert "[events:enter_sync,enter_async]" in stdout.decode()

    assert "Traceback" not in stderr.decode()
    assert stop_duration < 2.0  # if this would be ~4.5s, then the task isn't being terminated timely
    assert servicer.task_result is None


@skip_github_non_linux
@pytest.mark.usefixtures("server_url_env")
@pytest.mark.parametrize("exit_type", ["sync_exit", "async_exit"])
def test_sigint_termination_exit_handler(servicer, exit_type):
    # Sync and async container lifecycle methods on a sync function.
    with servicer.output_lockstep() as outputs:
        container_process = _run_container_process(
            servicer,
            "test.supports.functions",
            "LifecycleCls.*",
            inputs=[("delay", (0,), {})],
            cls_params=((), {"print_at_exit": True, f"{exit_type}_duration": 2}),
            is_class=True,
        )
        outputs.wait()  # wait for first output to be emitted
    time.sleep(1)  # give some time for container to end up in the exit handler
    os.kill(container_process.pid, signal.SIGINT)

    stdout, stderr = container_process.communicate(timeout=5)

    assert len(servicer.container_outputs) == 1
    assert container_process.returncode == 0
    assert "[events:enter_sync,enter_async,delay,exit_sync,exit_async]" in stdout.decode()
    assert "Traceback" not in stderr.decode()
    assert servicer.task_result is None


@skip_github_non_linux
def test_sandbox(servicer, event_loop):
    ret = _run_container(servicer, "test.supports.functions", "sandbox_f")
    assert _unwrap_scalar(ret) == "sb-123"


@skip_github_non_linux
def test_is_local(servicer, event_loop):
    assert is_local() == True

    ret = _run_container(servicer, "test.supports.functions", "is_local_f")
    assert _unwrap_scalar(ret) == False


class Foo:
    def __init__(self, x):
        self.x = x

    @enter()
    def some_enter(self):
        self.x += "_enter"

    @method()
    def method_a(self, y):
        return self.x + f"_a_{y}"

    @method()
    def method_b(self, y):
        return self.x + f"_b_{y}"


@skip_github_non_linux
def test_class_as_service_serialized(servicer):
    # TODO(elias): refactor once the loading code is merged

    app = modal.App()
    app.cls()(Foo)  # avoid errors about methods not being turned into functions

    # Class used by the container entrypoint to instantiate the object tied to the function
    servicer.class_serialized = serialize(Foo)
    # serialized versions of each PartialFunction - used by container entrypoint to execute the methods
    servicer.function_serialized = None

    result = _run_container(
        servicer,
        "nomodule",
        "Foo.*",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
        is_class=True,
        inputs=_get_multi_inputs_with_methods([("method_a", ("x",), {}), ("method_b", ("y",), {})]),
        serialized_params=serialize((((), {"x": "s"}))),
    )
    assert len(result.items) == 2
    res_0 = result.items[0].result
    res_1 = result.items[1].result
    assert res_0.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert res_1.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert deserialize(res_0.data, result.client) == "s_enter_a_x"
    assert deserialize(res_1.data, result.client) == "s_enter_b_y"


@skip_github_non_linux
def test_function_lazy_resolution(servicer, set_env_client):
    # Deploy some global objects
    Volume.from_name("my-vol", create_if_missing=True).resolve()
    Queue.from_name("my-queue", create_if_missing=True).resolve()

    # Run container
    deploy_app_externally(servicer, "test.supports.lazy_hydration", "app", capture_output=False)
    ret = _run_container(servicer, "test.supports.lazy_hydration", "f", deps=["im-1", "vo-0"])
    assert _unwrap_scalar(ret) is None


@skip_github_non_linux
def test_no_warn_on_remote_local_volume_mount(client, servicer, recwarn, set_env_client):
    _run_container(
        servicer,
        "test.supports.volume_local",
        "volume_func_outer",
        inputs=_get_inputs(((), {})),
    )

    warnings = len(recwarn)
    for w in range(warnings):
        warning = str(recwarn.pop().message)
        assert "and will not have access to the mounted Volume or NetworkFileSystem data" not in warning
    assert len(recwarn) == 0
