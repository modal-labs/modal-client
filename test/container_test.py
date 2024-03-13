# Copyright Modal Labs 2022

import asyncio
import base64
import dataclasses
import json
import os
import pathlib
import pickle
import pytest
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock

from grpclib.exceptions import GRPCError

from modal import Client
from modal._container_entrypoint import UserException, main
from modal._serialization import (
    deserialize,
    deserialize_data_format,
    serialize,
    serialize_data_format,
)
from modal._utils import async_utils
from modal.exception import InvalidError
from modal.partial_function import enter
from modal.stub import _Stub
from modal_proto import api_pb2

from .helpers import deploy_stub_externally
from .supports.skip import skip_windows_signals, skip_windows_unix_socket

EXTRA_TOLERANCE_DELAY = 2.0 if sys.platform == "linux" else 5.0
FUNCTION_CALL_ID = "fc-123"
SLEEP_DELAY = 0.1


def _get_inputs(args: Tuple[Tuple, Dict] = ((42,), {}), n: int = 1) -> List[api_pb2.FunctionGetInputsResponse]:
    input_pb = api_pb2.FunctionInput(args=serialize(args), data_format=api_pb2.DATA_FORMAT_PICKLE)
    inputs = [
        *(
            api_pb2.FunctionGetInputsItem(input_id=f"in-xyz{i}", function_call_id="fc-123", input=input_pb)
            for i in range(n)
        ),
        api_pb2.FunctionGetInputsItem(kill_switch=True),
    ]
    return [api_pb2.FunctionGetInputsResponse(inputs=[x]) for x in inputs]


@dataclasses.dataclass
class ContainerResult:
    client: Client
    items: List[api_pb2.FunctionPutOutputsItem]
    data_chunks: List[api_pb2.DataChunk]
    task_result: api_pb2.GenericResult


def _get_multi_inputs(args: List[Tuple[Tuple, Dict]] = []) -> List[api_pb2.FunctionGetInputsResponse]:
    responses = []
    for input_n, input_args in enumerate(args):
        resp = api_pb2.FunctionGetInputsResponse(
            inputs=[
                api_pb2.FunctionGetInputsItem(
                    input_id=f"in-{input_n:03}", input=api_pb2.FunctionInput(args=serialize(input_args))
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
    stub_name: str = "",
    is_builder_function: bool = False,
    allow_concurrent_inputs: Optional[int] = None,
    serialized_params: Optional[bytes] = None,
    is_checkpointing_function: bool = False,
    deps: List[str] = ["im-1"],
    volume_mounts: Optional[List[api_pb2.VolumeMount]] = None,
    is_auto_snapshot: bool = False,
    max_inputs: Optional[int] = None,
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
        stub_name=stub_name or "",
        is_builder_function=is_builder_function,
        is_auto_snapshot=is_auto_snapshot,
        allow_concurrent_inputs=allow_concurrent_inputs,
        is_checkpointing_function=is_checkpointing_function,
        object_dependencies=[api_pb2.ObjectDependency(object_id=object_id) for object_id in deps],
        max_inputs=max_inputs,
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
    stub_name: str = "",
    is_builder_function: bool = False,
    allow_concurrent_inputs: Optional[int] = None,
    serialized_params: Optional[bytes] = None,
    is_checkpointing_function: bool = False,
    deps: List[str] = ["im-1"],
    volume_mounts: Optional[List[api_pb2.VolumeMount]] = None,
    is_auto_snapshot: bool = False,
    max_inputs: Optional[int] = None,
) -> ContainerResult:
    container_args = _container_args(
        module_name,
        function_name,
        function_type,
        webhook_type,
        definition_type,
        stub_name,
        is_builder_function,
        allow_concurrent_inputs,
        serialized_params,
        is_checkpointing_function,
        deps,
        volume_mounts,
        is_auto_snapshot,
        max_inputs,
    )
    with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret")) as client:
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
            env["MODAL_SERVER_URL"] = servicer.remote_addr

        # reset _Stub tracking state between runs
        _Stub._all_stubs = {}

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


@skip_windows_unix_socket
def test_success(unix_servicer, event_loop):
    t0 = time.time()
    ret = _run_container(unix_servicer, "modal_test_support.functions", "square")
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY
    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_generator_success(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "gen_n",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )

    items, exc = _unwrap_generator(ret)
    assert items == [i**2 for i in range(42)]
    assert exc is None


@skip_windows_unix_socket
def test_generator_failure(unix_servicer, event_loop):
    inputs = _get_inputs(((10, 5), {}))
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "gen_n_fail_on_m",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
        inputs=inputs,
    )
    items, exc = _unwrap_generator(ret)
    assert items == [i**2 for i in range(5)]
    assert isinstance(exc, Exception)
    assert exc.args == ("bad",)


@skip_windows_unix_socket
def test_async(unix_servicer):
    t0 = time.time()
    ret = _run_container(unix_servicer, "modal_test_support.functions", "square_async")
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY
    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_failure(unix_servicer):
    ret = _run_container(unix_servicer, "modal_test_support.functions", "raises")
    assert _unwrap_exception(ret) == "Exception('Failure!')"


@skip_windows_unix_socket
def test_raises_base_exception(unix_servicer):
    ret = _run_container(unix_servicer, "modal_test_support.functions", "raises_sysexit")
    assert _unwrap_exception(ret) == "SystemExit(1)"


@skip_windows_unix_socket
def test_keyboardinterrupt(unix_servicer):
    with pytest.raises(KeyboardInterrupt):
        _run_container(unix_servicer, "modal_test_support.functions", "raises_keyboardinterrupt")


@skip_windows_unix_socket
def test_rate_limited(unix_servicer, event_loop):
    t0 = time.time()
    unix_servicer.rate_limit_sleep_duration = 0.25
    ret = _run_container(unix_servicer, "modal_test_support.functions", "square")
    assert 0.25 <= time.time() - t0 < 0.25 + EXTRA_TOLERANCE_DELAY
    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_grpc_failure(unix_servicer, event_loop):
    # An error in "Modal code" should cause the entire container to fail
    with pytest.raises(GRPCError):
        _run_container(
            unix_servicer,
            "modal_test_support.functions",
            "square",
            fail_get_inputs=True,
        )

    # assert unix_servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    # assert "GRPCError" in unix_servicer.task_result.exception


@skip_windows_unix_socket
def test_missing_main_conditional(unix_servicer, event_loop):
    _run_container(unix_servicer, "modal_test_support.missing_main_conditional", "square")

    assert unix_servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert "modal run" in unix_servicer.task_result.traceback

    exc = deserialize(unix_servicer.task_result.data, None)
    assert isinstance(exc, InvalidError)


@skip_windows_unix_socket
def test_startup_failure(unix_servicer, event_loop):
    _run_container(unix_servicer, "modal_test_support.startup_failure", "f")

    assert unix_servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE

    exc = deserialize(unix_servicer.task_result.data, None)
    assert isinstance(exc, ImportError)


@skip_windows_unix_socket
def test_from_local_python_packages_inside_container(unix_servicer, event_loop):
    """`from_local_python_packages` shouldn't actually collect modules inside the container, because it's possible
    that there are modules that were present locally for the user that didn't get mounted into
    all the containers."""
    ret = _run_container(unix_servicer, "modal_test_support.package_mount", "num_mounts")
    assert _unwrap_scalar(ret) == 0


def _get_web_inputs(path="/"):
    scope = {
        "method": "GET",
        "type": "http",
        "path": path,
        "headers": {},
        "query_string": b"arg=space",
        "http_version": "2",
    }
    return _get_inputs(((scope,), {}))


@async_utils.synchronize_api  # needs to be synchronized so the asyncio.Queue gets used from the same event loop as the servicer
async def _put_web_body(servicer, body: bytes):
    asgi = {"type": "http.request", "body": body, "more_body": False}
    data = serialize_data_format(asgi, api_pb2.DATA_FORMAT_ASGI)

    q = servicer.fc_data_in.setdefault("fc-123", asyncio.Queue())
    q.put_nowait(api_pb2.DataChunk(data_format=api_pb2.DATA_FORMAT_ASGI, data=data, index=1))


@skip_windows_unix_socket
def test_webhook(unix_servicer, event_loop):
    inputs = _get_web_inputs()
    _put_web_body(unix_servicer, b"")
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
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


@skip_windows_unix_socket
def test_serialized_function(unix_servicer, event_loop):
    def triple(x):
        return 3 * x

    unix_servicer.function_serialized = serialize(triple)
    ret = _run_container(
        unix_servicer,
        "foo.bar.baz",
        "f",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )
    assert _unwrap_scalar(ret) == 3 * 42


@skip_windows_unix_socket
def test_webhook_serialized(unix_servicer, event_loop):
    inputs = _get_web_inputs()
    _put_web_body(unix_servicer, b"")

    # Store a serialized webhook function on the servicer
    def webhook(arg="world"):
        return f"Hello, {arg}"

    unix_servicer.function_serialized = serialize(webhook)

    ret = _run_container(
        unix_servicer,
        "foo.bar.baz",
        "f",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )

    _, second_message = _unwrap_asgi(ret)
    assert second_message["body"] == b'"Hello, space"'  # Note: JSON-encoded


@skip_windows_unix_socket
def test_function_returning_generator(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "fun_returning_gen",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )
    items, exc = _unwrap_generator(ret)
    assert len(items) == 42


@skip_windows_unix_socket
def test_asgi(unix_servicer, event_loop):
    inputs = _get_web_inputs(path="/foo")
    _put_web_body(unix_servicer, b"")
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
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


@skip_windows_unix_socket
def test_wsgi(unix_servicer, event_loop):
    inputs = _get_web_inputs(path="/")
    _put_web_body(unix_servicer, b"my wsgi body")
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
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


@skip_windows_unix_socket
def test_webhook_streaming_sync(unix_servicer, event_loop):
    inputs = _get_web_inputs()
    _put_web_body(unix_servicer, b"")
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "webhook_streaming",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )
    data = _unwrap_asgi(ret)
    bodies = [d["body"].decode() for d in data if d.get("body")]
    assert bodies == [f"{i}..." for i in range(10)]


@skip_windows_unix_socket
def test_webhook_streaming_async(unix_servicer, event_loop):
    inputs = _get_web_inputs()
    _put_web_body(unix_servicer, b"")
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "webhook_streaming_async",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )

    data = _unwrap_asgi(ret)
    bodies = [d["body"].decode() for d in data if d.get("body")]
    assert bodies == [f"{i}..." for i in range(10)]


@skip_windows_unix_socket
def test_cls_function(unix_servicer, event_loop):
    ret = _run_container(unix_servicer, "modal_test_support.functions", "Cls.f")
    assert _unwrap_scalar(ret) == 42 * 111


@skip_windows_unix_socket
def test_lifecycle_enter_sync(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer, "modal_test_support.functions", "LifecycleCls.f_sync", inputs=_get_inputs(((False,), {}))
    )
    assert _unwrap_scalar(ret) == ["enter_sync", "enter_async", "f_sync"]


@skip_windows_unix_socket
def test_lifecycle_enter_async(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer, "modal_test_support.functions", "LifecycleCls.f_async", inputs=_get_inputs(((False,), {}))
    )
    assert _unwrap_scalar(ret) == ["enter_sync", "enter_async", "f_async"]


@skip_windows_unix_socket
def test_param_cls_function(unix_servicer, event_loop):
    serialized_params = pickle.dumps(([111], {"y": "foo"}))
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "ParamCls.f",
        serialized_params=serialized_params,
    )
    assert _unwrap_scalar(ret) == "111 foo 42"


@skip_windows_unix_socket
def test_cls_web_endpoint(unix_servicer, event_loop):
    inputs = _get_web_inputs()
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "Cls.web",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
    )

    _, second_message = _unwrap_asgi(ret)
    assert json.loads(second_message["body"]) == {"ret": "space" * 111}


@skip_windows_unix_socket
def test_serialized_cls(unix_servicer, event_loop):
    class Cls:
        @enter()
        def enter(self):
            self.power = 5

        def method(self, x):
            return x**self.power

    unix_servicer.class_serialized = serialize(Cls)
    unix_servicer.function_serialized = serialize(Cls.method)
    ret = _run_container(
        unix_servicer,
        "module.doesnt.matter",
        "function.doesnt.matter",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )
    assert _unwrap_scalar(ret) == 42**5


@skip_windows_unix_socket
def test_cls_generator(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "Cls.generator",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )
    items, exc = _unwrap_generator(ret)
    assert items == [42**3]
    assert exc is None


@skip_windows_unix_socket
def test_checkpointing_cls_function(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "CheckpointingCls.f",
        inputs=_get_inputs((("D",), {})),
        is_checkpointing_function=True,
    )
    assert any(isinstance(request, api_pb2.ContainerCheckpointRequest) for request in unix_servicer.requests)
    for request in unix_servicer.requests:
        if isinstance(request, api_pb2.ContainerCheckpointRequest):
            assert request.checkpoint_id
    assert _unwrap_scalar(ret) == "ABCD"


@skip_windows_unix_socket
def test_cls_enter_uses_event_loop(unix_servicer):
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "EventLoopCls.f",
        inputs=_get_inputs(((), {})),
    )
    assert _unwrap_scalar(ret) == True


@skip_windows_unix_socket
def test_container_heartbeats(unix_servicer, event_loop):
    _run_container(unix_servicer, "modal_test_support.functions", "square")
    assert any(isinstance(request, api_pb2.ContainerHeartbeatRequest) for request in unix_servicer.requests)


@skip_windows_unix_socket
def test_cli(unix_servicer, event_loop):
    # This tests the container being invoked as a subprocess (the if __name__ == "__main__" block)

    # Build up payload we pass through sys args
    function_def = api_pb2.Function(
        module_name="modal_test_support.functions",
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
    unix_servicer.app_objects["ap-123"] = {"": "im-123"}

    # Inputs that will be consumed by the container
    unix_servicer.container_inputs = _get_inputs()

    # Launch subprocess
    env = {"MODAL_SERVER_URL": unix_servicer.remote_addr}
    lib_dir = pathlib.Path(__file__).parent.parent
    args: List[str] = [sys.executable, "-m", "modal._container_entrypoint", data_base64]
    ret = subprocess.run(args, cwd=lib_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = ret.stdout.decode()
    stderr = ret.stderr.decode()
    if ret.returncode != 0:
        raise Exception(f"Failed with {ret.returncode} stdout: {stdout} stderr: {stderr}")

    assert stdout == ""
    assert stderr == ""


@skip_windows_unix_socket
def test_function_sibling_hydration(unix_servicer):
    deploy_stub_externally(unix_servicer, "modal_test_support.functions", "stub")
    ret = _run_container(unix_servicer, "modal_test_support.functions", "check_sibling_hydration")
    assert _unwrap_scalar(ret) is None


@skip_windows_unix_socket
def test_multistub(unix_servicer, caplog):
    deploy_stub_externally(unix_servicer, "modal_test_support.multistub", "a")
    ret = _run_container(unix_servicer, "modal_test_support.multistub", "a_func")
    assert _unwrap_scalar(ret) is None
    assert (
        len(caplog.messages) == 1
    )  # warns in case the user would use is_inside checks... Hydration should work regardless
    assert "You have more than one unnamed stub" in caplog.text


@skip_windows_unix_socket
def test_multistub_privately_decorated(unix_servicer, caplog):
    # function handle does not override the original function, so we can't find the stub
    # and the two stubs are not named
    ret = _run_container(unix_servicer, "modal_test_support.multistub_privately_decorated", "foo")
    assert _unwrap_scalar(ret) == 1
    assert "You have more than one unnamed stub." in caplog.text


@skip_windows_unix_socket
def test_multistub_privately_decorated_named_stub(unix_servicer, caplog):
    # function handle does not override the original function, so we can't find the stub
    # but we can use the names of the stubs to determine the active stub
    ret = _run_container(
        unix_servicer,
        "modal_test_support.multistub_privately_decorated_named_stub",
        "foo",
        stub_name="dummy",
    )
    assert _unwrap_scalar(ret) == 1
    assert len(caplog.messages) == 0  # no warnings, since target stub is named


@skip_windows_unix_socket
def test_multistub_same_name_warning(unix_servicer, caplog):
    # function handle does not override the original function, so we can't find the stub
    # two stubs with the same name - warn since we won't know which one to hydrate
    ret = _run_container(
        unix_servicer,
        "modal_test_support.multistub_same_name",
        "foo",
        stub_name="dummy",
    )
    assert _unwrap_scalar(ret) == 1
    assert "You have more than one stub with the same name ('dummy')" in caplog.text


@skip_windows_unix_socket
def test_multistub_serialized_func(unix_servicer, caplog):
    # serialized functions shouldn't warn about multiple/not finding stubs, since they shouldn't load the module to begin with
    def dummy(x):
        return x

    unix_servicer.function_serialized = serialize(dummy)
    ret = _run_container(
        unix_servicer,
        "modal_test_support.multistub_serialized_func",
        "foo",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )
    assert _unwrap_scalar(ret) == 42
    assert len(caplog.messages) == 0


@skip_windows_unix_socket
def test_image_run_function_no_warn(unix_servicer, caplog):
    # builder functions currently aren't tied to any modal stub, so they shouldn't need to warn if they can't determine a stub to use
    ret = _run_container(
        unix_servicer,
        "modal_test_support.image_run_function",
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


@skip_windows_unix_socket
def test_concurrent_inputs_sync_function(unix_servicer):
    n_inputs = 18
    n_parallel = 6

    t0 = time.time()
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
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


@skip_windows_unix_socket
def test_concurrent_inputs_async_function(unix_servicer, event_loop):
    n_inputs = 18
    n_parallel = 6

    t0 = time.time()
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
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


@skip_windows_unix_socket
def test_unassociated_function(unix_servicer, event_loop):
    ret = _run_container(unix_servicer, "modal_test_support.functions", "unassociated_function")
    assert _unwrap_scalar(ret) == 58


@skip_windows_unix_socket
def test_param_cls_function_calling_local(unix_servicer, event_loop):
    serialized_params = pickle.dumps(([111], {"y": "foo"}))
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "ParamCls.g",
        serialized_params=serialized_params,
    )
    assert _unwrap_scalar(ret) == "111 foo 42"


@skip_windows_unix_socket
def test_derived_cls(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "DerivedCls.run",
        inputs=_get_inputs(((3,), {})),
    )
    assert _unwrap_scalar(ret) == 6


@skip_windows_unix_socket
def test_call_function_that_calls_function(unix_servicer, event_loop):
    deploy_stub_externally(unix_servicer, "modal_test_support.functions", "stub")
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "cube",
        inputs=_get_inputs(((42,), {})),
    )
    assert _unwrap_scalar(ret) == 42**3


@skip_windows_unix_socket
def test_call_function_that_calls_method(unix_servicer, event_loop):
    deploy_stub_externally(unix_servicer, "modal_test_support.functions", "stub")
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "function_calling_method",
        inputs=_get_inputs(((42, "abc", 123), {})),
    )
    assert _unwrap_scalar(ret) == 123**2  # servicer's implementation of function calling


@skip_windows_unix_socket
def test_checkpoint_and_restore_success(unix_servicer, event_loop):
    """Functions send a checkpointing request and continue to execute normally,
    simulating a restore operation."""
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "square",
        is_checkpointing_function=True,
    )
    assert any(isinstance(request, api_pb2.ContainerCheckpointRequest) for request in unix_servicer.requests)
    for request in unix_servicer.requests:
        if isinstance(request, api_pb2.ContainerCheckpointRequest):
            assert request.checkpoint_id

    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_volume_commit_on_exit(unix_servicer, event_loop):
    volume_mounts = [
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-123", allow_background_commits=True),
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-456", allow_background_commits=True),
    ]
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "square",
        volume_mounts=volume_mounts,
    )
    volume_commit_rpcs = [r for r in unix_servicer.requests if isinstance(r, api_pb2.VolumeCommitRequest)]
    assert volume_commit_rpcs
    assert {"vo-123", "vo-456"} == set(r.volume_id for r in volume_commit_rpcs)
    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_volume_commit_on_error(unix_servicer, event_loop):
    volume_mounts = [
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-foo", allow_background_commits=True),
        api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-bar", allow_background_commits=True),
    ]
    _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "raises",
        volume_mounts=volume_mounts,
    )
    volume_commit_rpcs = [r for r in unix_servicer.requests if isinstance(r, api_pb2.VolumeCommitRequest)]
    assert {"vo-foo", "vo-bar"} == set(r.volume_id for r in volume_commit_rpcs)


@skip_windows_unix_socket
def test_no_volume_commit_on_exit(unix_servicer, event_loop):
    volume_mounts = [api_pb2.VolumeMount(mount_path="/var/foo", volume_id="vo-999", allow_background_commits=False)]
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "square",
        volume_mounts=volume_mounts,
    )
    volume_commit_rpcs = [r for r in unix_servicer.requests if isinstance(r, api_pb2.VolumeCommitRequest)]
    assert not volume_commit_rpcs  # No volume commit on exit for legacy volumes
    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_volume_commit_on_exit_doesnt_fail_container(unix_servicer, event_loop):
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
        unix_servicer,
        "modal_test_support.functions",
        "square",
        volume_mounts=volume_mounts,
    )
    volume_commit_rpcs = [r for r in unix_servicer.requests if isinstance(r, api_pb2.VolumeCommitRequest)]
    assert len(volume_commit_rpcs) == 3
    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_function_dep_hydration(unix_servicer):
    deploy_stub_externally(unix_servicer, "modal_test_support.functions", "stub")
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "check_dep_hydration",
        deps=["im-1", "vo-1", "im-1", "im-2", "vo-1", "vo-2"],
    )
    assert _unwrap_scalar(ret) is None


@skip_windows_unix_socket
def test_build_decorator_cls(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "BuildCls.build1",
        inputs=_get_inputs(((), {})),
        is_builder_function=True,
        is_auto_snapshot=True,
    )
    print(ret)
    assert _unwrap_scalar(ret) == 101
    # TODO: this is GENERIC_STATUS_FAILURE when `@exit` fails,
    # but why is it not set when `@exit` is successful?
    # assert ret.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert ret.task_result is None


@skip_windows_unix_socket
def test_multiple_build_decorator_cls(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "BuildCls.build2",
        inputs=_get_inputs(((), {})),
        is_builder_function=True,
        is_auto_snapshot=True,
    )
    assert _unwrap_scalar(ret) == 1001
    assert ret.task_result is None


@skip_windows_unix_socket
@pytest.mark.timeout(10.0)
def test_function_io_doesnt_inspect_args_or_return_values(monkeypatch, unix_servicer):
    synchronizer = async_utils.synchronizer

    # set up spys to track synchronicity calls to _translate_scalar_in/out
    translate_in_spy = MagicMock(wraps=synchronizer._translate_scalar_in)
    monkeypatch.setattr(synchronizer, "_translate_scalar_in", translate_in_spy)
    translate_out_spy = MagicMock(wraps=synchronizer._translate_scalar_out)
    monkeypatch.setattr(synchronizer, "_translate_scalar_out", translate_out_spy)

    # don't do blobbing for this test
    monkeypatch.setattr("modal._container_entrypoint.MAX_OBJECT_SIZE_BYTES", 1e100)

    large_data_list = list(range(int(1e6)))  # large data set

    t0 = time.perf_counter()
    # pr = cProfile.Profile()
    # pr.enable()
    _run_container(
        unix_servicer,
        "modal_test_support.functions",
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
    inputs: List[Tuple[Tuple[Any], Dict[str, Any]]],
    allow_concurrent_inputs: Optional[int] = None,
) -> subprocess.Popen:
    container_args = _container_args(module_name, function_name, allow_concurrent_inputs=allow_concurrent_inputs)
    encoded_container_args = base64.b64encode(container_args.SerializeToString())
    servicer.container_inputs = _get_multi_inputs(inputs)
    return subprocess.Popen(
        [sys.executable, "-m", "modal._container_entrypoint", encoded_container_args],
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@skip_windows_signals
@pytest.mark.usefixtures("server_url_env")
@pytest.mark.parametrize(
    ["function_name", "input_args", "cancelled_input_ids", "expected_container_output"],
    [
        # the 10 second inputs here are to be cancelled:
        ("delay", [0.01, 20, 0.02], ["in-001"], [0.01, 0.02]),  # cancel second input
        ("delay_async", [0.01, 20, 0.02], ["in-001"], [0.01, 0.02]),  # async variant
        # cancel first input, but it has already been processed, so all three should come through:
        (
            "delay",
            [0.01, 0.5, 0.03],
            ["in-000"],
            [0.01, 0.5, 0.03],
        ),
        (
            "delay_async",
            [0.01, 0.5, 0.03],
            ["in-000"],
            [0.01, 0.5, 0.03],
        ),
    ],
)
def test_cancellation_aborts_current_input_on_match(
    servicer, function_name, input_args, cancelled_input_ids, expected_container_output
):
    # NOTE: for a cancellation to actually happen in this test, it needs to be
    #    triggered while the relevant input is being processed. A future input
    #    would not be cancelled, since those are expected to be handled by
    #    the backend
    with servicer.input_lockstep() as input_lock:
        container_process = _run_container_process(
            servicer, "modal_test_support.functions", function_name, inputs=[((arg,), {}) for arg in input_args]
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
    assert container_process.wait() == 0  # wait for container to exit
    duration = time.monotonic() - t0  # time from heartbeat to container exit

    items = _flatten_outputs(servicer.container_outputs)
    assert len(items) == len(expected_container_output)
    data = [deserialize(i.result.data, client=None) for i in items]
    assert data == expected_container_output
    # should never run for ~20s, which is what the input would take if the sleep isn't interrupted
    assert duration < 10  # should typically be < 1s, but for some reason in gh actions, it takes a really long time!


@skip_windows_signals
@pytest.mark.usefixtures("server_url_env")
@pytest.mark.parametrize(
    ["function_name"],
    [("delay",), ("delay_async",)],
)
def test_cancellation_stops_task_with_concurrent_inputs(servicer, function_name):
    # send three inputs in container: in-100, in-101, in-102
    with servicer.input_lockstep() as input_lock:
        container_process = _run_container_process(
            servicer, "modal_test_support.functions", function_name, inputs=[((20,), {})], allow_concurrent_inputs=2
        )
        input_lock.wait()

    time.sleep(0.05)  # let the container get and start processing the input
    servicer.container_heartbeat_return_now(
        api_pb2.ContainerHeartbeatResponse(cancel_input_event=api_pb2.CancelInputEvent(input_ids=["in-000"]))
    )
    # container should exit soon!
    exit_code = container_process.wait(5)
    assert exit_code == 0  # container should exit gracefully


@skip_windows_signals
@pytest.mark.usefixtures("server_url_env")
def test_lifecycle_full(servicer):
    # Sync and async container lifecycle methods on a sync function.
    container_process = _run_container_process(
        servicer, "modal_test_support.functions", "LifecycleCls.f_sync", inputs=[((True,), {})]
    )
    stdout, _ = container_process.communicate(timeout=5)
    assert container_process.returncode == 0
    assert "[events:enter_sync,enter_async,f_sync,exit_sync,exit_async]" in stdout.decode()

    # Sync and async container lifecycle methods on an async function.
    container_process = _run_container_process(
        servicer, "modal_test_support.functions", "LifecycleCls.f_async", inputs=[((True,), {})]
    )
    stdout, _ = container_process.communicate(timeout=5)
    assert container_process.returncode == 0
    assert "[events:enter_sync,enter_async,f_async,exit_sync,exit_async]" in stdout.decode()
