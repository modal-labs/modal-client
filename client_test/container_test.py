# Copyright Modal Labs 2022
from __future__ import annotations
import base64
import json
import pathlib
import pytest
import subprocess
import sys
import time
from typing import List

from grpclib.exceptions import GRPCError

from modal._container_entrypoint import UserException, main

# from modal_test_support import SLEEP_DELAY
from modal._serialization import deserialize, serialize
from modal.client import Client
from modal.exception import InvalidError
from modal_proto import api_pb2

from .supports.skip import skip_windows_unix_socket

EXTRA_TOLERANCE_DELAY = 1.0
FUNCTION_CALL_ID = "fc-123"
SLEEP_DELAY = 0.1


def _get_inputs(args=((42,), {})) -> list[api_pb2.FunctionGetInputsResponse]:
    input_pb = api_pb2.FunctionInput(args=serialize(args))

    return [
        api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(input_id="in-xyz", input=input_pb)]),
        api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(kill_switch=True)]),
    ]


def _run_container(
    servicer,
    module_name,
    function_name,
    fail_get_inputs=False,
    inputs=None,
    function_type=api_pb2.Function.FUNCTION_TYPE_FUNCTION,
    webhook_type=api_pb2.WEBHOOK_TYPE_UNSPECIFIED,
    definition_type=api_pb2.Function.DEFINITION_TYPE_FILE,
) -> tuple[Client, list[api_pb2.FunctionPutOutputsItem]]:
    with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret")) as client:
        if inputs is None:
            servicer.container_inputs = _get_inputs()
        else:
            servicer.container_inputs = inputs
        servicer.fail_get_inputs = fail_get_inputs

        if webhook_type:
            webhook_config = api_pb2.WebhookConfig(
                type=webhook_type,
                method="GET",
                wait_for_response=True,
            )
            function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
        else:
            webhook_config = None

        function_def = api_pb2.Function(
            module_name=module_name,
            function_name=function_name,
            function_type=function_type,
            webhook_config=webhook_config,
            definition_type=definition_type,
        )

        container_args = api_pb2.ContainerArguments(
            task_id="ta-123",
            function_id="fu-123",
            app_id="se-123",
            function_def=function_def,
        )

        try:
            main(container_args, client)
        except UserException:
            # Handle it gracefully
            pass

        # Flatten outputs
        items: list[api_pb2.FunctionPutOutputsItem] = []
        for req in servicer.container_outputs:
            items += list(req.outputs)

        return client, items


@skip_windows_unix_socket
def test_success(unix_servicer, event_loop):
    t0 = time.time()
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "square")
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(42**2)


@skip_windows_unix_socket
def test_generator_success(unix_servicer, event_loop):
    client, items = _run_container(
        unix_servicer, "modal_test_support.functions", "gen_n", function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR
    )

    assert 1 <= len(items) <= 43
    assert len(items) == 43  # The generator creates N outputs, and N is 42 from the autogenerated input

    for i in range(42):
        result = items[i].result
        assert result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        assert result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
        assert deserialize(result.data, client) == i**2
        assert items[i].gen_index == i

    last_result = items[-1].result
    assert last_result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert last_result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE
    assert last_result.data == b""  # no data in generator complete marker result


@skip_windows_unix_socket
def test_generator_failure(unix_servicer, event_loop):
    inputs = _get_inputs(((10, 5), {}))
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "gen_n_fail_on_m",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
        inputs=inputs,
    )
    assert len(items) == 6  # 5 successful outputs, 1 failure

    for i in range(5):
        result = items[i].result
        assert result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        assert result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
        assert deserialize(result.data, client) == i**2
        assert items[i].gen_index == i

    last_result = items[-1].result
    assert last_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert last_result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_UNSPECIFIED
    data = deserialize(last_result.data, client)
    assert isinstance(data, Exception)
    assert data.args == ("bad",)


@skip_windows_unix_socket
def test_async(unix_servicer):
    t0 = time.time()
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "square_async")
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(42**2)


@skip_windows_unix_socket
def test_failure(unix_servicer):
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "raises")
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert items[0].result.exception == "Exception('Failure!')"
    assert "Traceback" in items[0].result.traceback


@skip_windows_unix_socket
def test_raises_base_exception(unix_servicer):
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "raises_sysexit")
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert items[0].result.exception == "SystemExit(1)"


@skip_windows_unix_socket
def test_keyboardinterrupt(unix_servicer):
    with pytest.raises(KeyboardInterrupt):
        _run_container(unix_servicer, "modal_test_support.functions", "raises_keyboardinterrupt")


@skip_windows_unix_socket
def test_rate_limited(unix_servicer, event_loop):
    t0 = time.time()
    unix_servicer.rate_limit_sleep_duration = 0.25
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "square")
    assert 0.25 <= time.time() - t0 < 0.25 + EXTRA_TOLERANCE_DELAY
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(42**2)


@skip_windows_unix_socket
def test_grpc_failure(unix_servicer, event_loop):
    # An error in "Modal code" should cause the entire container to fail
    with pytest.raises(GRPCError):
        _run_container(unix_servicer, "modal_test_support.functions", "square", fail_get_inputs=True)

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
def test_class_scoped_function(unix_servicer, event_loop):
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "Cube.f")
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(42**3)

    Cube = sys.modules["modal_test_support.functions"].Cube  # don't redefine

    assert Cube._events == ["init", "enter", "call", "exit"]


@skip_windows_unix_socket
def test_class_scoped_function_async(unix_servicer, event_loop):
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "CubeAsync.f")
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(42**3)

    CubeAsync = sys.modules["modal_test_support.functions"].CubeAsync

    assert CubeAsync._events == ["init", "enter", "call", "exit"]


@skip_windows_unix_socket
def test_create_package_mounts_inside_container(unix_servicer, event_loop):
    """`create_package_mounts` shouldn't actually run inside the container, because it's possible
    that there are modules that were present locally for the user that didn't get mounted into
    all the containers."""

    client, items = _run_container(unix_servicer, "modal_test_support.package_mount", "num_mounts")
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(0)


@skip_windows_unix_socket
def test_webhook(unix_servicer, event_loop):
    scope = {
        "method": "GET",
        "type": "http",
        "path": "/",
        "headers": {},
        "query_string": "arg=space",
        "http_version": "2",
    }
    body = b""
    inputs = _get_inputs(([scope, body], {}))
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "webhook",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
    )

    # There should be one message for the header, one for the body, one for the EOF
    assert len(items) == 3

    # Check the headers
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
    first_message = deserialize(items[0].result.data, client)
    assert first_message["status"] == 200
    headers = dict(first_message["headers"])
    assert headers[b"content-type"] == b"application/json"

    # Check body
    assert items[1].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[1].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
    second_message = deserialize(items[1].result.data, client)
    assert json.loads(second_message["body"]) == {"hello": "space"}

    # Check EOF
    assert items[2].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[2].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE


@skip_windows_unix_socket
def test_webhook_lifecycle(unix_servicer, event_loop):
    scope = {
        "method": "GET",
        "type": "http",
        "path": "/",
        "headers": {},
        "query_string": "arg=space",
        "http_version": "2",
    }
    body = b""
    inputs = _get_inputs(([scope, body], {}))
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "WebhookLifecycleClass.webhook",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
    )

    assert len(items) == 3
    assert items[1].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[1].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
    second_message = deserialize(items[1].result.data, client)
    assert json.loads(second_message["body"]) == {"hello": "space"}


@skip_windows_unix_socket
def test_serialized_function(unix_servicer, event_loop):
    def triple(x):
        return 3 * x

    unix_servicer.function_serialized = serialize(triple)
    client, items = _run_container(
        unix_servicer,
        "foo.bar.baz",
        "f",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(3 * 42)


@skip_windows_unix_socket
def test_webhook_serialized(unix_servicer, event_loop):
    scope = {
        "method": "GET",
        "type": "http",
        "path": "/",
        "headers": {},
        "query_string": "arg=space",
        "http_version": "2",
    }
    body = b""
    inputs = _get_inputs(([scope, body], {}))

    # Store a serialized webhook function on the servicer
    def webhook(arg="world"):
        return f"Hello, {arg}"

    unix_servicer.function_serialized = serialize(webhook)

    client, items = _run_container(
        unix_servicer,
        "foo.bar.baz",
        "f",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )

    assert len(items) == 3
    assert items[1].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[1].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
    second_message = deserialize(items[1].result.data, client)
    assert second_message["body"] == b'"Hello, space"'  # Note: JSON-encoded


@skip_windows_unix_socket
def test_function_returning_generator(unix_servicer, event_loop):
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "fun_returning_gen",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )
    assert len(items) == 43  # The generator creates N outputs, and N is 42 from the autogenerated input
    assert items[-1].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[-1].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE


@skip_windows_unix_socket
def test_asgi(unix_servicer, event_loop):
    scope = {
        "method": "GET",
        "type": "http",
        "path": "/foo",
        "headers": {},
        "query_string": "arg=space",
        "http_version": "2",
    }
    body = b""
    inputs = _get_inputs(([scope, body], {}))
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "fastapi_app",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
    )

    # There should be one message for the header, one for the body, one for the EOF
    assert len(items) == 3

    # Check the headers
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
    first_message = deserialize(items[0].result.data, client)
    assert first_message["status"] == 200
    headers = dict(first_message["headers"])
    assert headers[b"content-type"] == b"application/json"

    # Check body
    assert items[1].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[1].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
    second_message = deserialize(items[1].result.data, client)
    assert json.loads(second_message["body"]) == {"hello": "space"}

    # Check EOF
    assert items[2].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[2].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE


@skip_windows_unix_socket
def test_webhook_streaming_sync(unix_servicer, event_loop):
    scope = {
        "method": "GET",
        "type": "http",
        "path": "/",
        "headers": {},
        "query_string": "",
        "http_version": "2",
    }
    body = b""
    inputs = _get_inputs(([scope, body], {}))
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "webhook_streaming",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
    )

    assert len(items) > 3


@skip_windows_unix_socket
def test_webhook_streaming_async(unix_servicer, event_loop):
    scope = {
        "method": "GET",
        "type": "http",
        "path": "/",
        "headers": {},
        "query_string": "",
        "http_version": "2",
    }
    body = b""
    inputs = _get_inputs(([scope, body], {}))
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "webhook_streaming_async",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
    )

    assert len(items) > 3


@skip_windows_unix_socket
def test_container_heartbeats(unix_servicer, event_loop):
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "square")
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
    )
    container_args = api_pb2.ContainerArguments(
        task_id="ta-123",
        function_id="fu-123",
        app_id="se-123",
        function_def=function_def,
    )
    data_base64: str = base64.b64encode(container_args.SerializeToString()).decode("ascii")

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
    # assert stderr == ""  # TODO(erikbern): this doesn't work right now:
