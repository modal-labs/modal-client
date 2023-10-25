# Copyright Modal Labs 2022
from __future__ import annotations

import base64
import json
import os
import pathlib
import pickle
import pytest
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

from grpclib.exceptions import GRPCError

from modal import Client
from modal._container_entrypoint import UserException, main
from modal._serialization import deserialize, deserialize_data_format, serialize
from modal.exception import InvalidError
from modal.stub import _Stub
from modal_proto import api_pb2

from .helpers import deploy_stub_externally
from .supports.skip import skip_windows_unix_socket

EXTRA_TOLERANCE_DELAY = 2.0 if sys.platform == "linux" else 5.0
FUNCTION_CALL_ID = "fc-123"
SLEEP_DELAY = 0.1


def _get_inputs(args: Tuple[Tuple, Dict] = ((42,), {}), n: int = 1) -> list[api_pb2.FunctionGetInputsResponse]:
    input_pb = api_pb2.FunctionInput(args=serialize(args), data_format=api_pb2.DATA_FORMAT_PICKLE)

    return [
        api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(input_id=f"in-xyz{i}", input=input_pb)])
        for i in range(n)
    ] + [api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(kill_switch=True)])]


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
                async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
            )
        else:
            webhook_config = None

        function_def = api_pb2.Function(
            module_name=module_name,
            function_name=function_name,
            function_type=function_type,
            webhook_config=webhook_config,
            definition_type=definition_type,
            stub_name=stub_name or "",
            is_builder_function=is_builder_function,
            allow_concurrent_inputs=allow_concurrent_inputs,
            is_checkpointing_function=is_checkpointing_function,
        )

        container_args = api_pb2.ContainerArguments(
            task_id="ta-123",
            function_id="fu-123",
            app_id="ap-1",
            function_def=function_def,
            serialized_params=serialized_params,
        )

        if module_name in sys.modules:
            # Drop the module from sys.modules since some function code relies on the
            # assumption that that the app is created before the user code is imported.
            # This is really only an issue for tests.
            sys.modules.pop(module_name)

        env = os.environ.copy()
        if is_checkpointing_function:
            # Environment variable is set to allow restore from a checkpoint.
            # Override server URL to reproduce restore behavior.
            env["MODAL_FUNCTION_RESTORED"] = "1"
            env["MODAL_SERVER_URL"] = servicer.remote_addr

        try:
            with mock.patch.dict(os.environ, env):
                main(container_args, client)
        except UserException:
            # Handle it gracefully
            pass

        # Flatten outputs
        items: list[api_pb2.FunctionPutOutputsItem] = []
        for req in servicer.container_outputs:
            items += list(req.outputs)

        return client, items


def _unwrap_asgi_response(item: api_pb2.FunctionPutOutputsItem) -> Any:
    assert item.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
    assert item.data_format == api_pb2.DATA_FORMAT_ASGI
    return deserialize_data_format(item.result.data, item.data_format, None)


def _unwrap_asgi_done(item: api_pb2.FunctionPutOutputsItem) -> None:
    assert item.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE


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
def test_from_local_python_packages_inside_container(unix_servicer, event_loop, monkeypatch):
    """`from_local_python_packages` shouldn't actually collect modules inside the container, because it's possible
    that there are modules that were present locally for the user that didn't get mounted into
    all the containers."""
    client, items = _run_container(unix_servicer, "modal_test_support.package_mount", "num_mounts")
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert deserialize(items[0].result.data, None) == 0


def _get_web_inputs(path="/"):
    scope = {
        "method": "GET",
        "type": "http",
        "path": path,
        "headers": {},
        "query_string": "arg=space",
        "http_version": "2",
    }
    body = b""
    return _get_inputs(((scope, body), {}))


@skip_windows_unix_socket
def test_webhook(unix_servicer, event_loop):
    inputs = _get_web_inputs()
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
    first_message = _unwrap_asgi_response(items[0])
    assert first_message["status"] == 200
    headers = dict(first_message["headers"])
    assert headers[b"content-type"] == b"application/json"

    # Check body
    second_message = _unwrap_asgi_response(items[1])
    assert json.loads(second_message["body"]) == {"hello": "space"}

    # Check EOF
    _unwrap_asgi_done(items[2])


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
    inputs = _get_web_inputs()

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
    second_message = _unwrap_asgi_response(items[1])
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
    inputs = _get_web_inputs(path="/foo")
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
    first_message = _unwrap_asgi_response(items[0])
    assert first_message["status"] == 200
    headers = dict(first_message["headers"])
    assert headers[b"content-type"] == b"application/json"

    # Check body=
    second_message = _unwrap_asgi_response(items[1])
    assert json.loads(second_message["body"]) == {"hello": "space"}

    # Check EOF
    _unwrap_asgi_done(items[2])


@skip_windows_unix_socket
def test_webhook_streaming_sync(unix_servicer, event_loop):
    inputs = _get_web_inputs()
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "webhook_streaming",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )

    data = [_unwrap_asgi_response(item) for item in items if item.result.data]
    bodies = [d["body"].decode() for d in data if d.get("body")]
    assert bodies == [f"{i}..." for i in range(10)]


@skip_windows_unix_socket
def test_webhook_streaming_async(unix_servicer, event_loop):
    inputs = _get_web_inputs()
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "webhook_streaming_async",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )

    data = [_unwrap_asgi_response(item) for item in items if item.result.data]
    bodies = [d["body"].decode() for d in data if d.get("body")]
    assert bodies == [f"{i}..." for i in range(10)]


@skip_windows_unix_socket
def test_cls_function(unix_servicer, event_loop):
    result = _run_e2e_function(unix_servicer, "modal_test_support.functions", "stub", "Cls.f")
    assert result == 42 * 111


@skip_windows_unix_socket
def test_param_cls_function(unix_servicer, event_loop):
    serialized_params = pickle.dumps(([111], {"y": "foo"}))
    result = _run_e2e_function(
        unix_servicer, "modal_test_support.functions", "stub", "ParamCls.f", serialized_params=serialized_params
    )
    assert result == "111 foo 42"


@skip_windows_unix_socket
def test_cls_web_endpoint(unix_servicer, event_loop):
    inputs = _get_web_inputs()
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "Cls.web",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_FUNCTION,
    )

    assert len(items) == 3
    second_message = _unwrap_asgi_response(items[1])
    assert json.loads(second_message["body"]) == {"ret": "space" * 111}


@skip_windows_unix_socket
def test_serialized_cls(unix_servicer, event_loop):
    class Cls:
        def __enter__(self):
            self.power = 5

        def method(self, x):
            return x**self.power

    unix_servicer.class_serialized = serialize(Cls)
    unix_servicer.function_serialized = serialize(Cls.method)
    client, items = _run_container(
        unix_servicer,
        "module.doesnt.matter",
        "function.doesnt.matter",
        definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(42**5)


@skip_windows_unix_socket
def test_cls_generator(unix_servicer, event_loop):
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "Cls.generator",
        function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR,
    )
    assert len(items) == 2
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
    assert items[0].result.data == serialize(42**3)
    assert items[1].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[1].result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE


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
    assert stderr == ""


def _run_e2e_function(
    servicer,
    module_name,
    stub_var_name,
    function_name,
    *,
    stub_name="",
    assert_result=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
    function_definition_type=api_pb2.Function.DEFINITION_TYPE_FILE,
    inputs=None,
    is_builder_function: bool = False,
    serialized_params=None,
):
    # TODO(elias): make this a bit more prod-like in how it connects the load and run parts by returning function definitions from _load_stub so we don't have to double specify things like definition type
    _Stub._all_stubs = {}  # reset _Stub tracking state between runs
    deploy_stub_externally(servicer, module_name, stub_var_name)
    client, items = _run_container(
        servicer,
        module_name,
        function_name,
        stub_name=stub_name,
        definition_type=function_definition_type,
        inputs=inputs,
        is_builder_function=is_builder_function,
        serialized_params=serialized_params,
    )
    assert items[0].result.status == assert_result
    return deserialize(items[0].result.data, client)


@skip_windows_unix_socket
def test_function_hydration(unix_servicer):
    _run_e2e_function(unix_servicer, "modal_test_support.functions", "stub", "check_sibling_hydration")


@skip_windows_unix_socket
def test_multistub(unix_servicer, caplog):
    _run_e2e_function(unix_servicer, "modal_test_support.multistub", "a", "a_func")
    assert (
        len(caplog.messages) == 1
    )  # warns in case the user would use is_inside checks... Hydration should work regardless
    assert "You have more than one unnamed stub" in caplog.text


@skip_windows_unix_socket
def test_multistub_privately_decorated(unix_servicer, caplog):
    # function handle does not override the original function, so we can't find the stub
    # and the two stubs are not named
    _run_e2e_function(unix_servicer, "modal_test_support.multistub_privately_decorated", "stub", "foo")
    assert "You have more than one unnamed stub." in caplog.text


@skip_windows_unix_socket
def test_multistub_privately_decorated_named_stub(unix_servicer, caplog):
    # function handle does not override the original function, so we can't find the stub
    # but we can use the names of the stubs to determine the active stub
    _run_e2e_function(
        unix_servicer, "modal_test_support.multistub_privately_decorated_named_stub", "stub", "foo", stub_name="dummy"
    )
    assert len(caplog.messages) == 0  # no warnings, since target stub is named


@skip_windows_unix_socket
def test_multistub_same_name_warning(unix_servicer, caplog):
    # function handle does not override the original function, so we can't find the stub
    # two stubs with the same name - warn since we won't know which one to hydrate
    _run_e2e_function(unix_servicer, "modal_test_support.multistub_same_name", "stub", "foo", stub_name="dummy")
    assert "You have more than one stub with the same name ('dummy')" in caplog.text


@skip_windows_unix_socket
def test_multistub_serialized_func(unix_servicer, caplog):
    # serialized functions shouldn't warn about multiple/not finding stubs, since they shouldn't load the module to begin with
    def dummy(x):
        return x

    unix_servicer.function_serialized = serialize(dummy)
    _run_e2e_function(
        unix_servicer,
        "modal_test_support.multistub_serialized_func",
        "stub",
        "foo",
        function_definition_type=api_pb2.Function.DEFINITION_TYPE_SERIALIZED,
    )
    assert len(caplog.messages) == 0


@skip_windows_unix_socket
def test_image_run_function_no_warn(unix_servicer, caplog):
    # builder functions currently aren't tied to any modal stub, so they shouldn't need to warn if they can't determine a stub to use
    _run_e2e_function(
        unix_servicer,
        "modal_test_support.image_run_function",
        "stub",
        "builder_function",
        inputs=_get_inputs(((), {})),
        is_builder_function=True,
    )
    assert len(caplog.messages) == 0


@skip_windows_unix_socket
def test_is_inside(unix_servicer, caplog, capsys):
    _run_e2e_function(
        unix_servicer,
        "modal_test_support.is_inside",
        "stub",
        "foo",
    )
    assert len(caplog.messages) == 0
    out, err = capsys.readouterr()
    assert "in container!" in out
    assert "in local" not in out


@skip_windows_unix_socket
def test_multistub_is_inside(unix_servicer, caplog, capsys):
    _run_e2e_function(unix_servicer, "modal_test_support.multistub_is_inside", "a_stub", "foo", stub_name="a")
    assert len(caplog.messages) == 0
    out, err = capsys.readouterr()
    assert "inside a" in out
    assert "inside b" not in out


@skip_windows_unix_socket
def test_multistub_is_inside_warning(unix_servicer, caplog, capsys):
    _run_e2e_function(
        unix_servicer,
        "modal_test_support.multistub_is_inside_warning",
        "a_stub",
        "foo",
    )
    assert len(caplog.messages) == 1
    assert "You have more than one unnamed stub" in caplog.text
    out, err = capsys.readouterr()
    assert "inside a" in out
    assert (
        "inside b" in out
    )  # can't determine which of two anonymous stubs is the active one at import time, so both will trigger


SLEEP_TIME = 0.7


def verify_concurrent_input_outputs(n_inputs: int, n_parallel: int, output_items: list[api_pb2.FunctionPutOutputsItem]):
    # Ensure that outputs align with expectation of running concurrent inputs

    # Each group of n_parallel inputs should start together of each other
    # and different groups should start SLEEP_TIME apart.
    assert len(output_items) == n_inputs
    for i in range(1, len(output_items)):
        diff = output_items[i].input_started_at - output_items[i - 1].input_started_at
        expected_diff = SLEEP_TIME if i % n_parallel == 0 else 0
        assert diff == pytest.approx(expected_diff, abs=0.2)

    for item in output_items:
        assert item.output_created_at - item.input_started_at == pytest.approx(SLEEP_TIME, abs=0.2)
        assert item.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        assert item.result.data == serialize(42**2)


@skip_windows_unix_socket
def test_concurrent_inputs_sync_function(unix_servicer):
    n_inputs = 18
    n_parallel = 6

    t0 = time.time()
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "sleep_700_sync",
        inputs=_get_inputs(n=n_inputs),
        allow_concurrent_inputs=n_parallel,
    )

    expected_execution = n_inputs / n_parallel * SLEEP_TIME
    assert expected_execution <= time.time() - t0 < expected_execution + EXTRA_TOLERANCE_DELAY
    verify_concurrent_input_outputs(n_inputs, n_parallel, items)


@skip_windows_unix_socket
def test_concurrent_inputs_async_function(unix_servicer, event_loop):
    n_inputs = 18
    n_parallel = 6

    t0 = time.time()
    client, items = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "sleep_700_async",
        inputs=_get_inputs(n=n_inputs),
        allow_concurrent_inputs=n_parallel,
    )

    expected_execution = n_inputs / n_parallel * SLEEP_TIME
    assert expected_execution <= time.time() - t0 < expected_execution + EXTRA_TOLERANCE_DELAY
    verify_concurrent_input_outputs(n_inputs, n_parallel, items)


@skip_windows_unix_socket
def test_unassociated_function(unix_servicer, event_loop):
    client, items = _run_container(unix_servicer, "modal_test_support.functions", "unassociated_function")
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data
    assert deserialize(items[0].result.data, client) == 58


@skip_windows_unix_socket
def test_param_cls_function_calling_local(unix_servicer, event_loop):
    serialized_params = pickle.dumps(([111], {"y": "foo"}))
    client, items = _run_container(
        unix_servicer, "modal_test_support.functions", "ParamCls.g", serialized_params=serialized_params
    )
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize("111 foo 42")


@skip_windows_unix_socket
def test_derived_cls(unix_servicer, event_loop):
    client, items = _run_container(
        unix_servicer, "modal_test_support.functions", "DerivedCls.run", inputs=_get_inputs(((3,), {}))
    )
    assert len(items) == 1
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(6)


@skip_windows_unix_socket
def test_call_function_that_calls_function(unix_servicer, event_loop):
    result = _run_e2e_function(
        unix_servicer, "modal_test_support.functions", "stub", "cube", inputs=_get_inputs(((42,), {}))
    )
    assert result == 42**3


@skip_windows_unix_socket
def test_call_function_that_calls_method(unix_servicer, event_loop):
    _run_e2e_function(
        unix_servicer,
        "modal_test_support.functions",
        "stub",
        "function_calling_method",
        inputs=_get_inputs(((42, "abc", 123), {})),
    )


@skip_windows_unix_socket
def test_checkpoint_and_restore_success(unix_servicer, event_loop):
    """Functions send a checkpointing request and continue to execute normally,
    simulating a restore operation."""
    _, items = _run_container(unix_servicer, "modal_test_support.functions", "square", is_checkpointing_function=True)
    assert any(isinstance(request, api_pb2.ContainerCheckpointRequest) for request in unix_servicer.requests)
    assert items[0].result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert items[0].result.data == serialize(42**2)
