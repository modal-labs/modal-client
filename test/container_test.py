# Copyright Modal Labs 2022

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
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

from grpclib.exceptions import GRPCError

from modal import Client
from modal._container_entrypoint import UserException, main
from modal._serialization import deserialize, deserialize_data_format, serialize
from modal.exception import DeprecationError, InvalidError
from modal.stub import _Stub
from modal_proto import api_pb2

from .helpers import deploy_stub_externally
from .supports.skip import skip_windows_unix_socket

EXTRA_TOLERANCE_DELAY = 2.0 if sys.platform == "linux" else 5.0
FUNCTION_CALL_ID = "fc-123"
SLEEP_DELAY = 0.1


def _get_inputs(args: Tuple[Tuple, Dict] = ((42,), {}), n: int = 1) -> List[api_pb2.FunctionGetInputsResponse]:
    input_pb = api_pb2.FunctionInput(args=serialize(args), data_format=api_pb2.DATA_FORMAT_PICKLE)

    return [
        api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(input_id=f"in-xyz{i}", input=input_pb)])
        for i in range(n)
    ] + [api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(kill_switch=True)])]


@dataclasses.dataclass
class ContainerResult:
    client: Client
    items: List[api_pb2.FunctionPutOutputsItem]


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
) -> ContainerResult:
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
            object_dependencies=[api_pb2.ObjectDependency(object_id=object_id) for object_id in deps],
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
        temp_restore_file_path = tempfile.NamedTemporaryFile()
        if is_checkpointing_function:
            # State file is written to allow for a restore to happen.
            tmep_file_name = temp_restore_file_path.name
            with pathlib.Path(tmep_file_name).open("w") as target:
                json.dump({}, target)
            env["MODAL_RESTORE_STATE_PATH"] = tmep_file_name

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
        items: List[api_pb2.FunctionPutOutputsItem] = []
        for req in servicer.container_outputs:
            items += list(req.outputs)

        return ContainerResult(client, items)


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
    items = []
    for i in range(len(ret.items) - 1):
        result = ret.items[i].result
        assert result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        assert result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
        assert ret.items[i].gen_index == i
        items.append(deserialize(result.data, ret.client))

    last_result = ret.items[-1].result
    if last_result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
        assert last_result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE
        assert last_result.data == b""  # no data in generator complete marker result
        return (items, None)
    elif last_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
        assert last_result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_UNSPECIFIED
        exc = deserialize(last_result.data, ret.client)
        return (items, exc)
    else:
        raise RuntimeError("unknown result type")


def _unwrap_asgi(ret: ContainerResult):
    items = []
    for item in ret.items[:-1]:
        assert item.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        assert item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
        assert item.data_format == api_pb2.DATA_FORMAT_ASGI
        items.append(deserialize_data_format(item.result.data, item.data_format, None))

    last_item = ret.items[-1]
    assert last_item.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert last_item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE

    return items


@skip_windows_unix_socket
def test_success(unix_servicer, event_loop):
    t0 = time.time()
    ret = _run_container(unix_servicer, "modal_test_support.functions", "square")
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY
    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_generator_success(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer, "modal_test_support.functions", "gen_n", function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR
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
    ret = _run_container(unix_servicer, "modal_test_support.package_mount", "num_mounts")
    assert _unwrap_scalar(ret) == 0


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
    ret = _run_container(
        unix_servicer,
        "modal_test_support.functions",
        "fastapi_app",
        inputs=inputs,
        webhook_type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
    )

    # There should be one message for the header, one for the body, one for the EOF
    # EOF is removed by _unwrap_asgi
    first_message, second_message = _unwrap_asgi(ret)

    # Check the headers
    assert first_message["status"] == 200
    headers = dict(first_message["headers"])
    assert headers[b"content-type"] == b"application/json"

    # Check body
    assert json.loads(second_message["body"]) == {"hello": "space"}


@skip_windows_unix_socket
def test_webhook_streaming_sync(unix_servicer, event_loop):
    inputs = _get_web_inputs()
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
def test_param_cls_function(unix_servicer, event_loop):
    serialized_params = pickle.dumps(([111], {"y": "foo"}))
    ret = _run_container(
        unix_servicer, "modal_test_support.functions", "ParamCls.f", serialized_params=serialized_params
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
        def __enter__(self):
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
        unix_servicer, "modal_test_support.multistub_privately_decorated_named_stub", "foo", stub_name="dummy"
    )
    assert _unwrap_scalar(ret) == 1
    assert len(caplog.messages) == 0  # no warnings, since target stub is named


@skip_windows_unix_socket
def test_multistub_same_name_warning(unix_servicer, caplog):
    # function handle does not override the original function, so we can't find the stub
    # two stubs with the same name - warn since we won't know which one to hydrate
    ret = _run_container(unix_servicer, "modal_test_support.multistub_same_name", "foo", stub_name="dummy")
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


@skip_windows_unix_socket
def test_is_inside(unix_servicer, caplog, capsys):
    with pytest.warns(DeprecationError, match="run_inside"):
        ret = _run_container(unix_servicer, "modal_test_support.is_inside", "foo")
    assert _unwrap_scalar(ret) is None
    assert len(caplog.messages) == 0
    out, err = capsys.readouterr()
    assert "in container!" in out
    assert "in local" not in out


@skip_windows_unix_socket
def test_multistub_is_inside(unix_servicer, caplog, capsys):
    with pytest.warns(DeprecationError, match="run_inside"):
        ret = _run_container(unix_servicer, "modal_test_support.multistub_is_inside", "foo", stub_name="a")
    assert _unwrap_scalar(ret) is None
    assert len(caplog.messages) == 0
    out, err = capsys.readouterr()
    assert "inside a" in out
    assert "inside b" not in out


@skip_windows_unix_socket
def test_multistub_is_inside_warning(unix_servicer, caplog, capsys):
    with pytest.warns(DeprecationError, match="run_inside"):
        ret = _run_container(unix_servicer, "modal_test_support.multistub_is_inside_warning", "foo")
    assert _unwrap_scalar(ret) is None
    assert len(caplog.messages) == 1
    assert "You have more than one unnamed stub" in caplog.text
    out, err = capsys.readouterr()
    assert "inside a" in out
    assert (
        "inside b" in out
    )  # can't determine which of two anonymous stubs is the active one at import time, so both will trigger


SLEEP_TIME = 0.7


def _unwrap_concurrent_input_outputs(n_inputs: int, n_parallel: int, ret: ContainerResult):
    # Ensure that outputs align with expectation of running concurrent inputs

    # Each group of n_parallel inputs should start together of each other
    # and different groups should start SLEEP_TIME apart.
    assert len(ret.items) == n_inputs
    for i in range(1, len(ret.items)):
        diff = ret.items[i].input_started_at - ret.items[i - 1].input_started_at
        expected_diff = SLEEP_TIME if i % n_parallel == 0 else 0
        assert diff == pytest.approx(expected_diff, abs=0.2)

    outputs = []
    for item in ret.items:
        assert item.output_created_at - item.input_started_at == pytest.approx(SLEEP_TIME, abs=0.2)
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
    assert _unwrap_concurrent_input_outputs(n_inputs, n_parallel, ret) == [42**2] * n_inputs


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
    assert _unwrap_concurrent_input_outputs(n_inputs, n_parallel, ret) == [42**2] * n_inputs


@skip_windows_unix_socket
def test_unassociated_function(unix_servicer, event_loop):
    ret = _run_container(unix_servicer, "modal_test_support.functions", "unassociated_function")
    assert _unwrap_scalar(ret) == 58


@skip_windows_unix_socket
def test_param_cls_function_calling_local(unix_servicer, event_loop):
    serialized_params = pickle.dumps(([111], {"y": "foo"}))
    ret = _run_container(
        unix_servicer, "modal_test_support.functions", "ParamCls.g", serialized_params=serialized_params
    )
    assert _unwrap_scalar(ret) == "111 foo 42"


@skip_windows_unix_socket
def test_derived_cls(unix_servicer, event_loop):
    ret = _run_container(
        unix_servicer, "modal_test_support.functions", "DerivedCls.run", inputs=_get_inputs(((3,), {}))
    )
    assert _unwrap_scalar(ret) == 6


@skip_windows_unix_socket
def test_call_function_that_calls_function(unix_servicer, event_loop):
    deploy_stub_externally(unix_servicer, "modal_test_support.functions", "stub")
    ret = _run_container(unix_servicer, "modal_test_support.functions", "cube", inputs=_get_inputs(((42,), {})))
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
    ret = _run_container(unix_servicer, "modal_test_support.functions", "square", is_checkpointing_function=True)
    assert any(isinstance(request, api_pb2.ContainerCheckpointRequest) for request in unix_servicer.requests)
    assert _unwrap_scalar(ret) == 42**2


@skip_windows_unix_socket
def test_function_dep_hydration(unix_servicer):
    deploy_stub_externally(unix_servicer, "modal_test_support.functions", "stub")
    ret = _run_container(unix_servicer, "modal_test_support.functions", "check_dep_hydration", deps=["im-1", "vo-1"])
    assert _unwrap_scalar(ret) is None
