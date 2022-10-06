import os
import platform
import pytest
import sys
import time

from grpclib.exceptions import GRPCError

from modal._container_entrypoint import UserException, main

# from modal_test_support import SLEEP_DELAY
from modal._serialization import deserialize, serialize
from modal.client import Client
from modal.exception import InvalidError
from modal_proto import api_pb2

# Something with timing is flaky in OSX & Windows tests
EXTRA_TOLERANCE_DELAY = {"Darwin": 1.0, "Windows": 3.0}.get(platform.system(), 0.25)
FUNCTION_CALL_ID = "fc-123"
SLEEP_DELAY = 0.1


skip_github_actions_non_linux = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") and platform.system() != "Linux",
    reason="sleep is inaccurate on Github Actions runners.",
)


def _get_inputs(client):
    args = ((42,), {})
    input_pb = api_pb2.FunctionInput(args=serialize(args))

    return [
        api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(input_id="in-xyz", input=input_pb)]),
        api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionGetInputsItem(kill_switch=True)]),
    ]


def _get_output(function_output_req: api_pb2.FunctionPutOutputsRequest) -> api_pb2.GenericResult:
    assert len(function_output_req.outputs) == 1
    return function_output_req.outputs[0].result


def _run_container(servicer, module_name, function_name, fail_get_inputs=False, inputs=None):
    with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret")) as client:
        if inputs is None:
            servicer.container_inputs = _get_inputs(client)
        else:
            servicer.container_inputs = inputs
        servicer.fail_get_inputs = fail_get_inputs

        function_def = api_pb2.Function(
            module_name=module_name,
            function_name=function_name,
        )

        # Note that main is a synchronous function, so we need to run it in a separate thread
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

        return client, servicer.container_outputs


def test_container_entrypoint_success(servicer, event_loop):
    t0 = time.time()
    client, outputs = _run_container(servicer, "modal_test_support.functions", "square")
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert output.data == serialize(42**2)


@skip_github_actions_non_linux
def test_container_entrypoint_async(servicer):
    t0 = time.time()
    client, outputs = _run_container(servicer, "modal_test_support.functions", "square_async")
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert output.data == serialize(42**2)


@skip_github_actions_non_linux
def test_container_entrypoint_failure(servicer):
    client, outputs = _run_container(servicer, "modal_test_support.functions", "raises")

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert output.exception == "Exception('Failure!')"
    assert "Traceback" in output.traceback


@skip_github_actions_non_linux
def test_container_entrypoint_raises_base_exception(servicer):
    client, outputs = _run_container(servicer, "modal_test_support.functions", "raises_sysexit")

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert output.exception == "SystemExit(1)"


@skip_github_actions_non_linux
def test_container_entrypoint_keyboardinterrupt(servicer):
    with pytest.raises(KeyboardInterrupt):
        client, outputs = _run_container(servicer, "modal_test_support.functions", "raises_keyboardinterrupt")


def test_container_entrypoint_rate_limited(servicer, event_loop):
    t0 = time.time()
    servicer.rate_limit_sleep_duration = 0.25
    client, outputs = _run_container(servicer, "modal_test_support.functions", "square")
    assert 0.25 <= time.time() - t0 < 0.25 + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert output.data == serialize(42**2)


def test_container_entrypoint_grpc_failure(servicer, event_loop):
    # An error in "Modal code" should cause the entire container to fail
    with pytest.raises(GRPCError):
        _run_container(servicer, "modal_test_support.functions", "square", fail_get_inputs=True)

    # assert servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    # assert "GRPCError" in servicer.task_result.exception


def test_container_entrypoint_missing_main_conditional(servicer, event_loop):
    _run_container(servicer, "modal_test_support.missing_main_conditional", "square")

    assert servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert 'if __name__ == "__main__":' in servicer.task_result.traceback

    exc = deserialize(servicer.task_result.data, None)
    assert isinstance(exc, InvalidError)


def test_container_entrypoint_startup_failure(servicer, event_loop):
    _run_container(servicer, "modal_test_support.startup_failure", "f")

    assert servicer.task_result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE

    exc = deserialize(servicer.task_result.data, None)
    assert isinstance(exc, ImportError)


def test_container_entrypoint_class_scoped_function(servicer, event_loop):
    client, outputs = _run_container(servicer, "modal_test_support.functions", "Cube.f")
    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert output.data == serialize(42**3)

    Cube = sys.modules["modal_test_support.functions"].Cube  # don't redefine

    assert Cube._events == ["init", "enter", "call", "exit"]


def test_container_entrypoint_class_scoped_function_async(servicer, event_loop):
    client, outputs = _run_container(servicer, "modal_test_support.functions", "CubeAsync.f")
    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert output.data == serialize(42**3)

    CubeAsync = sys.modules["modal_test_support.functions"].CubeAsync

    assert CubeAsync._events == ["init", "enter", "call", "exit"]


def test_create_package_mounts_inside_container(servicer, event_loop):
    """`create_package_mounts` shouldn't actually run inside the container, because it's possible
    that there are modules that were present locally for the user that didn't get mounted into
    all the containers."""

    client, outputs = _run_container(servicer, "modal_test_support.package_mount", "num_mounts")

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert output.data == serialize(0)
