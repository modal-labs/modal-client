import asyncio
import pytest
import time

from modal import App, debian_slim
from modal._client import Client
from modal._container_entrypoint import main
from modal._test_support import SLEEP_DELAY
from modal.proto import api_pb2

EXTRA_TOLERANCE_DELAY = 0.08
FUNCTION_CALL_ID = "fc-123"

app = App()  # Just used for (de)serialization


def _get_inputs(client):
    function_input = api_pb2.FunctionInput(args=app.serialize(((42,), {})), function_call_id=FUNCTION_CALL_ID)

    return [
        api_pb2.FunctionGetInputsResponse(inputs=[function_input], status=api_pb2.READ_STATUS_SUCCESS),
        api_pb2.FunctionGetInputsResponse(inputs=[api_pb2.FunctionInput(EOF=True)], status=api_pb2.READ_STATUS_SUCCESS),
    ]


def _get_output(function_output_req: api_pb2.FunctionPutOutputsRequest) -> api_pb2.GenericResult:
    assert len(function_output_req.outputs) == 1
    return function_output_req.outputs[0]


async def _run_container(servicer, module_name, function_name):
    async with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret")) as client:
        servicer.container_inputs = _get_inputs(client)

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

        servicer.object_ids = {
            debian_slim.tag: "1",
            "modal._test_support.square": "2",
            "modal._test_support.square_sync_returning_async": "3",
            "modal._test_support.square_async": "4",
            "modal._test_support.raises": "5",
        }
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, main, container_args, client)

        return client, servicer.container_outputs


@pytest.mark.asyncio
async def test_container_entrypoint_success(servicer, reset_global_apps, event_loop):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "modal._test_support", "square")
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    app = App()
    assert output.data == app.serialize(42**2)


@pytest.mark.asyncio
async def test_container_entrypoint_async(servicer, reset_global_apps):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "modal._test_support", "square_async")
    print(time.time() - t0, outputs)
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert output.data == app.serialize(42**2)


@pytest.mark.asyncio
async def test_container_entrypoint_sync_returning_async(servicer, reset_global_apps):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "modal._test_support", "square_sync_returning_async")
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
    assert output.data == app.serialize(42**2)


@pytest.mark.asyncio
async def test_container_entrypoint_failure(servicer, reset_global_apps):
    client, outputs = await _run_container(servicer, "modal._test_support", "raises")

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionPutOutputsRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE
    assert output.exception in ["Exception('Failure!')", "Exception('Failure!',)"]  # The 2nd is 3.6
    assert "Traceback" in output.traceback
