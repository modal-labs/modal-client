import asyncio
import time

import pytest
from modal import Client, Session, debian_slim
from modal.container_entrypoint import main
from modal.function import Function, pack_input_buffer_item
from modal.proto import api_pb2
from modal.test_support import SLEEP_DELAY

EXTRA_TOLERANCE_DELAY = 0.08
OUTPUT_BUFFER = "output_buffer_id"
INPUT_BUFFER = "input_buffer_id"

session = Session()  # Just used for (de)serialization


def _get_inputs(client):
    item = pack_input_buffer_item(session.serialize((42,)), session.serialize({}), OUTPUT_BUFFER)

    return [
        api_pb2.BufferReadResponse(items=[item], status=api_pb2.BufferReadResponse.BufferReadStatus.SUCCESS),
        api_pb2.BufferReadResponse(
            items=[api_pb2.BufferItem(EOF=True)], status=api_pb2.BufferReadResponse.BufferReadStatus.SUCCESS
        ),
    ]


def _get_output(function_output_req: api_pb2.FunctionOutputRequest) -> api_pb2.GenericResult:
    output = api_pb2.GenericResult()
    assert len(function_output_req.buffer_req.items) == 1
    function_output_req.buffer_req.items[0].data.Unpack(output)
    return output


async def _run_container(servicer, module_name, function_name):
    async with Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ("ta-123", "task-secret")) as client:
        servicer.inputs = _get_inputs(client)

        function_def = api_pb2.Function(
            module_name=module_name,
            function_name=function_name,
        )

        # Note that main is a synchronous function, so we need to run it in a separate thread
        container_args = api_pb2.ContainerArguments(
            task_id="ta-123",
            function_id="fu-123",
            input_buffer_id=INPUT_BUFFER,
            session_id="se-123",
            function_def=function_def,
        )

        servicer.object_ids = {
            debian_slim.tag: "1",
            "modal.test_support.square": "2",
            "modal.test_support.square_sync_returning_async": "3",
            "modal.test_support.square_async": "4",
            "modal.test_support.raises": "5",
        }
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, main, container_args, client)

        return client, servicer.outputs


@pytest.mark.asyncio
async def test_container_entrypoint_success(servicer, reset_session_singleton):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "modal.test_support", "square")
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.Status.SUCCESS
    session = Session()
    assert output.data == session.serialize(42 ** 2)


@pytest.mark.asyncio
async def test_container_entrypoint_async(servicer, reset_session_singleton):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "modal.test_support", "square_async")
    print(time.time() - t0, outputs)
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.Status.SUCCESS
    assert output.data == session.serialize(42 ** 2)


@pytest.mark.asyncio
async def test_container_entrypoint_sync_returning_async(servicer, reset_session_singleton):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "modal.test_support", "square_sync_returning_async")
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.Status.SUCCESS
    assert output.data == session.serialize(42 ** 2)


@pytest.mark.asyncio
async def test_container_entrypoint_failure(servicer, reset_session_singleton):
    client, outputs = await _run_container(servicer, "modal.test_support", "raises")

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.Status.FAILURE
    assert output.exception in ["Exception('Failure!')", "Exception('Failure!',)"]  # The 2nd is 3.6
    assert "Traceback" in output.traceback
