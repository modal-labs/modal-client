import asyncio
import pytest
import time

from google.protobuf.any_pb2 import Any
from polyester.client import Client
from polyester.container_entrypoint import main
from polyester.function import Function
from polyester.proto import api_pb2
from polyester.test_support import SLEEP_DELAY

EXTRA_TOLERANCE_DELAY = 0.05
OUTPUT_BUFFER = "output_buffer_id"
INPUT_BUFFER = "input_buffer_id"


def _get_inputs(client):
    data = Any()
    data.Pack(api_pb2.FunctionInput(
        args=client.serialize((42,)),
        kwargs=client.serialize({}),
        output_buffer_id=OUTPUT_BUFFER,
    ))

    return [
        api_pb2.BufferReadResponse(item=api_pb2.BufferItem(data=data)),
        api_pb2.BufferReadResponse(item=api_pb2.BufferItem(EOF=True)),
    ]

def _get_output(function_output_req: api_pb2.FunctionOutputRequest) -> api_pb2.GenericResult:
    output = api_pb2.GenericResult()
    function_output_req.buffer_req.item.data.Unpack(output)
    return output


async def _run_container(servicer, module_name, function_name):
    async with Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ("ta-123", "task-secret")) as client:
        servicer.inputs = _get_inputs(client)

        # Note that main is a synchronous function, so we need to run it in a separate thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, main, "ta-123", "fu-123", INPUT_BUFFER, module_name, function_name, client)

        return client, servicer.outputs


@pytest.mark.asyncio
async def test_container_entrypoint_success(servicer):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "polyester.test_support", "square")
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 2
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.Status.SUCCESS
    assert output.data == client.serialize(42 ** 2)


@pytest.mark.asyncio
async def test_container_entrypoint_async(servicer):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "polyester.test_support", "square_async")
    print(time.time() - t0, outputs)
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 2
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.Status.SUCCESS
    assert output.data == client.serialize(42 ** 2)


@pytest.mark.asyncio
async def test_container_entrypoint_sync_returning_async(servicer):
    t0 = time.time()
    client, outputs = await _run_container(servicer, "polyester.test_support", "square_sync_returning_async")
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 2
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.Status.SUCCESS
    assert output.data == client.serialize(42 ** 2)


@pytest.mark.asyncio
async def test_container_entrypoint_failure(servicer):
    client, outputs = await _run_container(servicer, "polyester.test_support", "raises")

    assert len(outputs) == 2
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)

    output = _get_output(outputs[0])
    assert output.status == api_pb2.GenericResult.Status.FAILURE
    assert output.exception == "Exception('Failure!')"
    assert "Traceback" in output.traceback


def test_import_function_dynamically():
    f = Function.get_function("polyester.test_support", "square")
    assert f(42) == 42 * 42
