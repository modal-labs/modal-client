import asyncio
import pytest
import time

from polyester.client import Client
from polyester.container_entrypoint import main
from polyester.proto import api_pb2
from polyester.function import Function
from polyester.test_support import SLEEP_DELAY

EXTRA_TOLERANCE_DELAY = 0.05


def _get_inputs(client):
    return [
        api_pb2.FunctionGetNextInputResponse(
            data=client.serialize(((42,), {})),
            input_id='in-123',
            stop=False,
        ),
        api_pb2.FunctionGetNextInputResponse(
            data=None,
            input_id=None,
            stop=True,
        )
    ]


async def _run_container(servicer, module_name, function_name):
    client = Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ('ta-123', 'task-secret'))

    await client._start()
    await client._start_client()

    servicer.inputs = _get_inputs(client)

    # Note that main is a synchronous function, so we need to run it in a separate thread
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, main, 'ta-123', 'fu-123', module_name, function_name, client)

    return client, servicer.outputs


@pytest.mark.asyncio
async def test_container_entrypoint_success(servicer):
    t0 = time.time()
    client, outputs = await _run_container(servicer, 'polyester.test_support', 'square')
    assert 0 <= time.time() - t0 < EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)
    assert outputs[0].output.status == api_pb2.GenericResult.Status.SUCCESS
    assert outputs[0].output.data == client.serialize(42**2)


@pytest.mark.asyncio
async def test_container_entrypoint_async(servicer):
    t0 = time.time()
    client, outputs = await _run_container(servicer, 'polyester.test_support', 'square_async')
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)
    assert outputs[0].output.status == api_pb2.GenericResult.Status.SUCCESS
    assert outputs[0].output.data == client.serialize(42**2)


@pytest.mark.asyncio
async def test_container_entrypoint_sync_returning_async(servicer):
    t0 = time.time()
    client, outputs = await _run_container(servicer, 'polyester.test_support', 'square_sync_returning_async')
    assert SLEEP_DELAY <= time.time() - t0 < SLEEP_DELAY + EXTRA_TOLERANCE_DELAY

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)
    assert outputs[0].output.status == api_pb2.GenericResult.Status.SUCCESS
    assert outputs[0].output.data == client.serialize(42**2)


@pytest.mark.asyncio
async def test_container_entrypoint_failure(servicer):
    client, outputs = await _run_container(servicer, 'polyester.test_support', 'raises')

    assert len(outputs) == 1
    assert isinstance(outputs[0], api_pb2.FunctionOutputRequest)
    assert outputs[0].output.status == api_pb2.GenericResult.Status.FAILURE
    assert outputs[0].output.exception == 'Exception(\'Failure!\')'
    assert 'Traceback' in outputs[0].output.traceback


def test_import_function_dynamically():
    f = Function.get_function('polyester.test_support', 'square')
    assert f(42) == 42*42
