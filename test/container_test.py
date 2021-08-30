import pytest

from polyester.client import Client
from polyester.container_entrypoint import FunctionRunner
from polyester.proto import api_pb2
from polyester.function import Function


def get_inputs(client):
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


def square(x):
    return x**2


def raises(x):
    raise Exception('Failure!')


class FakeContainerClient:
    def __init__(self):
        self.inputs = [
            (((42,), {}), 'in-123', False),
            (None, None, True),
        ]
        self.outputs = []

    async def function_get_next_input(self, task_id, function_id):
        return self.inputs.pop(0)

    async def function_output(self, input_id, status, data, exception, traceback):
        self.outputs.append((status, data, exception, traceback))


@pytest.mark.asyncio
async def test_container_entrypoint_success(servicer):
    client = Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ('ta-123', 'task-secret'))

    await client._start()
    await client._start_client()

    servicer.inputs = get_inputs(client)
    function_runner = FunctionRunner(client, 'ta-123', 'fu-123', square)
    await function_runner.run()
    output = servicer.requests[-1]
    assert isinstance(output, api_pb2.FunctionOutputRequest)

    assert output.output.status == api_pb2.GenericResult.Status.SUCCESS
    assert output.output.data == client.serialize(42**2)


@pytest.mark.asyncio
async def test_container_entrypoint_failure(servicer):
    client = Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ('ta-123', 'task-secret'))

    await client._start()
    await client._start_client()

    servicer.inputs = get_inputs(client)
    function_runner = FunctionRunner(client, 'ta-123', 'fu-123', raises)
    await function_runner.run()
    output = servicer.requests[-1]
    assert isinstance(output, api_pb2.FunctionOutputRequest)

    assert output.output.status == api_pb2.GenericResult.Status.FAILURE
    assert output.output.exception == 'Exception(\'Failure!\')'
    assert 'Traceback' in output.output.traceback


def test_import_function_dynamically():
    f = Function.get_function('polyester.test_support', 'square')
    assert f(42) == 42*42
