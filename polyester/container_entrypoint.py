import asyncio
import inspect
import sys
import traceback
import typing
import uuid

from .async_utils import retry, synchronizer
from .client import Client
from .config import logger
from .function import Function
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .object import Object
from .proto import api_pb2


@synchronizer
class FunctionContext:
    """This class isn't much more than a helper method for some gRPC calls."""

    def __init__(self, client, task_id, function_id, module_name, function_name):
        self.client = client
        self.task_id = task_id
        self.function_id = function_id
        self.module_name = module_name
        self.function_name = function_name

    def get_function(self) -> typing.Callable:
        return Function.get_function(self.module_name, self.function_name)

    async def get_inputs(
        self,
    ) -> typing.AsyncIterator[api_pb2.FunctionGetNextInputResponse]:
        while True:
            idempotency_key = str(uuid.uuid4())
            request = api_pb2.FunctionGetNextInputRequest(
                task_id=self.task_id,
                function_id=self.function_id,
                idempotency_key=idempotency_key,
                timeout=BLOCKING_REQUEST_TIMEOUT,
            )
            response = await retry(self.client.stub.FunctionGetNextInput)(request, timeout=GRPC_REQUEST_TIMEOUT)
            if not response.data:
                logger.info(f"Task {self.task_id} input request received no data.")
                break
            if response.stop:
                logger.info(f"Task {self.task_id} received stop response.")
                break
            yield response

    async def output(self, input_id: str, output: api_pb2.GenericResult):
        idempotency_key = str(uuid.uuid4())
        request = api_pb2.FunctionOutputRequest(input_id=input_id, idempotency_key=idempotency_key, output=output)
        await retry(self.client.stub.FunctionOutput)(request)


def call_function(
    function: typing.Callable,
    serializer: typing.Callable[[typing.Any], bytes],
    deserializer: typing.Callable[[bytes], typing.Any],
    function_input: api_pb2.FunctionGetNextInputResponse,
) -> api_pb2.GenericResult:
    try:
        args, kwargs = deserializer(function_input.data)
        res = function(*args, **kwargs)
        # TODO: handle generators etc
        if inspect.iscoroutine(res):
            res = asyncio.run(res)

        return api_pb2.GenericResult(
            status=api_pb2.GenericResult.Status.SUCCESS,
            data=serializer(res),
        )

    except Exception as exc:
        # Note that we have to stringify the exception/traceback since
        # it isn't always possible to unpickle on the client side
        return api_pb2.GenericResult(
            status=api_pb2.GenericResult.Status.FAILURE,
            exception=repr(exc),
            traceback=traceback.format_exc(),
        )


def main(task_id, function_id, module_name, function_name, client=None):
    # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
    # This is good because if the function is long running then we the client can still send heartbeats
    # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
    if client is None:
        client = Client.current()
    function_context = FunctionContext(client, task_id, function_id, module_name, function_name)
    function = function_context.get_function()
    for function_input in function_context.get_inputs():
        result = call_function(
            function,
            client.serialize,
            client.deserialize,
            function_input,
        )
        function_context.output(function_input.input_id, result)


if __name__ == "__main__":
    # TODO: we need to do something here to set up the session!
    tag, task_id, function_id, module_name, function_name = sys.argv[1:]
    assert tag == "function"
    logger.debug("Container: starting")
    main(task_id, function_id, module_name, function_name)
    logger.debug("Container: done")
