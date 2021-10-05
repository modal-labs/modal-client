import asyncio
import inspect
import sys
import traceback
import typing
import uuid

from .async_utils import synchronizer
from .buffer_utils import buffered_read_all, buffered_write_all
from .client import Client
from .config import logger
from .function import Function
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .object import Object
from .proto import api_pb2


@synchronizer
class FunctionContext:
    """This class isn't much more than a helper method for some gRPC calls."""

    def __init__(self, client, task_id, function_id, input_buffer_id, module_name, function_name):
        self.client = client
        self.task_id = task_id
        self.function_id = function_id
        self.input_buffer_id = input_buffer_id
        self.module_name = module_name
        self.function_name = function_name

    def get_function(self) -> typing.Callable:
        return Function.get_function(self.module_name, self.function_name)

    def get_inputs(
        self,
    ) -> typing.AsyncIterator[api_pb2.BufferReadResponse]:
        request = api_pb2.FunctionGetNextInputRequest(
            function_id=self.function_id,
            task_id=self.task_id,
        )
        return buffered_read_all(self.client.stub.FunctionGetNextInput, request, self.input_buffer_id, read_until_EOF=False)

    async def stream_outputs(self, requests):
        await buffered_write_all(self.client.stub.FunctionOutput, requests)


def call_function(
    function: typing.Callable,
    args: any,
    kwargs: any,
    serializer: typing.Callable[[typing.Any], bytes],
) -> api_pb2.GenericResult:
    try:
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


def main(task_id, function_id, input_buffer_id, module_name, function_name, client=None):
    # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
    # This is good because if the function is long running then we the client can still send heartbeats
    # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
    if client is None:
        client = Client.current()
    function_context = FunctionContext(client, task_id, function_id, input_buffer_id, module_name, function_name)
    function = function_context.get_function()

    async def generate_output_requests():
        async for buffer_item in function_context.get_inputs():
            if buffer_item.EOF:
                break

            input_id = buffer_item.item_id
            args, kwargs, output_buffer_id = client.deserialize(buffer_item.data)

            # function
            output = call_function(
                function,
                args,
                kwargs,
                client.serialize,
            )
            output_bytes = output.SerializeToString()

            buffer_req = api_pb2.BufferWriteRequest(item=api_pb2.BufferItem(data=output_bytes), buffer_id=output_buffer_id)
            request = api_pb2.FunctionOutputRequest(input_id=input_id, buffer_req=buffer_req)
            yield request

    function_context.stream_outputs(generate_output_requests())


if __name__ == "__main__":
    # TODO: we need to do something here to set up the session!
    tag, task_id, function_id, input_buffer_id, module_name, function_name = sys.argv[1:]
    assert tag == "function"
    logger.debug("Container: starting")
    main(task_id, function_id, input_buffer_id, module_name, function_name)
    logger.debug("Container: done")
