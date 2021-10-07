import asyncio
import inspect
import sys
import threading
import traceback
import typing
import uuid

from aiostream import stream

from .async_utils import synchronizer
from .buffer_utils import buffered_read_all, buffered_write_all
from .client import Client
from .config import logger
from .function import Function, pack_output_buffer_item, unpack_input_buffer_item
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
        return buffered_read_all(
            self.client.stub.FunctionGetNextInput, request, self.input_buffer_id, read_until_EOF=False
        )

    async def stream_outputs(self, requests):
        await buffered_write_all(self.client.stub.FunctionOutput, requests)


def make_output_request(input_id, output_buffer_id, **kwargs):
    result = api_pb2.GenericResult(**kwargs)
    item = pack_output_buffer_item(result)
    buffer_req = api_pb2.BufferWriteRequest(item=item, buffer_id=output_buffer_id)

    return api_pb2.FunctionOutputRequest(input_id=input_id, buffer_req=buffer_req)


async def call_function(
    function: typing.Callable,
    buffer_item: api_pb2.BufferItem,
    serializer: typing.Callable[[typing.Any], bytes],
    deserializer: typing.Callable[[bytes], typing.Any],
) -> (str, api_pb2.GenericResult):
    input_id = buffer_item.item_id

    input = unpack_input_buffer_item(buffer_item)
    args = deserializer(input.args)
    kwargs = deserializer(input.kwargs)
    output_buffer_id = input.output_buffer_id

    try:
        res = function(*args, **kwargs)

        if inspect.iscoroutine(res):
            res = await res

        if inspect.isgenerator(res):
            for value in res:
                yield make_output_request(
                    input_id,
                    output_buffer_id,
                    status=api_pb2.GenericResult.Status.SUCCESS,
                    data=serializer(value),
                    incomplete=True,
                )
            return

        yield make_output_request(
            input_id,
            output_buffer_id,
            status=api_pb2.GenericResult.Status.SUCCESS,
            data=serializer(res),
        )

    except Exception as exc:
        # Note that we have to stringify the exception/traceback since
        # it isn't always possible to unpickle on the client side
        yield make_output_request(
            input_id,
            output_buffer_id,
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

    async def generate_outputs():
        try:
            # Create a stream for each call_function; needed so we can multiplex generator outputs.
            generators = []

            async for buffer_item in function_context.get_inputs():
                gen = call_function(function, buffer_item, client.serialize, client.deserialize)
                generators.append(gen)

            # TODO: draining the input iterator should also happen async.
            interleaved = stream.merge(*generators)

            async with interleaved.stream() as streamer:
                async for output in streamer:
                    yield output

        except Exception as exc:
            logger.exception(f"Failed generating outputs: {repr(exc)}")

    function_context.stream_outputs(generate_outputs())


if __name__ == "__main__":
    # TODO: we need to do something here to set up the session!
    tag, task_id, function_id, input_buffer_id, module_name, function_name = sys.argv[1:]
    assert tag == "function"
    logger.debug("Container: starting")
    main(task_id, function_id, input_buffer_id, module_name, function_name)
    logger.debug("Container: done")
