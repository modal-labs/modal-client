import asyncio
import importlib
import inspect
import sys
import threading
import traceback
import typing
import uuid

import aiostream
import cloudpickle
import google.protobuf.json_format

from .async_utils import asyncio_run, synchronizer
from .buffer_utils import buffered_rpc_read, buffered_rpc_write
from .client import Client
from .config import logger
from .function import Function, pack_output_buffer_item, unpack_input_buffer_item
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .object import Object
from .proto import api_pb2
from .session import Session


def _path_to_function(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except ModuleNotFoundError:
        # Just print some debug stuff, then re-raise
        logger.info(f"cwd: {os.getcwd()}")
        logger.info(f"path: {sys.path}")
        logger.info(f"ls: {os.listdir()}")
        raise


@synchronizer
class FunctionContext:
    """This class isn't much more than a helper method for some gRPC calls."""

    def __init__(self, container_args, client):
        self.task_id = container_args.task_id
        self.function_id = container_args.function_id
        self.input_buffer_id = container_args.input_buffer_id
        self.session_id = container_args.session_id
        self.function_def = container_args.function_def
        self.client = client

    async def serialize(self, obj: typing.Any) -> bytes:
        # Call session.flush first. We need this because the function might have defined new objects
        # that have not been created on the server-side, but are part of the return value of the function.
        await self.session.flush_objects()

        return self.session.serialize(obj)

    def deserialize(self, data: bytes) -> typing.Any:
        return self.session.deserialize(data)

    async def get_function(self) -> typing.Callable:
        """Note that this also initializes the session."""

        if self.function_def.definition_type == api_pb2.Function.DefinitionType.SERIALIZED:
            # Fetch the serialized function definition
            request = api_pb2.FunctionGetSerializedRequest(function_id=self.function_id)
            response = await self.client.stub.FunctionGetSerialized(request)
            raw_f = cloudpickle.loads(response.function_serialized)

            # Create a new session object. It will get initialized with the right object ID next.
            self.session = Session()
            fun = Function(self.session, raw_f)
            await self.session.initialize(self.session_id, self.client)
            # Function object is already created, so we need to associate the correct object ID.
            self.session._object_ids[fun.tag] = self.function_id

        else:
            fun = _path_to_function(self.function_def.module_name, self.function_def.function_name)
            assert isinstance(fun, Function)

            self.session = fun.session
            await self.session.initialize(self.session_id, self.client)

        return fun.get_raw_f()

    async def generate_inputs(
        self,
    ) -> typing.AsyncIterator[api_pb2.BufferReadResponse]:
        request = api_pb2.FunctionGetNextInputRequest(
            function_id=self.function_id,
            task_id=self.task_id,
        )
        while True:
            response = await buffered_rpc_read(
                self.client.stub.FunctionGetNextInput, request, self.input_buffer_id, timeout=GRPC_REQUEST_TIMEOUT
            )

            if response.status == api_pb2.BufferReadResponse.BufferReadStatus.TIMEOUT:
                logger.info(f"Task {self.task_id} input request timed out.")
                break

            if response.item.EOF:
                logger.debug(f"Task {self.task_id} input got EOF.")
                break

            yield response.item

    async def _output(self, request):
        # No timeout so this can block forever.
        await buffered_rpc_write(self.client.stub.FunctionOutput, request)

    async def output_request(self, input_id, output_buffer_id, **kwargs):
        result = api_pb2.GenericResult(**kwargs)
        item = pack_output_buffer_item(result)
        buffer_req = api_pb2.BufferWriteRequest(item=item, buffer_id=output_buffer_id)
        req = api_pb2.FunctionOutputRequest(input_id=input_id, buffer_req=buffer_req)
        return await self._output(req)


def call_function(
    function_context: FunctionContext,
    function: typing.Callable,
    buffer_item: api_pb2.BufferItem,
):
    input = unpack_input_buffer_item(buffer_item)
    output_buffer_id = input.output_buffer_id

    input_id = buffer_item.item_id
    args = function_context.deserialize(input.args)
    kwargs = function_context.deserialize(input.kwargs)

    try:
        res = function(*args, **kwargs)

        if inspect.isgenerator(res):
            for value in res:
                function_context.output_request(
                    input_id,
                    output_buffer_id,
                    status=api_pb2.GenericResult.Status.SUCCESS,
                    data=function_context.serialize(value),
                    gen_status=api_pb2.GenericResult.GeneratorStatus.INCOMPLETE,
                )

            # send EOF
            function_context.output_request(
                input_id,
                output_buffer_id,
                status=api_pb2.GenericResult.Status.SUCCESS,
                gen_status=api_pb2.GenericResult.GeneratorStatus.COMPLETE,
            )
        elif inspect.isasyncgen(res):

            async def run_asyncgen():
                async for value in res:
                    await function_context.output_request(
                        input_id,
                        output_buffer_id,
                        status=api_pb2.GenericResult.Status.SUCCESS,
                        data=await function_context.serialize(value),
                        gen_status=api_pb2.GenericResult.GeneratorStatus.INCOMPLETE,
                    )

                # send EOF
                await function_context.output_request(
                    input_id,
                    output_buffer_id,
                    status=api_pb2.GenericResult.Status.SUCCESS,
                    gen_status=api_pb2.GenericResult.GeneratorStatus.COMPLETE,
                )

            asyncio_run(run_asyncgen())

        else:
            if inspect.iscoroutine(res):
                res = asyncio_run(res)

            function_context.output_request(
                input_id,
                output_buffer_id,
                status=api_pb2.GenericResult.Status.SUCCESS,
                data=function_context.serialize(res),
            )

    except Exception as exc:
        # Note: we're not serializing the traceback since it contains
        # local references that means we can't unpickle it. We *are*
        # serializing the exception, which may have some issues (there
        # was an earlier note about it that it might not be possible
        # to unpickle it in some cases). Let's watch oout for issues.
        function_context.output_request(
            input_id,
            output_buffer_id,
            status=api_pb2.GenericResult.Status.FAILURE,
            data=function_context.serialize(exc),
            exception=repr(exc),
            traceback=traceback.format_exc(),
        )


def main(container_args, client):
    function_context = FunctionContext(container_args, client)
    function = function_context.get_function()

    for buffer_item in function_context.generate_inputs():
        # Note: this blocks the call_function as well. In the future we might want to stream outputs
        # back asynchronously, but then block the call_function if there is back-pressure.
        call_function(function_context, function, buffer_item)


if __name__ == "__main__":
    logger.debug("Container: starting")
    container_args = google.protobuf.json_format.Parse(
        sys.argv[1],
        api_pb2.ContainerArguments(),
    )
    # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
    # This is good because if the function is long running then we the client can still send heartbeats
    # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
    with Client.from_env() as client:
        main(container_args, client)

    logger.debug("Container: done")
