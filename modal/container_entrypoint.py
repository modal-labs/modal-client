import asyncio
import importlib
import inspect
import os
import sys
import threading
import traceback
import typing
import uuid

import aiostream
import cloudpickle
import google.protobuf.json_format

from .async_utils import TaskContext, asyncio_run, synchronizer
from .buffer_utils import buffered_rpc_read, buffered_rpc_write
from .client import Client
from .config import logger
from .exception import InvalidError
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


MAX_OUTPUT_BATCH_SIZE = 100


@synchronizer
class FunctionContext:
    """This class isn't much more than a helper method for some gRPC calls.

    TODO: maybe we shouldn't synchronize the whole class.
    Then we could potentially move a bunch of the global functions onto it.
    """

    def __init__(self, container_args, client):
        self.task_id = container_args.task_id
        self.function_id = container_args.function_id
        self.input_buffer_id = container_args.input_buffer_id
        self.session_id = container_args.session_id
        self.function_def = container_args.function_def
        self.client = client

    @synchronizer.asynccontextmanager
    async def send_outputs(self):
        self.output_queue = asyncio.Queue()

        async with TaskContext(grace=10) as tc:
            tc.create_task(self._send_outputs())
            yield
            await self.output_queue.put((None, None))

    async def get_function(self) -> typing.Callable:
        """Note that this also initializes the session."""

        # On the container, we know we're inside a session, so we initialize all Session
        # objects with the same singleton object. This then lets us pull the lookup
        # table of all the named objects
        Session.initialize_singleton()
        self.session = Session()
        await self.session.initialize_container(self.session_id, self.client, self.task_id)

        if self.function_def.definition_type == api_pb2.Function.DefinitionType.SERIALIZED:
            # Fetch the serialized function definition
            request = api_pb2.FunctionGetSerializedRequest(function_id=self.function_id)
            response = await self.client.stub.FunctionGetSerialized(request)
            raw_f = cloudpickle.loads(response.function_serialized)

            # Create a function dynamically
            # Function object is already created, so we need to associate the correct object ID.
            fun = Function(self.session, raw_f)
            fun.set_object_id(self.function_id, self.session_id)
        else:
            fun = _path_to_function(self.function_def.module_name, self.function_def.function_name)
            assert isinstance(fun, Function)

        return fun.get_raw_f()

    async def serialize(self, obj: typing.Any) -> bytes:
        # Call session.flush first. We need this because the function might have defined new objects
        # that have not been created on the server-side, but are part of the return value of the function.
        await self.session.flush_objects()

        return self.session.serialize(obj)

    def deserialize(self, data: bytes) -> typing.Any:
        return self.session.deserialize(data)

    async def generate_inputs(
        self,
    ) -> typing.AsyncIterator[api_pb2.BufferReadResponse]:
        request = api_pb2.FunctionGetNextInputRequest(
            function_id=self.function_id,
            task_id=self.task_id,
        )
        eof_received = False
        while not eof_received:
            response = await buffered_rpc_read(
                self.client.stub.FunctionGetNextInput, request, self.input_buffer_id, timeout=GRPC_REQUEST_TIMEOUT
            )

            if response.status == api_pb2.BufferReadResponse.BufferReadStatus.TIMEOUT:
                logger.info(f"Task {self.task_id} input request timed out.")
                break

            for item in response.items:
                if item.EOF:
                    logger.debug(f"Task {self.task_id} input got EOF.")
                    eof_received = True
                    break

                yield item

    async def _send_outputs(self):
        """Background task that tries to drain output queue until it's empty,
        or the output buffer changes, and then sends the entire batch in one request.
        """
        cur_output_buffer_id = None
        items = []

        async def _send():
            nonlocal items, cur_output_buffer_id

            if not items:
                return

            buffer_req = api_pb2.BufferWriteRequest(items=items, buffer_id=cur_output_buffer_id)
            req = api_pb2.FunctionOutputRequest(buffer_req=buffer_req, task_id=self.task_id)
            # No timeout so this can block forever.
            await buffered_rpc_write(self.client.stub.FunctionOutput, req)

            cur_output_buffer_id = None
            items = []

        def _try_get_output():
            try:
                return self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                return None

        while True:
            output = _try_get_output()

            if output is None:
                # Queue is empty, send what we have so far.
                await _send()
                # Block until the queue has values again.
                output = await self.output_queue.get()

            (item, output_buffer_id) = output

            # No more inputs coming.
            if item is None:
                await _send()
                break

            output_buffer_changed = cur_output_buffer_id is not None and output_buffer_id != cur_output_buffer_id
            cur_output_buffer_id = output_buffer_id

            # Send what we have so far for this output buffer and switch tracks
            if output_buffer_changed or len(items) >= MAX_OUTPUT_BATCH_SIZE:
                await _send()
                items = [item]
            else:
                items.append(item)

    async def enqueue_output(self, input_id, output_buffer_id, **kwargs):
        result = api_pb2.GenericResult(**kwargs)
        result.input_id = input_id
        item = pack_output_buffer_item(result)
        await self.output_queue.put((item, output_buffer_id))


def _call_function_generator(function_context, input_id, output_buffer_id, res):
    for value in res:
        function_context.enqueue_output(
            input_id,
            output_buffer_id,
            status=api_pb2.GenericResult.Status.SUCCESS,
            data=function_context.serialize(value),
            gen_status=api_pb2.GenericResult.GeneratorStatus.INCOMPLETE,
        )

    # send EOF
    function_context.enqueue_output(
        input_id,
        output_buffer_id,
        status=api_pb2.GenericResult.Status.SUCCESS,
        gen_status=api_pb2.GenericResult.GeneratorStatus.COMPLETE,
    )


def _call_function_asyncgen(function_context, input_id, output_buffer_id, res):
    async def run_asyncgen():
        async for value in res:
            await function_context.enqueue_output(
                input_id,
                output_buffer_id,
                status=api_pb2.GenericResult.Status.SUCCESS,
                data=await function_context.serialize(value),
                gen_status=api_pb2.GenericResult.GeneratorStatus.INCOMPLETE,
            )

        # send EOF
        await function_context.enqueue_output(
            input_id,
            output_buffer_id,
            status=api_pb2.GenericResult.Status.SUCCESS,
            gen_status=api_pb2.GenericResult.GeneratorStatus.COMPLETE,
        )

    asyncio_run(run_asyncgen())


def call_function(
    function_context: FunctionContext,
    function: typing.Callable,
    function_type: api_pb2.Function.FunctionType,
    buffer_item: api_pb2.BufferItem,
):
    input = unpack_input_buffer_item(buffer_item)
    output_buffer_id = input.output_buffer_id

    input_id = buffer_item.item_id
    args = function_context.deserialize(input.args)
    kwargs = function_context.deserialize(input.kwargs)

    try:
        res = function(*args, **kwargs)

        if function_type == api_pb2.Function.FunctionType.GENERATOR:
            if inspect.isgenerator(res):
                _call_function_generator(function_context, input_id, output_buffer_id, res)
            elif inspect.isasyncgen(res):
                _call_function_asyncgen(function_context, input_id, output_buffer_id, res)
            else:
                raise InvalidError("Function of type generator returned a non-generator output")

        elif function_type == api_pb2.Function.FunctionType.FUNCTION:
            if inspect.iscoroutine(res):
                res = asyncio_run(res)

            if inspect.isgenerator(res) or inspect.isasyncgen(res):
                raise InvalidError("Function which is not a generator returned a generator output")

            function_context.enqueue_output(
                input_id,
                output_buffer_id,
                status=api_pb2.GenericResult.Status.SUCCESS,
                data=function_context.serialize(res),
            )

        else:
            raise InvalidError(f"Unknown function type {function_type}")

    except Exception as exc:
        try:
            serialized_exc = function_context.serialize(exc)
        except Exception:
            # We can't always serialize exceptions.
            serialized_exc = None

        # Note: we're not serializing the traceback since it contains
        # local references that means we can't unpickle it. We *are*
        # serializing the exception, which may have some issues (there
        # was an earlier note about it that it might not be possible
        # to unpickle it in some cases). Let's watch oout for issues.
        function_context.enqueue_output(
            input_id,
            output_buffer_id,
            status=api_pb2.GenericResult.Status.FAILURE,
            data=serialized_exc,
            exception=repr(exc),
            traceback=traceback.format_exc(),
        )


def main(container_args, client):
    function_type = container_args.function_def.function_type

    function_context = FunctionContext(container_args, client)
    function = function_context.get_function()

    with function_context.send_outputs():
        for buffer_item in function_context.generate_inputs():
            # Note: this blocks the call_function as well. In the future we might want to stream outputs
            # back asynchronously, but then block the call_function if there is back-pressure.
            call_function(function_context, function, function_type, buffer_item)


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
