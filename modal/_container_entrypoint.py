import asyncio
import importlib
import inspect
import os
import sys
import traceback
from typing import Any, AsyncIterator, Callable, List

import cloudpickle
import google.protobuf.json_format

from ._async_utils import TaskContext, asyncio_run, synchronizer
from ._buffer_utils import buffered_rpc_read, buffered_rpc_write
from ._client import Client
from .app import App
from .config import config, logger
from .exception import InvalidError
from .functions import Function
from .proto import api_pb2


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
        self.app_id = container_args.app_id
        self.function_def = container_args.function_def
        self.client = client

    @synchronizer.asynccontextmanager
    async def send_outputs(self):
        self.output_queue: asyncio.Queue = asyncio.Queue()

        async with TaskContext(grace=10) as tc:
            tc.create_task(self._send_outputs())
            yield
            await self.output_queue.put((None, None))

    async def get_function(self) -> Callable:
        """Note that this also initializes the app."""

        # On the container, we know we're inside a app, so we initialize all App
        # objects with the same singleton object. This then lets us pull the lookup
        # table of all the named objects
        App.initialize_container_app()
        self.app = App()
        await self.app.initialize_container(self.app_id, self.client, self.task_id)

        if self.function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
            # Fetch the serialized function definition
            request = api_pb2.FunctionGetSerializedRequest(function_id=self.function_id)
            response = await self.client.stub.FunctionGetSerialized(request)
            raw_f = cloudpickle.loads(response.function_serialized)

            # Create a function dynamically
            # Function object is already created, so we need to associate the correct object ID.
            fun = Function(raw_f)
            fun.set_object_id(self.function_id, self.app)
        else:
            fun = _path_to_function(self.function_def.module_name, self.function_def.function_name)
            assert isinstance(fun, Function)

        return fun.get_raw_f()

    async def serialize(self, obj: Any) -> bytes:
        return self.app.serialize(obj)

    def deserialize(self, data: bytes) -> Any:
        return self.app.deserialize(data)

    async def generate_inputs(
        self,
    ) -> AsyncIterator[api_pb2.FunctionInput]:
        request = api_pb2.FunctionGetInputsRequest(
            function_id=self.function_id,
            task_id=self.task_id,
        )
        eof_received = False
        while not eof_received:
            response = await buffered_rpc_read(
                self.client.stub.FunctionGetInputs, request, timeout=config["container_input_timeout"]
            )

            if response.status == api_pb2.READ_STATUS_RATE_LIMIT_EXCEEDED:
                logger.info(f"Task {self.task_id} exceeded rate limit.")
                await asyncio.sleep(1)
                continue

            if response.status == api_pb2.READ_STATUS_TIMEOUT:
                logger.info(f"Task {self.task_id} input request timed out.")
                break

            for item in response.inputs:
                if item.EOF:
                    logger.debug(f"Task {self.task_id} input got EOF.")
                    eof_received = True
                    break

                yield item

    async def _send_outputs(self):
        """Background task that tries to drain output queue until it's empty,
        or the output buffer changes, and then sends the entire batch in one request.
        """
        cur_function_call_id = None
        outputs: List[Any] = []

        async def _send():
            nonlocal outputs, cur_function_call_id

            if not outputs:
                return

            req = api_pb2.FunctionPutOutputsRequest(
                outputs=outputs, function_call_id=cur_function_call_id, task_id=self.task_id
            )
            # No timeout so this can block forever.
            await buffered_rpc_write(self.client.stub.FunctionPutOutputs, req)

            cur_function_call_id = None
            outputs = []

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

            (item, function_call_id) = output

            # No more inputs coming.
            if item is None:
                await _send()
                break

            function_call_id_changed = cur_function_call_id is not None and function_call_id != cur_function_call_id

            # Send what we have so far for this output buffer and switch tracks
            if function_call_id_changed or len(outputs) >= MAX_OUTPUT_BATCH_SIZE:
                await _send()
                outputs = [item]
            else:
                outputs.append(item)

            cur_function_call_id = function_call_id

    async def enqueue_output(self, function_call_id, input_id, idx, **kwargs):
        result = api_pb2.GenericResult(input_id=input_id, idx=idx, **kwargs)
        await self.output_queue.put((result, function_call_id))


def _call_function_generator(function_context, function_call_id, input_id, res, idx):
    for value in res:
        function_context.enqueue_output(
            function_call_id,
            input_id,
            idx,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=function_context.serialize(value),
            gen_status=api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE,
        )

    # send EOF
    function_context.enqueue_output(
        function_call_id,
        input_id,
        idx,
        status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
        gen_status=api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE,
    )


def _call_function_asyncgen(function_context, function_call_id, input_id, res, idx):
    async def run_asyncgen():
        async for value in res:
            await function_context.enqueue_output(
                function_call_id,
                input_id,
                idx,
                status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                data=await function_context.serialize(value),
                gen_status=api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE,
            )

        # send EOF
        await function_context.enqueue_output(
            function_call_id,
            input_id,
            idx,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            gen_status=api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE,
        )

    asyncio_run(run_asyncgen())


def call_function(
    function_context: FunctionContext,
    function: Callable,
    function_type: api_pb2.Function.FunctionType,
    function_input: api_pb2.FunctionInput,
):
    input_id = function_input.input_id
    args = function_context.deserialize(function_input.args) if function_input.args else ()
    kwargs = function_context.deserialize(function_input.kwargs) if function_input.kwargs else {}
    idx = function_input.idx
    function_call_id = function_input.function_call_id

    try:
        res = function(*args, **kwargs)

        if function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR:
            if inspect.isgenerator(res):
                _call_function_generator(function_context, function_call_id, input_id, res, idx)
            elif inspect.isasyncgen(res):
                _call_function_asyncgen(function_context, function_call_id, input_id, res, idx)
            else:
                raise InvalidError("Function of type generator returned a non-generator output")

        else:
            if inspect.iscoroutine(res):
                res = asyncio_run(res)

            if inspect.isgenerator(res) or inspect.isasyncgen(res):
                raise InvalidError("Function which is not a generator returned a generator output")

            function_context.enqueue_output(
                function_call_id,
                input_id,
                idx,
                status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                data=function_context.serialize(res),
            )

    except Exception as exc:
        # print exception so it's logged
        traceback.print_exc()

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
            function_call_id,
            input_id,
            idx,
            status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
            data=serialized_exc,
            exception=repr(exc),
            traceback=traceback.format_exc(),
        )


def main(container_args, client):
    # TODO: if there's an exception in this scope (in particular when we import code dynamically),
    # we could catch that exception and set it properly serialized to the client. Right now the
    # whole container fails with a non-zero exit code and we send back a more opaque error message.
    function_type = container_args.function_def.function_type

    function_context = FunctionContext(container_args, client)
    function = function_context.get_function()

    with function_context.send_outputs():
        for function_input in function_context.generate_inputs():  # type: ignore
            # Note: this blocks the call_function as well. In the future we might want to stream outputs
            # back asynchronously, but then block the call_function if there is back-pressure.
            call_function(function_context, function, function_type, function_input)  # type: ignore


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
