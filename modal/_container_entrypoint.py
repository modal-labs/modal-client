import asyncio
import base64
import importlib
import inspect
import math
import os
import sys
import time
import traceback
from typing import Any, AsyncIterator, Callable, List

import cloudpickle
from grpc import StatusCode
from grpc.aio import AioRpcError

from modal_proto import api_pb2
from modal_utils.async_utils import TaskContext, synchronize_apis, synchronizer

from ._asgi import asgi_app_wrapper, fastAPI_function_wrapper
from ._blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._buffer_utils import buffered_rpc_read, buffered_rpc_write
from ._serialization import deserialize, serialize
from .client import Client, _Client
from .config import config, logger
from .exception import InvalidError
from .functions import AioFunction, Function
from .running_app import _RunningApp


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
RTT_S = 0.5  # conservative estimate of RTT in seconds.


class _FunctionContext:
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
        self.start_time = time.time()
        self.calls_completed = 0
        self._client = synchronizer._translate_in(self.client)  # make it a _Client object
        assert isinstance(self._client, _Client)

    @synchronizer.asynccontextmanager
    async def send_outputs(self):
        self.output_queue: asyncio.Queue = asyncio.Queue()

        async with TaskContext(grace=10) as tc:
            tc.create_task(self._send_outputs())
            yield
            await self.output_queue.put((None, None))

    async def initialize_app(self):
        await _RunningApp.init_container(self._client, self.app_id, self.task_id)

    async def get_serialized_function(self) -> Callable:
        # Fetch the serialized function definition
        request = api_pb2.FunctionGetSerializedRequest(function_id=self.function_id)
        response = await self.client.stub.FunctionGetSerialized(request)
        raw_f = cloudpickle.loads(response.function_serialized)

        # TODO(erikbern): there was some code here to create the _Function object,
        # I think related to notebooks, but it was never used. Deleted it for now,
        # will go back to it once we fix notebooks.
        return raw_f

    async def serialize(self, obj: Any) -> bytes:
        return serialize(obj)

    def deserialize(self, data: bytes) -> Any:
        return deserialize(data, self._client)

    async def populate_input_blobs(self, item):
        args = await blob_download(item.args_blob_id, self.client.stub)

        # Mutating
        item.ClearField("args_blob_id")
        item.args = args
        return item

    def get_max_inputs_to_fetch(self):
        if self.calls_completed == 0:
            return 1

        time_elapsed = time.time() - self.start_time
        average_handling_time = time_elapsed / self.calls_completed

        return math.ceil(RTT_S / max(average_handling_time, 1e-6))

    async def generate_inputs(
        self,
    ) -> AsyncIterator[api_pb2.FunctionInput]:
        request = api_pb2.FunctionGetInputsRequest(
            function_id=self.function_id,
            task_id=self.task_id,
        )
        eof_received = False
        while not eof_received:
            request.max_values = self.get_max_inputs_to_fetch()

            try:
                response = await buffered_rpc_read(
                    self.client.stub.FunctionGetInputs, request, timeout=config["container_input_timeout"]
                )
            except AioRpcError as exc:
                if exc.code() == StatusCode.RESOURCE_EXHAUSTED:
                    logger.debug(f"Task {self.task_id} exceeded rate limit.")
                    await asyncio.sleep(1)
                    continue

                if exc.code() == StatusCode.DEADLINE_EXCEEDED:
                    logger.debug(f"Task {self.task_id} input request timed out.")
                    break

                raise

            for item in response.inputs:
                if item.kill_switch:
                    logger.debug(f"Task {self.task_id} input received kill signal.")
                    eof_received = True
                    break

                # If we got a pointer to a blob, download it from S3.
                if item.WhichOneof("args_oneof") == "args_blob_id":
                    yield await self.populate_input_blobs(item)
                else:
                    yield item

                if item.final_input:
                    eof_received = True
                    break

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
        # upload data to S3 if too big.
        if "data" in kwargs and kwargs["data"] and len(kwargs["data"]) > MAX_OBJECT_SIZE_BYTES:
            data_blob_id = await blob_upload(kwargs["data"], self.client.stub)
            # mutating kwargs.
            kwargs.pop("data")
            kwargs["data_blob_id"] = data_blob_id

        result = api_pb2.GenericResult(input_id=input_id, idx=idx, **kwargs)
        await self.output_queue.put((result, function_call_id))

        self.calls_completed += 1


# just to mark the class as synchronized, we don't care about the interfaces
FunctionContext, AioFunctionContext = synchronize_apis(_FunctionContext)


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

    asyncio.run(run_asyncgen())


def call_function(
    function_context,  #: FunctionContext,  # TODO: this type is generated in runtime
    aio_function_context,  #: AioFunctionContext,  # TODO: this one too
    function: Callable,
    function_type: api_pb2.Function.FunctionType,
    function_input: api_pb2.FunctionInput,
):
    input_id = function_input.input_id
    idx = function_input.idx
    function_call_id = function_input.function_call_id

    # TODO: this is somewhat hacky. We need to know whether the function is async or not in order to
    # coerce the input arguments to the right type. The proper way to do is to call the function and
    # see if you get a coroutine (or async generator) back. However at this point, it's too late to
    # coerce the type. For now let's make a determination based on inspecting the function definition.
    # This sometimes isn't correct, since a "vanilla" Python function can return a coroutine if it
    # wraps async code or similar. Let's revisit this shortly.
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        args, kwargs = aio_function_context.deserialize(function_input.args) if function_input.args else ((), {})
    elif inspect.isfunction(function) or inspect.isgeneratorfunction(function):
        args, kwargs = function_context.deserialize(function_input.args) if function_input.args else ((), {})
    else:
        raise RuntimeError(f"Function {function} is a strange type {type(function)}")

    try:
        res = function(*args, **kwargs)

        if function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR:
            if inspect.isgenerator(res):
                _call_function_generator(function_context, function_call_id, input_id, res, idx)
            elif inspect.isasyncgen(res):
                _call_function_asyncgen(aio_function_context, function_call_id, input_id, res, idx)
            else:
                raise InvalidError("Function of type generator returned a non-generator output")

        else:
            if inspect.iscoroutine(res):
                res = asyncio.run(res)

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
        # to unpickle it in some cases). Let's watch out for issues.
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

    # This is a bit weird but we need both the blocking and async versions of FunctionContext.
    # At some point, we should fix that by having built-in support for running "user code"
    _function_context = _FunctionContext(container_args, client)
    function_context, aio_function_context = synchronize_apis(_function_context)
    function_context.initialize_app()

    if container_args.function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
        function = function_context.get_serialized_function()
    else:
        # This is not in function_context, so that any global scope code that runs during import
        # runs on the main thread.
        imported_function = _path_to_function(
            container_args.function_def.module_name, container_args.function_def.function_name
        )
        if isinstance(imported_function, (Function, AioFunction)):
            # We want the internal type of this, not the external
            _function_proxy = synchronizer._translate_in(imported_function)
            function = _function_proxy.get_raw_f()
        else:
            function = imported_function

        if container_args.function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP:
            # function returns an asgi_app, that we can use as a callable.
            asgi_app = function()
            function = asgi_app_wrapper(asgi_app)
        elif container_args.function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
            # function is webhook without an ASGI app. Create one for it.
            function = fastAPI_function_wrapper(function, container_args.function_def.webhook_config.method)

    with function_context.send_outputs():
        for function_input in function_context.generate_inputs():  # type: ignore
            # Note: this blocks the call_function as well. In the future we might want to stream outputs
            # back asynchronously, but then block the call_function if there is back-pressure.
            call_function(function_context, aio_function_context, function, function_type, function_input)  # type: ignore


if __name__ == "__main__":
    logger.debug("Container: starting")

    container_args = api_pb2.ContainerArguments()
    container_args.ParseFromString(base64.b64decode(sys.argv[1]))

    # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
    # This is good because if the function is long running then we the client can still send heartbeats
    # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
    with Client.from_env() as client:
        main(container_args, client)

    logger.debug("Container: done")
