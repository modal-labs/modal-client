import asyncio
import base64
import importlib
import inspect
import math
import sys
import time
import traceback
from typing import Any, AsyncIterator, Callable, Tuple

import cloudpickle

from modal_proto import api_pb2
from modal_utils.async_utils import (
    TaskContext,
    queue_batch_iterator,
    synchronize_apis,
    synchronizer,
)
from modal_utils.grpc_utils import retry_transient_errors

from ._asgi import asgi_app_wrapper, fastAPI_function_wrapper, wsgi_app_wrapper
from ._blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._proxy_tunnel import proxy_tunnel
from ._serialization import deserialize, serialize
from .app import _App
from .client import Client, _Client
from .config import logger
from .exception import InvalidError
from .functions import AioFunctionHandle, FunctionHandle, _set_current_input_id


def _path_to_function(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


MAX_OUTPUT_BATCH_SIZE = 100
RTT_S = 0.5  # conservative estimate of RTT in seconds.

CONTAINER_IDLE_TIMEOUT = 60


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
        self.calls_completed = 0
        self.total_user_time: float = 0
        self._client = synchronizer._translate_in(self.client)  # make it a _Client object
        assert isinstance(self._client, _Client)

    async def initialize_app(self):
        await _App._init_container(self._client, self.app_id, self.task_id)

    async def get_serialized_function(self) -> Callable:
        # Fetch the serialized function definition
        request = api_pb2.FunctionGetSerializedRequest(function_id=self.function_id)
        response = await self.client.stub.FunctionGetSerialized(request)
        raw_f = cloudpickle.loads(response.function_serialized)

        # TODO(erikbern): there was some code here to create the _Function object,
        # I think related to notebooks, but it was never used. Deleted it for now,
        # will go back to it once we fix notebooks.
        return raw_f

    def serialize(self, obj: Any) -> bytes:
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

        average_handling_time = self.total_user_time / self.calls_completed

        return math.ceil(RTT_S / max(average_handling_time, 1e-6))

    async def _generate_inputs(
        self,
    ) -> AsyncIterator[Tuple[str, api_pb2.FunctionInput]]:
        request = api_pb2.FunctionGetInputsRequest(function_id=self.function_id)
        eof_received = False
        last_input = time.time()
        while not eof_received:
            time_left = last_input + CONTAINER_IDLE_TIMEOUT - time.time()

            request.max_values = self.get_max_inputs_to_fetch()
            # clamp to between 0.01 and 15s.
            request.timeout = min(max(time_left, 0.01), 15)

            response = await retry_transient_errors(self.client.stub.FunctionGetInputs, request)
            if response.rate_limit_sleep_duration:
                logger.info(
                    "Task exceeded rate limit, sleeping for %.2fs before trying again."
                    % response.rate_limit_sleep_duration
                )
                await asyncio.sleep(response.rate_limit_sleep_duration)
                continue

            if not response.inputs:
                if time_left < 0:
                    logger.debug(f"Task {self.task_id} reached idle time-out.")
                    break

                continue

            last_input = time.time()

            for item in response.inputs:
                if item.kill_switch:
                    logger.debug(f"Task {self.task_id} input received kill signal.")
                    eof_received = True
                    break

                # If we got a pointer to a blob, download it from S3.
                if item.input.WhichOneof("args_oneof") == "args_blob_id":
                    input_pb = await self.populate_input_blobs(item.input)
                else:
                    input_pb = item.input

                yield (item.input_id, input_pb)

                if item.input.final_input:
                    eof_received = True
                    break

    async def _send_outputs(self):
        """Background task that tries to drain output queue until it's empty,
        or the output buffer changes, and then sends the entire batch in one request.
        """
        async for outputs in queue_batch_iterator(self.output_queue, MAX_OUTPUT_BATCH_SIZE, 0):
            req = api_pb2.FunctionPutOutputsRequest(outputs=outputs)
            # No timeout so this can block forever.
            await retry_transient_errors(self.client.stub.FunctionPutOutputs, req, max_retries=None)

    async def run_inputs_outputs(self):
        # This also makes sure to terminate the outputs
        self.output_queue: asyncio.Queue = asyncio.Queue()

        async with TaskContext(grace=10) as tc:
            tc.create_task(self._send_outputs())
            try:
                async for input_id, input_pb in self._generate_inputs():
                    _set_current_input_id(input_id)
                    t0 = time.time()
                    try:
                        yield input_id, input_pb
                    finally:
                        self.total_user_time += time.time() - t0
                        self.calls_completed += 1
            finally:
                await self.output_queue.put(None)

    async def enqueue_output(self, input_id, **kwargs):
        # upload data to S3 if too big.
        if "data" in kwargs and kwargs["data"] and len(kwargs["data"]) > MAX_OBJECT_SIZE_BYTES:
            data_blob_id = await blob_upload(kwargs["data"], self.client.stub)
            # mutating kwargs.
            kwargs.pop("data")
            kwargs["data_blob_id"] = data_blob_id

        output = api_pb2.FunctionPutOutputsItem(input_id=input_id, result=api_pb2.GenericResult(**kwargs))
        await self.output_queue.put(output)

    @synchronizer.asynccontextmanager
    async def handle_general_exception(self):
        try:
            yield
        except Exception as exc:
            # Since this is on a different thread, sys.exc_info() can't find the exception in the stack.
            traceback.print_exception(type(exc), exc, exc.__traceback__)

            try:
                serialized_exc = self.serialize(exc)
            except Exception as serialization_exc:
                logger.info(f"Failed to serialize exception {exc}: {serialization_exc}")
                # We can't always serialize exceptions.
                serialized_exc = None

            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                data=serialized_exc,
                exception=repr(exc),
                traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )

            req = api_pb2.TaskResultRequest(task_id=self.task_id, result=result)
            await retry_transient_errors(self.client.stub.TaskResult, req)

    @synchronizer.asynccontextmanager
    async def handle_input_exception(self, input_id):
        try:
            yield
        except Exception as exc:
            # print exception so it's logged
            traceback.print_exc()

            try:
                serialized_exc = self.serialize(exc)
            except Exception as serialization_exc:
                logger.info(f"Failed serializing exception {exc}: {serialization_exc}")
                # We can't always serialize exceptions.
                serialized_exc = None

            # Note: we're not serializing the traceback since it contains
            # local references that means we can't unpickle it. We *are*
            # serializing the exception, which may have some issues (there
            # was an earlier note about it that it might not be possible
            # to unpickle it in some cases). Let's watch out for issues.
            await self.enqueue_output(
                input_id,
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                data=serialized_exc,
                exception=repr(exc),
                traceback=traceback.format_exc(),
            )


# just to mark the class as synchronized, we don't care about the interfaces
FunctionContext, AioFunctionContext = synchronize_apis(_FunctionContext)


def _call_function_generator(function_context, input_id, res):
    for value in res:
        function_context.enqueue_output(
            input_id,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=function_context.serialize(value),
            gen_status=api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE,
        )

    # send EOF
    function_context.enqueue_output(
        input_id,
        status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
        gen_status=api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE,
    )


async def _call_function_asyncgen(aio_function_context, input_id, res):
    async for value in res:
        await aio_function_context.enqueue_output(
            input_id,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=aio_function_context.serialize(value),
            gen_status=api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE,
        )

    # send EOF
    await aio_function_context.enqueue_output(
        input_id,
        status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
        gen_status=api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE,
    )


def is_async(function):
    # TODO: this is somewhat hacky. We need to know whether the function is async or not in order to
    # coerce the input arguments to the right type. The proper way to do is to call the function and
    # see if you get a coroutine (or async generator) back. However at this point, it's too late to
    # coerce the type. For now let's make a determination based on inspecting the function definition.
    # This sometimes isn't correct, since a "vanilla" Python function can return a coroutine if it
    # wraps async code or similar. Let's revisit this shortly.
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        return True
    elif inspect.isfunction(function) or inspect.isgeneratorfunction(function):
        return False
    else:
        raise RuntimeError(f"Function {function} is a strange type {type(function)}")


def call_function_sync(
    function_context,  #: FunctionContext,  # TODO: this type is generated in runtime
    function: Callable,
    function_type: api_pb2.Function.FunctionType,
    input_id: str,
    input_pb: api_pb2.FunctionInput,
):
    args, kwargs = function_context.deserialize(input_pb.args) if input_pb.args else ((), {})

    res = function(*args, **kwargs)

    if function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR:
        if not inspect.isgenerator(res):
            raise InvalidError(f"Generator function returned value of type {type(res)}")
        _call_function_generator(function_context, input_id, res)
    else:
        if inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
            raise InvalidError(f"Sync (non-generator) function return value of type {type(res)}")
        function_context.enqueue_output(
            input_id,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=function_context.serialize(res),
        )


async def call_function_async(
    aio_function_context,  #: AioFunctionContext,  # TODO: this one too
    function: Callable,
    function_type: api_pb2.Function.FunctionType,
    input_id: str,
    input_pb: api_pb2.FunctionInput,
):
    args, kwargs = aio_function_context.deserialize(input_pb.args) if input_pb.args else ((), {})

    res = function(*args, **kwargs)

    if function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR:
        if not inspect.isasyncgen(res):
            raise InvalidError(f"Async generator function returned value of type {type(res)}")
        await _call_function_asyncgen(aio_function_context, input_id, res)
    else:
        if not inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
            raise InvalidError(f"Async (non-generator) function returned value of type {type(res)}")
        value = await res
        await aio_function_context.enqueue_output(
            input_id,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=aio_function_context.serialize(value),
        )


def _wait_for_gpu_init():
    from cuda import cuda

    for i in range(3):
        try:
            cuda.cuInit(0)
            logger.info("CUDA device initialized successfully.")
            return
        except Exception:
            time.sleep(1)
    logger.info("Failed to initialize CUDA device.")


def import_function(function_def: api_pb2.Function) -> Callable:
    # This is not in function_context, so that any global scope code that runs during import
    # runs on the main thread.
    imported_function = _path_to_function(function_def.module_name, function_def.function_name)
    if isinstance(imported_function, (FunctionHandle, AioFunctionHandle)):
        # We want the internal type of this, not the external
        _function_proxy = synchronizer._translate_in(imported_function)
        function = _function_proxy.get_raw_f()
    else:
        function = imported_function

    if function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP:
        # function returns an asgi_app, that we can use as a callable.
        asgi_app = function()
        return asgi_app_wrapper(asgi_app)
    elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP:
        # function returns an wsgi_app, that we can use as a callable.
        wsgi_app = function()
        return wsgi_app_wrapper(wsgi_app)
    elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
        # function is webhook without an ASGI app. Create one for it.
        return fastAPI_function_wrapper(function, function_def.webhook_config.method)
    else:
        return function


def main(container_args: api_pb2.ContainerArguments, client: Client):
    # TODO: if there's an exception in this scope (in particular when we import code dynamically),
    # we could catch that exception and set it properly serialized to the client. Right now the
    # whole container fails with a non-zero exit code and we send back a more opaque error message.
    function_type = container_args.function_def.function_type

    if container_args.function_def.resources.gpu:
        _wait_for_gpu_init()

    # This is a bit weird but we need both the blocking and async versions of FunctionContext.
    # At some point, we should fix that by having built-in support for running "user code"
    _function_context = _FunctionContext(container_args, client)
    function_context, aio_function_context = synchronize_apis(_function_context)

    with function_context.handle_general_exception():
        function_context.initialize_app()

        if container_args.function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
            function = function_context.get_serialized_function()
        else:
            function = import_function(container_args.function_def)

        for input_id, input_pb in function_context.run_inputs_outputs():  # type: ignore
            with function_context.handle_input_exception(input_id):
                # Note: this blocks the call_function as well. In the future we might want to stream outputs
                # back asynchronously, but then block the call_function if there is back-pressure.
                if not is_async(function):
                    call_function_sync(function_context, function, function_type, input_id, input_pb)  # type: ignore
                else:
                    asyncio.run(call_function_async(aio_function_context, function, function_type, input_id, input_pb))  # type: ignore


if __name__ == "__main__":
    logger.debug("Container: starting")

    container_args = api_pb2.ContainerArguments()
    container_args.ParseFromString(base64.b64decode(sys.argv[1]))

    # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
    # This is good because if the function is long running then we the client can still send heartbeats
    # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
    client = Client.from_env()
    with proxy_tunnel(container_args.proxy_info):
        main(container_args, client)

    logger.debug("Container: done")
