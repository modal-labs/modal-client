import asyncio
import base64
import importlib
import inspect
import math
import sys
import time
import traceback
from typing import Any, AsyncIterator, Callable, Optional, Tuple, Type

import cloudpickle
from grpclib import Status

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
from ._traceback import extract_traceback
from ._tracing import extract_tracing_context, set_span_tag, trace, wrap
from .app import _App
from .client import Client, _Client
from .config import logger
from .exception import InvalidError
from .functions import AioFunctionHandle, FunctionHandle, _set_current_input_id

MAX_OUTPUT_BATCH_SIZE = 100

RTT_S = 0.5  # conservative estimate of RTT in seconds.


class UserException(Exception):
    # Used to shut down the task gracefully
    pass


class SequenceNumber:
    def __init__(self, initial_value: int):
        self._value: int = initial_value

    def increase(self):
        self._value += 1

    @property
    def value(self) -> int:
        return self._value


class _FunctionIOManager:
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
        self.input_started_at: float = 0
        self._client = synchronizer._translate_in(self.client)  # make it a _Client object
        assert isinstance(self._client, _Client)

    @wrap()
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

    @wrap()
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
        while not eof_received:
            request.max_values = self.get_max_inputs_to_fetch()

            with trace("get_inputs"):
                response = await retry_transient_errors(self.client.stub.FunctionGetInputs, request)

            if response.rate_limit_sleep_duration:
                logger.info(
                    "Task exceeded rate limit, sleeping for %.2fs before trying again."
                    % response.rate_limit_sleep_duration
                )
                await asyncio.sleep(response.rate_limit_sleep_duration)
                continue

            if not response.inputs:
                continue

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
            await retry_transient_errors(
                self.client.stub.FunctionPutOutputs,
                req,
                attempt_timeout=3.0,
                total_timeout=20.0,
                additional_status_codes=[Status.RESOURCE_EXHAUSTED],
            )
            # TODO(erikbern): we'll get a RESOURCE_EXCHAUSTED if the buffer is full server-side.
            # It's possible we want to retry "harder" for this particular error.

    async def run_inputs_outputs(self):
        # This also makes sure to terminate the outputs
        self.output_queue: asyncio.Queue = asyncio.Queue()

        async with TaskContext(grace=10) as tc:
            tc.create_task(self._send_outputs())
            try:
                async for input_id, input_pb in self._generate_inputs():
                    _set_current_input_id(input_id)
                    args, kwargs = self.deserialize(input_pb.args) if input_pb.args else ((), {})
                    self.input_started_at = time.time()
                    yield input_id, args, kwargs
                    self.total_user_time += time.time() - self.input_started_at
                    self.calls_completed += 1
            finally:
                await self.output_queue.put(None)

    async def _enqueue_output(self, input_id, gen_index, **kwargs):
        # upload data to S3 if too big.
        if "data" in kwargs and kwargs["data"] and len(kwargs["data"]) > MAX_OBJECT_SIZE_BYTES:
            data_blob_id = await blob_upload(kwargs["data"], self.client.stub)
            # mutating kwargs.
            kwargs.pop("data")
            kwargs["data_blob_id"] = data_blob_id

        output = api_pb2.FunctionPutOutputsItem(
            input_id=input_id,
            input_started_at=self.input_started_at,
            output_created_at=time.time(),
            gen_index=gen_index,
            result=api_pb2.GenericResult(**kwargs),
        )
        await self.output_queue.put(output)

    def serialize_exception(self, exc: BaseException) -> Optional[bytes]:
        try:
            return self.serialize(exc)
        except Exception as serialization_exc:
            logger.info(f"Failed to serialize exception {exc}: {serialization_exc}")
            # We can't always serialize exceptions.
            return None

    def serialize_traceback(self, exc: BaseException) -> Tuple[Optional[bytes], Optional[bytes]]:
        serialized_tb, tb_line_cache = None, None

        try:
            tb_dict, line_cache = extract_traceback(exc, self.task_id)
            serialized_tb = self.serialize(tb_dict)
            tb_line_cache = self.serialize(line_cache)
        except Exception:
            logger.info("Failed to serialize exception traceback.")

        return serialized_tb, tb_line_cache

    @synchronizer.asynccontextmanager
    async def handle_user_exception(self):
        """Sets the task as failed in a way where it's not retried

        Only used for importing user code atm
        """
        try:
            yield
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            # Since this is on a different thread, sys.exc_info() can't find the exception in the stack.
            traceback.print_exception(type(exc), exc, exc.__traceback__)

            serialized_tb, tb_line_cache = self.serialize_traceback(exc)

            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                data=self.serialize_exception(exc),
                exception=repr(exc),
                traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                serialized_tb=serialized_tb,
                tb_line_cache=tb_line_cache,
            )

            req = api_pb2.TaskResultRequest(task_id=self.task_id, result=result)
            await retry_transient_errors(self.client.stub.TaskResult, req)

            # Shut down the task gracefully
            raise UserException()

    @synchronizer.asynccontextmanager
    async def handle_input_exception(self, input_id, output_index: SequenceNumber):
        try:
            with trace("input"):
                set_span_tag("input_id", input_id)
                yield
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            # print exception so it's logged
            traceback.print_exc()
            serialized_tb, tb_line_cache = self.serialize_traceback(exc)

            # Note: we're not serializing the traceback since it contains
            # local references that means we can't unpickle it. We *are*
            # serializing the exception, which may have some issues (there
            # was an earlier note about it that it might not be possible
            # to unpickle it in some cases). Let's watch out for issues.
            await self._enqueue_output(
                input_id,
                output_index.value,
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                data=self.serialize_exception(exc),
                exception=repr(exc),
                traceback=traceback.format_exc(),
                serialized_tb=serialized_tb,
                tb_line_cache=tb_line_cache,
            )

    async def enqueue_output(self, input_id, output_index: int, data):
        await self._enqueue_output(
            input_id,
            gen_index=output_index,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=self.serialize(data),
        )

    async def enqueue_generator_value(self, input_id, output_index: int, data):
        await self._enqueue_output(
            input_id,
            gen_index=output_index,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=self.serialize(data),
            gen_status=api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE,
        )

    async def enqueue_generator_eof(self, input_id, output_index: int):
        await self._enqueue_output(
            input_id,
            gen_index=output_index,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            gen_status=api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE,
        )


# just to mark the class as synchronized, we don't care about the interfaces
FunctionIOManager, AioFunctionIOManager = synchronize_apis(_FunctionIOManager)


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
    function_io_manager,  #: FunctionIOManager,  # TODO: this type is generated in runtime
    cls: Optional[Type],
    fun: Callable,
    is_generator: bool,
):
    # If this function is on a class, instantiate it and enter it
    if cls is not None:
        self = cls()
        if hasattr(cls, "__enter__"):
            # Call a user-defined method
            with function_io_manager.handle_user_exception():
                cls.__enter__(self)
        elif hasattr(cls, "__aenter__"):
            logger.warning("Not running asynchronous enter/exit handlers with a sync function")
    else:
        self = None

    try:
        for input_id, args, kwargs in function_io_manager.run_inputs_outputs():
            output_index = SequenceNumber(0)
            with function_io_manager.handle_input_exception(input_id, output_index):
                if self is not None:
                    res = fun(self, *args, **kwargs)
                else:
                    res = fun(*args, **kwargs)

                # TODO(erikbern): any exception below shouldn't be considered a user exception
                if is_generator:
                    if not inspect.isgenerator(res):
                        raise InvalidError(f"Generator function returned value of type {type(res)}")

                    for value in res:
                        function_io_manager.enqueue_generator_value(input_id, output_index.value, value)
                        output_index.increase()

                    function_io_manager.enqueue_generator_eof(input_id, output_index.value)
                else:
                    if inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                        raise InvalidError(f"Sync (non-generator) function return value of type {type(res)}")
                    function_io_manager.enqueue_output(input_id, output_index.value, res)
    finally:
        if cls is not None and hasattr(cls, "__exit__"):
            with function_io_manager.handle_user_exception():
                cls.__exit__(self, *sys.exc_info())


@wrap()
async def call_function_async(
    aio_function_io_manager,  #: AioFunctionIOManager,  # TODO: this one too
    cls: Optional[Type],
    fun: Callable,
    is_generator: bool,
):
    # If this function is on a class, instantiate it and enter it
    if cls is not None:
        self = cls()
        if hasattr(cls, "__aenter__"):
            # Call a user-defined method
            async with aio_function_io_manager.handle_user_exception():
                await cls.__aenter__(self)
        elif hasattr(cls, "__enter__"):
            async with aio_function_io_manager.handle_user_exception():
                cls.__enter__(self)
    else:
        self = None

    try:
        async for input_id, args, kwargs in aio_function_io_manager.run_inputs_outputs():
            output_index = SequenceNumber(0)  # mutable number we can increase from the generator loop
            async with aio_function_io_manager.handle_input_exception(input_id, output_index):
                if self:
                    res = fun(self, *args, **kwargs)
                else:
                    res = fun(*args, **kwargs)

                # TODO(erikbern): any exception below shouldn't be considered a user exception
                if is_generator:
                    if not inspect.isasyncgen(res):
                        raise InvalidError(f"Async generator function returned value of type {type(res)}")
                    async for value in res:
                        await aio_function_io_manager.enqueue_generator_value(input_id, output_index.value, value)
                        output_index.increase()
                    await aio_function_io_manager.enqueue_generator_eof(input_id, output_index.value)
                else:
                    if not inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                        raise InvalidError(f"Async (non-generator) function returned value of type {type(res)}")
                    value = await res
                    await aio_function_io_manager.enqueue_output(input_id, output_index.value, value)
    finally:
        if cls is not None:
            if hasattr(cls, "__aexit__"):
                async with aio_function_io_manager.handle_user_exception():
                    await cls.__aexit__(self, *sys.exc_info())
            elif hasattr(cls, "__exit__"):
                async with aio_function_io_manager.handle_user_exception():
                    cls.__exit__(self, *sys.exc_info())


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


@wrap()
def import_function(function_def: api_pb2.Function) -> Tuple[Optional[Type], Callable]:
    # This is not in function_io_manager, so that any global scope code that runs during import
    # runs on the main thread.
    module = importlib.import_module(function_def.module_name)

    # The function might be defined inside a class scope (e.g mymodule.MyClass.f)
    objs: list[Any] = [module]
    for path in function_def.function_name.split("."):
        objs.append(getattr(objs[-1], path))

    # If this function is defined on a class, return that too
    cls: Optional[Type] = None
    fun: Callable = objs[-1]
    if len(objs) >= 3:
        cls = objs[-2]

    # The decorator is typically in global scope, but may have been applied independently
    if isinstance(fun, (FunctionHandle, AioFunctionHandle)):
        _function_proxy = synchronizer._translate_in(fun)
        fun = _function_proxy.get_raw_f()

    if function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP:
        # function returns an asgi_app, that we can use as a callable.
        asgi_app = fun()
        return cls, asgi_app_wrapper(asgi_app)
    elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP:
        # function returns an wsgi_app, that we can use as a callable.
        wsgi_app = fun()
        return cls, wsgi_app_wrapper(wsgi_app)
    elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
        # function is webhook without an ASGI app. Create one for it.
        return cls, fastAPI_function_wrapper(fun, function_def.webhook_config.method)
    else:
        return cls, fun


def main(container_args: api_pb2.ContainerArguments, client: Client):
    # TODO: if there's an exception in this scope (in particular when we import code dynamically),
    # we could catch that exception and set it properly serialized to the client. Right now the
    # whole container fails with a non-zero exit code and we send back a more opaque error message.
    function_type = container_args.function_def.function_type

    if container_args.function_def.resources.gpu:
        _wait_for_gpu_init()

    # This is a bit weird but we need both the blocking and async versions of FunctionIOManager.
    # At some point, we should fix that by having built-in support for running "user code"
    _function_io_manager = _FunctionIOManager(container_args, client)
    function_io_manager, aio_function_io_manager = synchronize_apis(_function_io_manager)

    function_io_manager.initialize_app()

    is_generator = function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR
    if container_args.function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
        cls = None
        fun = function_io_manager.get_serialized_function()
    else:
        with function_io_manager.handle_user_exception():
            cls, fun = import_function(container_args.function_def)

    if not is_async(fun):
        call_function_sync(function_io_manager, cls, fun, is_generator)
    else:
        asyncio.run(call_function_async(aio_function_io_manager, cls, fun, is_generator))


if __name__ == "__main__":
    logger.debug("Container: starting")

    container_args = api_pb2.ContainerArguments()
    container_args.ParseFromString(base64.b64decode(sys.argv[1]))

    extract_tracing_context(dict(container_args.tracing_context.items()))

    with trace("main"):
        # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
        # This is good because if the function is long running then we the client can still send heartbeats
        # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
        with trace("client_from_env"):
            client = Client.from_env()

        try:
            with proxy_tunnel(container_args.proxy_info):
                try:
                    main(container_args, client)
                except UserException:
                    logger.info("User exception caught, exiting")
        except KeyboardInterrupt:
            logger.debug("Container: interrupted")

    logger.debug("Container: done")
