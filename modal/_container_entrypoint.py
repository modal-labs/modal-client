# Copyright Modal Labs 2022
from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import inspect
import math
import pickle
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Optional

from grpclib import Status
from synchronicity.interface import Interface

from modal.stub import _Stub
from modal_proto import api_pb2
from modal_utils.async_utils import (
    TaskContext,
    queue_batch_iterator,
    synchronize_api,
    synchronizer,
)
from modal_utils.grpc_utils import retry_transient_errors

from ._asgi import asgi_app_wrapper, webhook_asgi_app, wsgi_app_wrapper
from ._blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._function_utils import load_function_from_module
from ._proxy_tunnel import proxy_tunnel
from ._pty import run_in_pty
from ._serialization import deserialize, serialize
from ._traceback import extract_traceback
from ._tracing import extract_tracing_context, set_span_tag, trace, wrap
from .app import _App
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, Client, _Client
from .config import logger
from .exception import InvalidError
from .functions import FunctionHandle, _set_current_input_id  # type: ignore

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


def get_is_async(function):
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


def run_with_signal_handler(coro):
    """Execute coro in an event loop, with a signal handler that cancels
    the task in the case of SIGINT or SIGTERM. Prevents stray cancellation errors
    from propagating up."""

    loop = asyncio.new_event_loop()
    task = asyncio.ensure_future(coro, loop=loop)
    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(s, task.cancel)
    try:
        result = loop.run_until_complete(task)
    finally:
        loop.close()
    return result


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
        self.current_input_id: Optional[str] = None
        self.current_input_started_at: Optional[float] = None
        self._client = synchronizer._translate_in(self.client)  # make it a _Client object
        self._stub_name = self.function_def.stub_name
        assert isinstance(self._client, _Client)

    @wrap()
    async def initialize_app(self):
        return await _App.init_container(self._client, self.app_id, self._stub_name)

    async def _heartbeat(self):
        request = api_pb2.ContainerHeartbeatRequest()
        if self.current_input_id is not None:
            request.current_input_id = self.current_input_id
        if self.current_input_started_at is not None:
            request.current_input_started_at = self.current_input_started_at

        # TODO(erikbern): capture exceptions?
        await retry_transient_errors(self.client.stub.ContainerHeartbeat, request, attempt_timeout=HEARTBEAT_TIMEOUT)

    @contextlib.asynccontextmanager
    async def heartbeats(self):
        async with TaskContext(grace=1) as tc:
            tc.infinite_loop(self._heartbeat, sleep=HEARTBEAT_INTERVAL)
            yield

    async def get_serialized_function(self) -> tuple[Optional[Any], Callable]:
        # Fetch the serialized function definition
        request = api_pb2.FunctionGetSerializedRequest(function_id=self.function_id)
        response = await self.client.stub.FunctionGetSerialized(request)
        fun = self.deserialize(response.function_serialized)

        if response.class_serialized:
            cls = self.deserialize(response.class_serialized)
        else:
            cls = None

        return cls, fun

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

    def get_average_call_time(self) -> float:
        if self.calls_completed == 0:
            return 0

        return self.total_user_time / self.calls_completed

    def get_max_inputs_to_fetch(self):
        if self.calls_completed == 0:
            return 1

        return math.ceil(RTT_S / max(self.get_average_call_time(), 1e-6))

    async def _generate_inputs(
        self,
    ) -> AsyncIterator[tuple[str, api_pb2.FunctionInput]]:
        request = api_pb2.FunctionGetInputsRequest(function_id=self.function_id)
        eof_received = False
        iteration = 0
        while not eof_received:
            request.average_call_time = self.get_average_call_time()
            request.max_values = self.get_max_inputs_to_fetch()  # Deprecated; remove.

            with trace("get_inputs"):
                set_span_tag("iteration", str(iteration))  # force this to be a tag string
                iteration += 1
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
                    args, kwargs = self.deserialize(input_pb.args) if input_pb.args else ((), {})
                    _set_current_input_id(input_id)
                    self.current_input_id, self.current_input_started_at = (input_id, time.time())
                    yield input_id, args, kwargs
                    _set_current_input_id(None)
                    self.total_user_time += time.time() - self.current_input_started_at
                    self.current_input_id, self.current_input_started_at = (None, None)
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
            input_started_at=self.current_input_started_at,
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

    def serialize_traceback(self, exc: BaseException) -> tuple[Optional[bytes], Optional[bytes]]:
        serialized_tb, tb_line_cache = None, None

        try:
            tb_dict, line_cache = extract_traceback(exc, self.task_id)
            serialized_tb = self.serialize(tb_dict)
            tb_line_cache = self.serialize(line_cache)
        except Exception:
            logger.info("Failed to serialize exception traceback.")

        return serialized_tb, tb_line_cache

    @contextlib.asynccontextmanager
    async def handle_user_exception(self) -> AsyncGenerator[None, None]:
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

            req = api_pb2.TaskResultRequest(result=result)
            await retry_transient_errors(self.client.stub.TaskResult, req)

            # Shut down the task gracefully
            raise UserException()

    @contextlib.asynccontextmanager
    async def handle_input_exception(self, input_id, output_index: SequenceNumber) -> AsyncGenerator[None, None]:
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
FunctionIOManager = synchronize_api(_FunctionIOManager)


def call_function_sync(
    function_io_manager,  #: FunctionIOManager,  # TODO: this type is generated in runtime
    obj: Optional[Any],
    fun: Callable,
    is_generator: bool,
):
    # If this function is on a class, instantiate it and enter it
    if obj is not None:
        if hasattr(obj, "__enter__"):
            # Call a user-defined method
            with function_io_manager.handle_user_exception():
                obj.__enter__()
        elif hasattr(obj, "__aenter__"):
            logger.warning("Not running asynchronous enter/exit handlers with a sync function")

    try:
        for input_id, args, kwargs in function_io_manager.run_inputs_outputs():
            output_index = SequenceNumber(0)
            with function_io_manager.handle_input_exception(input_id, output_index):
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
                        raise InvalidError(
                            f"Sync (non-generator) function return value of type {type(res)}."
                            " You might need to use @stub.function(..., is_generator=True)."
                        )
                    function_io_manager.enqueue_output(input_id, output_index.value, res)
    finally:
        if obj is not None and hasattr(obj, "__exit__"):
            with function_io_manager.handle_user_exception():
                obj.__exit__(*sys.exc_info())


@wrap()
async def call_function_async(
    function_io_manager,  #: FunctionIOManager,  # TODO: this one too
    obj: Optional[Any],
    fun: Callable,
    is_generator: bool,
):
    # If this function is on a class, instantiate it and enter it
    if obj is not None:
        if hasattr(obj, "__aenter__"):
            # Call a user-defined method
            async with function_io_manager.handle_user_exception.aio():
                await obj.__aenter__()
        elif hasattr(obj, "__enter__"):
            async with function_io_manager.handle_user_exception.aio():
                obj.__enter__()

    try:
        async for input_id, args, kwargs in function_io_manager.run_inputs_outputs.aio():
            output_index = SequenceNumber(0)  # mutable number we can increase from the generator loop
            async with function_io_manager.handle_input_exception.aio(input_id, output_index):
                res = fun(*args, **kwargs)

                # TODO(erikbern): any exception below shouldn't be considered a user exception
                if is_generator:
                    if not inspect.isasyncgen(res):
                        raise InvalidError(f"Async generator function returned value of type {type(res)}")
                    async for value in res:
                        await function_io_manager.enqueue_generator_value.aio(input_id, output_index.value, value)
                        output_index.increase()
                    await function_io_manager.enqueue_generator_eof.aio(input_id, output_index.value)
                else:
                    if not inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                        raise InvalidError(
                            f"Async (non-generator) function returned value of type {type(res)}"
                            " You might need to use @stub.function(..., is_generator=True)."
                        )
                    value = await res
                    await function_io_manager.enqueue_output.aio(input_id, output_index.value, value)
    finally:
        if obj is not None:
            if hasattr(obj, "__aexit__"):
                async with function_io_manager.handle_user_exception.aio():
                    await obj.__aexit__(*sys.exc_info())
            elif hasattr(obj, "__exit__"):
                async with function_io_manager.handle_user_exception.aio():
                    obj.__exit__(*sys.exc_info())


@dataclass
class ImportedFunction:
    obj: Any
    fun: Callable
    stub: Optional[_Stub]
    is_async: bool
    is_generator: bool


@wrap()
def import_function(function_def: api_pb2.Function, ser_cls, ser_fun, ser_params: Optional[bytes]) -> ImportedFunction:
    # This is not in function_io_manager, so that any global scope code that runs during import
    # runs on the main thread.
    module = None
    if ser_fun is not None:
        # This is a serialized function we already fetched from the server
        cls, fun = ser_cls, ser_fun
    else:
        # Load the module dynamically
        module = importlib.import_module(function_def.module_name)
        cls, fun = load_function_from_module(module, function_def.function_name)

    # The decorator is typically in global scope, but may have been applied independently
    active_stub = None
    if isinstance(fun, FunctionHandle):
        _function_proxy = synchronizer._translate_in(fun)
        fun = _function_proxy.get_raw_f()
        active_stub = _function_proxy._stub
    elif module is not None and not function_def.is_builder_function:
        # This branch is reached in the special case that the imported function is 1) not serialized, and 2) isn't a FunctionHandle - i.e, not decorated at definition time
        # Look at all instantiated stubs - if there is only one with the indicated name, use that one
        matching_stubs = _Stub._all_stubs.get(function_def.stub_name, [])
        if len(matching_stubs) > 1:
            logger.warning(
                "You have multiple stubs with the same name which may prevent you from calling into other functions or using stub.is_inside(). It's recommended to name all your Stubs uniquely."
            )
        elif len(matching_stubs) == 1:
            active_stub = matching_stubs[0]
        # there could also technically be zero found stubs, but that should probably never be an issue since that would mean user won't use is_inside or other function handles anyway

    # Check this property before we turn it into a method (overriden by webhooks)
    is_async = get_is_async(fun)

    # Use the function definition for whether this is a generator (overriden by webhooks)
    is_generator = function_def.function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR

    # Instantiate the class if it's defined
    if cls:
        if ser_params:
            args, kwargs = pickle.loads(ser_params)
            obj = cls(*args, **kwargs)
        else:
            obj = cls()
        # Bind the function to the instance (using the descriptor protocol!)
        fun = fun.__get__(obj, cls)
    else:
        obj = None

    if function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP:
        # function returns an asgi_app, that we can use as a callable.
        asgi_app = fun()
        fun = asgi_app_wrapper(asgi_app)
        is_async = True
        is_generator = True
    elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP:
        # function returns an wsgi_app, that we can use as a callable.
        wsgi_app = fun()
        fun = wsgi_app_wrapper(wsgi_app)
        is_async = True
        is_generator = True
    elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
        # function is webhook without an ASGI app. Create one for it.
        asgi_app = webhook_asgi_app(fun, function_def.webhook_config.method)
        fun = asgi_app_wrapper(asgi_app)
        is_async = True
        is_generator = True

    return ImportedFunction(obj, fun, active_stub, is_async, is_generator)


def main(container_args: api_pb2.ContainerArguments, client: Client):
    # TODO: if there's an exception in this scope (in particular when we import code dynamically),
    # we could catch that exception and set it properly serialized to the client. Right now the
    # whole container fails with a non-zero exit code and we send back a more opaque error message.

    # This is a bit weird but we need both the blocking and async versions of FunctionIOManager.
    # At some point, we should fix that by having built-in support for running "user code"
    _function_io_manager = _FunctionIOManager(container_args, client)
    function_io_manager = synchronize_api(_function_io_manager)

    container_app = function_io_manager.initialize_app()

    with function_io_manager.heartbeats():
        # If this is a serialized function, fetch the definition from the server
        if container_args.function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
            ser_cls, ser_fun = function_io_manager.get_serialized_function()
        else:
            ser_cls, ser_fun = None, None

        # Initialize the function
        with function_io_manager.handle_user_exception():
            imp_fun = import_function(container_args.function_def, ser_cls, ser_fun, container_args.serialized_params)

        pty_info: api_pb2.PTYInfo = container_args.function_def.pty_info
        if pty_info.pty_type or pty_info.enabled:
            # TODO(erikbern): the second condition is for legacy compatibility, remove soon
            # TODO(erikbern): there is no client test for this branch
            input_stream_unwrapped = synchronizer._translate_in(container_app._pty_input_stream)
            input_stream_blocking = synchronizer._translate_out(input_stream_unwrapped, Interface.BLOCKING)
            imp_fun.fun = run_in_pty(imp_fun.fun, input_stream_blocking, pty_info)

        if not imp_fun.is_async:
            call_function_sync(function_io_manager, imp_fun.obj, imp_fun.fun, imp_fun.is_generator)
        else:
            run_with_signal_handler(
                call_function_async(function_io_manager, imp_fun.obj, imp_fun.fun, imp_fun.is_generator)
            )


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
