# Copyright Modal Labs 2022
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import importlib
import inspect
import json
import math
import pickle
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Callable, List, Optional, Type

from grpclib import Status

from modal.stub import _Stub
from modal_proto import api_pb2
from modal_utils.async_utils import (
    TaskContext,
    synchronize_api,
    synchronizer,
)
from modal_utils.grpc_utils import retry_transient_errors

from ._asgi import asgi_app_wrapper, webhook_asgi_app, wsgi_app_wrapper
from ._blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._function_utils import LocalFunctionError, is_async as get_is_async, is_global_function
from ._proxy_tunnel import proxy_tunnel
from ._pty import run_in_pty
from ._serialization import deserialize, deserialize_data_format, serialize, serialize_data_format
from ._traceback import extract_traceback
from ._tracing import extract_tracing_context, set_span_tag, trace, wrap
from .app import _container_app, _ContainerApp
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, Client, _Client
from .cls import Cls
from .config import config, logger
from .exception import InvalidError
from .functions import Function, _Function, _set_current_input_id  # type: ignore

if TYPE_CHECKING:
    from types import ModuleType

MAX_OUTPUT_BATCH_SIZE: int = 49

RTT_S: float = 0.5  # conservative estimate of RTT in seconds.


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

    def __init__(self, container_args: api_pb2.ContainerArguments, client: _Client):
        self.task_id = container_args.task_id
        self.function_id = container_args.function_id
        self.app_id = container_args.app_id
        self.function_def = container_args.function_def
        self.calls_completed = 0
        self.total_user_time: float = 0.0
        self.current_input_id: Optional[str] = None
        self.current_input_started_at: Optional[float] = None

        self._stub_name = self.function_def.stub_name
        self._input_concurrency: Optional[int] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._environment_name = container_args.environment_name
        self._waiting_for_checkpoint = False

        self._client = client
        assert isinstance(self._client, _Client)

    @wrap()
    async def initialize_app(self) -> _ContainerApp:
        await _container_app.init(self._client, self.app_id, self._stub_name, self._environment_name)
        return _container_app

    async def _heartbeat(self):
        # Don't send heartbeats for tasks waiting to be checkpointed.
        # Calling gRPC methods open new connections which block the
        # checkpointing process.
        if self._waiting_for_checkpoint:
            return

        request = api_pb2.ContainerHeartbeatRequest()
        if self.current_input_id is not None:
            request.current_input_id = self.current_input_id
        if self.current_input_started_at is not None:
            request.current_input_started_at = self.current_input_started_at

        # TODO(erikbern): capture exceptions?
        await retry_transient_errors(self._client.stub.ContainerHeartbeat, request, attempt_timeout=HEARTBEAT_TIMEOUT)

    @contextlib.asynccontextmanager
    async def heartbeats(self):
        async with TaskContext(grace=1.0) as tc:
            tc.infinite_loop(self._heartbeat, sleep=HEARTBEAT_INTERVAL)
            yield

    async def get_serialized_function(self) -> tuple[Optional[Any], Callable]:
        # Fetch the serialized function definition
        request = api_pb2.FunctionGetSerializedRequest(function_id=self.function_id)
        response = await self._client.stub.FunctionGetSerialized(request)
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

    def serialize_data_format(self, obj: Any, data_format: int) -> bytes:
        return serialize_data_format(obj, data_format)

    def deserialize_data_format(self, data: bytes, data_format: int) -> Any:
        return deserialize_data_format(data, data_format, self._client)

    @wrap()
    async def populate_input_blobs(self, item: api_pb2.FunctionInput):
        args = await blob_download(item.args_blob_id, self._client.stub)

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

    async def _generate_inputs(self) -> AsyncIterator[tuple[str, api_pb2.FunctionInput]]:
        request = api_pb2.FunctionGetInputsRequest(function_id=self.function_id)
        eof_received = False
        iteration = 0
        while not eof_received:
            request.average_call_time = self.get_average_call_time()
            request.max_values = self.get_max_inputs_to_fetch()  # Deprecated; remove.
            request.input_concurrency = self._input_concurrency

            await self._semaphore.acquire()
            try:
                # If number of active inputs is at max queue size, this will block.
                yielded = False
                with trace("get_inputs"):
                    set_span_tag("iteration", str(iteration))  # force this to be a tag string
                    iteration += 1
                    response: api_pb2.FunctionGetInputsResponse = await retry_transient_errors(
                        self._client.stub.FunctionGetInputs, request
                    )

                if response.rate_limit_sleep_duration:
                    logger.info(
                        "Task exceeded rate limit, sleeping for %.2fs before trying again."
                        % response.rate_limit_sleep_duration
                    )
                    await asyncio.sleep(response.rate_limit_sleep_duration)
                elif response.inputs:
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

                        # If yielded, allow semaphore to be released via push_outputs
                        yield (item.input_id, input_pb)
                        yielded = True

                        if item.input.final_input:
                            eof_received = True
                            break
            finally:
                if not yielded:
                    self._semaphore.release()

    async def run_inputs_outputs(self, input_concurrency: int = 1):
        # Ensure we do not fetch new inputs when container is too busy.
        # Before trying to fetch an input, acquire the semaphore:
        # - if no input is fetched, release the semaphore.
        # - or, when the output for the fetched input is sent, release the semaphore.
        self._input_concurrency = input_concurrency
        self._semaphore = asyncio.Semaphore(input_concurrency)

        try:
            async for input_id, input_pb in self._generate_inputs():
                args, kwargs = self.deserialize(input_pb.args) if input_pb.args else ((), {})
                _set_current_input_id(input_id)
                self.current_input_id, self.current_input_started_at = (input_id, time.time())
                yield input_id, args, kwargs
                _set_current_input_id(None)
                self.current_input_id, self.current_input_started_at = (None, None)
        finally:
            # collect all active input slots, meaning all inputs have wrapped up.
            for _ in range(input_concurrency):
                await self._semaphore.acquire()

    async def _push_output(
        self, input_id, started_at: float, gen_index: int, data_format=api_pb2.DATA_FORMAT_UNSPECIFIED, **kwargs
    ):
        # upload data to S3 if too big.
        if "data" in kwargs and kwargs["data"] and len(kwargs["data"]) > MAX_OBJECT_SIZE_BYTES:
            data_blob_id = await blob_upload(kwargs["data"], self._client.stub)
            # mutating kwargs.
            del kwargs["data"]
            kwargs["data_blob_id"] = data_blob_id

        output = api_pb2.FunctionPutOutputsItem(
            input_id=input_id,
            input_started_at=started_at,
            output_created_at=time.time(),
            gen_index=gen_index,
            result=api_pb2.GenericResult(**kwargs),
            data_format=data_format,
        )

        await retry_transient_errors(
            self._client.stub.FunctionPutOutputs,
            api_pb2.FunctionPutOutputsRequest(outputs=[output]),
            additional_status_codes=[Status.RESOURCE_EXHAUSTED],
            max_retries=None,  # Retry indefinitely, trying every 1s.
        )

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
            await retry_transient_errors(self._client.stub.TaskResult, req)

            # Shut down the task gracefully
            raise UserException()

    @contextlib.asynccontextmanager
    async def handle_input_exception(
        self, input_id, started_at: float, output_index: SequenceNumber
    ) -> AsyncGenerator[None, None]:
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
            await self._push_output(
                input_id,
                started_at=started_at,
                gen_index=output_index.value,
                data_format=api_pb2.DATA_FORMAT_PICKLE,
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                data=self.serialize_exception(exc),
                exception=repr(exc),
                traceback=traceback.format_exc(),
                serialized_tb=serialized_tb,
                tb_line_cache=tb_line_cache,
            )
            await self.complete_call(started_at)

    async def complete_call(self, started_at):
        self.total_user_time += time.time() - started_at
        self.calls_completed += 1
        self._semaphore.release()

    async def push_output(self, input_id, started_at: float, output_index: int, data: Any, data_format: int) -> None:
        await self._push_output(
            input_id,
            started_at=started_at,
            gen_index=output_index,
            data_format=data_format,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=self.serialize_data_format(data, data_format),
        )
        await self.complete_call(started_at)

    async def push_generator_value(
        self, input_id, started_at: float, output_index: int, data: Any, data_format: int
    ) -> None:
        await self._push_output(
            input_id,
            started_at=started_at,
            gen_index=output_index,
            data_format=data_format,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=self.serialize_data_format(data, data_format),
            gen_status=api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE,
        )

    async def push_generator_eof(self, input_id, started_at: float, output_index: int) -> None:
        await self._push_output(
            input_id,
            started_at=started_at,
            gen_index=output_index,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            gen_status=api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE,
        )
        await self.complete_call(started_at)

    async def checkpoint(self) -> None:
        """Message server indicating that function is ready to be checkpointed."""
        await self._client.stub.ContainerCheckpoint(api_pb2.ContainerCheckpointRequest())

        self._waiting_for_checkpoint = True
        await self._client._close()

        logger.debug("checkpointing request sent and connection closed")

        # Busy-wait for restore. `/opt/modal/restore-state.json` is created
        # by the worker process with updates to the container config.
        restored_path = Path(config.get("restore_state_path"))
        while not restored_path.exists():
            logger.debug("waiting for restore ...")
            await asyncio.sleep(0.01)
            continue

        # Look for state file and create new client with updated credentials.
        with restored_path.open("r") as file:
            restored_state = json.load(file)

        # State data is serialized with key-value pairs, example: {"task_id": "tk-000"}
        for key, value in restored_state.items():
            config.override_locally(key, value)

        self._client = await _Client.from_env()
        self._waiting_for_checkpoint = False


FunctionIOManager = synchronize_api(_FunctionIOManager)


def call_function_sync(
    function_io_manager,  #: FunctionIOManager,  # TODO: this type is generated in runtime
    imp_fun: ImportedFunction,
):
    # If this function is on a class, instantiate it and enter it
    if imp_fun.obj is not None and not imp_fun.is_auto_snapshot:
        if hasattr(imp_fun.obj, "__enter__"):
            # Call a user-defined method
            with function_io_manager.handle_user_exception():
                imp_fun.obj.__enter__()
        elif hasattr(imp_fun.obj, "__aenter__"):
            logger.warning("Not running asynchronous enter/exit handlers with a sync function")

    try:

        def run_inputs(input_id, args, kwargs):
            output_index = SequenceNumber(0)
            started_at = time.time()
            with function_io_manager.handle_input_exception(input_id, started_at, output_index):
                # TODO(gongy): run this in an executor
                res = imp_fun.fun(*args, **kwargs)

                # TODO(erikbern): any exception below shouldn't be considered a user exception
                if imp_fun.is_generator:
                    if not inspect.isgenerator(res):
                        raise InvalidError(f"Generator function returned value of type {type(res)}")

                    for value in res:
                        function_io_manager.push_generator_value(
                            input_id, started_at, output_index.value, value, imp_fun.data_format
                        )
                        output_index.increase()

                    function_io_manager.push_generator_eof(input_id, started_at, output_index.value)
                else:
                    if inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                        raise InvalidError(
                            f"Sync (non-generator) function return value of type {type(res)}."
                            " You might need to use @stub.function(..., is_generator=True)."
                        )
                    function_io_manager.push_output(input_id, started_at, output_index.value, res, imp_fun.data_format)

        if imp_fun.input_concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for input_id, args, kwargs in function_io_manager.run_inputs_outputs(imp_fun.input_concurrency):
                    executor.submit(run_inputs, input_id, args, kwargs)
        else:
            for input_id, args, kwargs in function_io_manager.run_inputs_outputs(imp_fun.input_concurrency):
                run_inputs(input_id, args, kwargs)
    finally:
        if imp_fun.obj is not None and hasattr(imp_fun.obj, "__exit__"):
            with function_io_manager.handle_user_exception():
                imp_fun.obj.__exit__(*sys.exc_info())


@wrap()
async def call_function_async(
    function_io_manager,  #: FunctionIOManager,  # TODO: this one too
    imp_fun: ImportedFunction,
):
    # If this function is on a class, instantiate it and enter it
    if imp_fun.obj is not None and not imp_fun.is_auto_snapshot:
        if hasattr(imp_fun.obj, "__aenter__"):
            # Call a user-defined method
            async with function_io_manager.handle_user_exception.aio():
                await imp_fun.obj.__aenter__()
        elif hasattr(imp_fun.obj, "__enter__"):
            async with function_io_manager.handle_user_exception.aio():
                imp_fun.obj.__enter__()

    try:

        async def run_input(input_id, args, kwargs):
            output_index = SequenceNumber(0)  # mutable number we can increase from the generator loop
            started_at = time.time()
            async with function_io_manager.handle_input_exception.aio(input_id, started_at, output_index):
                res = imp_fun.fun(*args, **kwargs)

                # TODO(erikbern): any exception below shouldn't be considered a user exception
                if imp_fun.is_generator:
                    if not inspect.isasyncgen(res):
                        raise InvalidError(f"Async generator function returned value of type {type(res)}")
                    async for value in res:
                        await function_io_manager.push_generator_value.aio(
                            input_id, started_at, output_index.value, value, imp_fun.data_format
                        )
                        output_index.increase()
                    await function_io_manager.push_generator_eof.aio(input_id, started_at, output_index.value)
                else:
                    if not inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                        raise InvalidError(
                            f"Async (non-generator) function returned value of type {type(res)}"
                            " You might need to use @stub.function(..., is_generator=True)."
                        )
                    value = await res
                    await function_io_manager.push_output.aio(
                        input_id, started_at, output_index.value, value, imp_fun.data_format
                    )

        if imp_fun.input_concurrency > 1:
            async with TaskContext() as execution_context:
                async for input_id, args, kwargs in function_io_manager.run_inputs_outputs.aio(
                    imp_fun.input_concurrency
                ):
                    execution_context.create_task(run_input(input_id, args, kwargs))
        else:
            async for input_id, args, kwargs in function_io_manager.run_inputs_outputs.aio(imp_fun.input_concurrency):
                await run_input(input_id, args, kwargs)
    finally:
        if imp_fun.obj is not None:
            if hasattr(imp_fun.obj, "__aexit__"):
                async with function_io_manager.handle_user_exception.aio():
                    await imp_fun.obj.__aexit__(*sys.exc_info())
            elif hasattr(imp_fun.obj, "__exit__"):
                async with function_io_manager.handle_user_exception.aio():
                    imp_fun.obj.__exit__(*sys.exc_info())


@dataclass
class ImportedFunction:
    obj: Any
    fun: Callable
    stub: Optional[_Stub]
    is_async: bool
    is_generator: bool
    data_format: int  # api_pb2.DataFormat
    input_concurrency: int
    is_auto_snapshot: bool
    function: _Function


@wrap()
def import_function(function_def: api_pb2.Function, ser_cls, ser_fun, ser_params: Optional[bytes]) -> ImportedFunction:
    # This is not in function_io_manager, so that any global scope code that runs during import
    # runs on the main thread.
    module: Optional[ModuleType] = None
    cls: Optional[Type] = None
    fun: Callable
    if ser_fun is not None:
        # This is a serialized function we already fetched from the server
        cls, fun = ser_cls, ser_fun
    else:
        # Load the module dynamically
        module = importlib.import_module(function_def.module_name)
        qual_name: str = function_def.function_name

        if not is_global_function(qual_name):
            raise LocalFunctionError("Attempted to load a function defined in a function scope")

        parts = qual_name.split(".")
        if len(parts) == 1:
            cls = None
            fun = getattr(module, qual_name)
        elif len(parts) == 2:
            cls_name, fun_name = parts
            cls = getattr(module, cls_name)
            fun = getattr(cls, fun_name)
        else:
            raise InvalidError(f"Invalid function qualname {qual_name}")

    # The decorator is typically in global scope, but may have been applied independently
    active_stub: Optional[_Stub] = None
    function: Optional[_Function] = None
    if isinstance(fun, Function):
        function = synchronizer._translate_in(fun)
        fun = function.get_raw_f()
        active_stub = function._stub
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

    # What data format is used for function inputs and outputs
    data_format = api_pb2.DATA_FORMAT_PICKLE

    # Container can fetch multiple inputs simultaneously
    input_concurrency = function_def.allow_concurrent_inputs or 1

    # Instantiate the class if it's defined
    if cls:
        if ser_params:
            args, kwargs = pickle.loads(ser_params)
        else:
            args, kwargs = (), {}
        obj = cls(*args, **kwargs)
        if isinstance(cls, Cls):
            obj = obj.get_obj()
        # Bind the function to the instance (using the descriptor protocol!)
        fun = fun.__get__(obj)
    else:
        obj = None

    if function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP:
        # function returns an asgi_app, that we can use as a callable.
        asgi_app = fun()
        fun = asgi_app_wrapper(asgi_app)
        is_async = True
        is_generator = True
        data_format = api_pb2.DATA_FORMAT_ASGI
    elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP:
        # function returns an wsgi_app, that we can use as a callable.
        wsgi_app = fun()
        fun = wsgi_app_wrapper(wsgi_app)
        is_async = True
        is_generator = True
        data_format = api_pb2.DATA_FORMAT_ASGI
    elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
        # function is webhook without an ASGI app. Create one for it.
        asgi_app = webhook_asgi_app(fun, function_def.webhook_config.method)
        fun = asgi_app_wrapper(asgi_app)
        is_async = True
        is_generator = True
        data_format = api_pb2.DATA_FORMAT_ASGI

    return ImportedFunction(
        obj,
        fun,
        active_stub,
        is_async,
        is_generator,
        data_format,
        input_concurrency,
        function_def.is_auto_snapshot,
        function,
    )


def main(container_args: api_pb2.ContainerArguments, client: Client):
    # TODO: if there's an exception in this scope (in particular when we import code dynamically),
    # we could catch that exception and set it properly serialized to the client. Right now the
    # whole container fails with a non-zero exit code and we send back a more opaque error message.

    # This is a bit weird but we need both the blocking and async versions of FunctionIOManager.
    # At some point, we should fix that by having built-in support for running "user code"
    function_io_manager = FunctionIOManager(container_args, client)

    # Define a global app (need to do this before imports)
    container_app = function_io_manager.initialize_app()

    with function_io_manager.heartbeats():
        # If this is a serialized function, fetch the definition from the server
        if container_args.function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
            ser_cls, ser_fun = function_io_manager.get_serialized_function()
        else:
            ser_cls, ser_fun = None, None

        # Initialize the function
        # NOTE: detecting the stub causes all objects to be associated with the app and hydrated
        with function_io_manager.handle_user_exception():
            imp_fun = import_function(container_args.function_def, ser_cls, ser_fun, container_args.serialized_params)

        # Hydrate all function dependencies
        if imp_fun.function:
            dep_object_ids: List[str] = [dep.object_id for dep in container_args.function_def.object_dependencies]
            container_app.hydrate_function_deps(imp_fun.function, dep_object_ids)

        # Checkpoint container after imports. Checkpointed containers start from this point
        # onwards. This assumes that everything up to this point has run successfully,
        # including global imports.
        if container_args.function_def.is_checkpointing_function:
            function_io_manager.checkpoint()

        pty_info: api_pb2.PTYInfo = container_args.function_def.pty_info
        if pty_info.pty_type or pty_info.enabled:
            # TODO(erikbern): the second condition is for legacy compatibility, remove soon
            # TODO(erikbern): there is no client test for this branch
            input_stream = container_app._get_pty()
            imp_fun.fun = run_in_pty(imp_fun.fun, input_stream, pty_info)

        if not imp_fun.is_async:
            call_function_sync(function_io_manager, imp_fun)
        else:
            run_with_signal_handler(call_function_async(function_io_manager, imp_fun))


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
