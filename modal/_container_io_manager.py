# Copyright Modal Labs 2024
import asyncio
import importlib.metadata
import inspect
import json
import math
import os
import signal
import sys
import time
import traceback
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
)

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message
from grpclib import Status
from synchronicity.async_wrap import asynccontextmanager

import modal_proto.api_pb2
from modal_proto import api_pb2

from ._serialization import deserialize, serialize, serialize_data_format
from ._traceback import extract_traceback, print_exception
from ._utils.async_utils import TaskContext, asyncify, synchronize_api, synchronizer
from ._utils.blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._utils.function_utils import _stream_function_call_data
from ._utils.grpc_utils import get_proto_oneof, retry_transient_errors
from ._utils.package_utils import parse_major_minor_version
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, _Client
from .config import config, logger
from .exception import ClientClosed, InputCancellation, InvalidError, SerializationError
from .running_app import RunningApp

if TYPE_CHECKING:
    import modal._asgi

DYNAMIC_CONCURRENCY_INTERVAL_SECS = 3
DYNAMIC_CONCURRENCY_TIMEOUT_SECS = 10
MAX_OUTPUT_BATCH_SIZE: int = 49

RTT_S: float = 0.5  # conservative estimate of RTT in seconds.


class UserException(Exception):
    """Used to shut down the task gracefully."""


class Sentinel:
    """Used to get type-stubs to work with this object."""


@dataclass
class FinalizedFunction:
    callable: Callable[..., Any]
    is_async: bool
    is_generator: bool
    data_format: int  # api_pb2.DataFormat
    lifespan_manager: Optional["modal._asgi.LifespanManager"] = None


class IOContext:
    """Context object for managing input, function calls, and function executions
    in a batched or single input context.
    """

    input_ids: List[str]
    function_call_ids: List[str]
    finalized_function: FinalizedFunction

    _cancel_issued: bool = False
    _cancel_callback: Optional[Callable[[], None]] = None

    def __init__(
        self,
        input_ids: List[str],
        function_call_ids: List[str],
        finalized_function: FinalizedFunction,
        function_inputs: List[api_pb2.FunctionInput],
        is_batched: bool,
        client: _Client,
    ):
        self.input_ids = input_ids
        self.function_call_ids = function_call_ids
        self.finalized_function = finalized_function
        self._function_inputs = function_inputs
        self._is_batched = is_batched
        self._client = client

    @classmethod
    async def create(
        cls,
        client: _Client,
        finalized_functions: Dict[str, FinalizedFunction],
        inputs: List[Tuple[str, str, api_pb2.FunctionInput]],
        is_batched: bool,
    ) -> "IOContext":
        assert len(inputs) >= 1 if is_batched else len(inputs) == 1
        input_ids, function_call_ids, function_inputs = zip(*inputs)

        async def _populate_input_blobs(client: _Client, input: api_pb2.FunctionInput) -> api_pb2.FunctionInput:
            # If we got a pointer to a blob, download it from S3.
            if input.WhichOneof("args_oneof") == "args_blob_id":
                args = await blob_download(input.args_blob_id, client.stub)
                # Mutating
                input.ClearField("args_blob_id")
                input.args = args

            return input

        function_inputs = await asyncio.gather(*[_populate_input_blobs(client, input) for input in function_inputs])
        # check every input in batch executes the same function
        method_name = function_inputs[0].method_name
        assert all(method_name == input.method_name for input in function_inputs)
        finalized_function = finalized_functions[method_name]
        return cls(input_ids, function_call_ids, finalized_function, function_inputs, is_batched, client)

    def set_cancel_callback(self, cb: Callable[[], None]):
        self._cancel_callback = cb

    def cancel(self):
        # Ensure we only issue the cancellation once.
        if self._cancel_issued:
            return

        if self._cancel_callback:
            logger.warning(f"Received a cancellation signal while processing input {self.input_ids}")
            self._cancel_issued = True
            self._cancel_callback()
        else:
            # TODO (elias): This should not normally happen but there is a small chance of a race
            #  between creating a new task for an input and attaching the cancellation callback
            logger.warning("Unexpected: Could not cancel input")

    def _args_and_kwargs(self) -> Tuple[Tuple[Any, ...], Dict[str, List[Any]]]:
        # deserializing here instead of the constructor
        # to make sure we handle user exceptions properly
        # and don't retry
        deserialized_args = [
            deserialize(input.args, self._client) if input.args else ((), {}) for input in self._function_inputs
        ]
        if not self._is_batched:
            return deserialized_args[0]

        func_name = self.finalized_function.callable.__name__

        param_names = []
        for param in inspect.signature(self.finalized_function.callable).parameters.values():
            param_names.append(param.name)

        # aggregate args and kwargs of all inputs into a kwarg dict
        kwargs_by_inputs: List[Dict[str, Any]] = [{} for _ in range(len(self.input_ids))]

        for i, (args, kwargs) in enumerate(deserialized_args):
            # check that all batched inputs should have the same number of args and kwargs
            if (num_params := len(args) + len(kwargs)) != len(param_names):
                raise InvalidError(
                    f"Modal batched function {func_name} takes {len(param_names)} positional arguments, but one invocation in the batch has {num_params}."  # noqa
                )

            for j, arg in enumerate(args):
                kwargs_by_inputs[i][param_names[j]] = arg
            for k, v in kwargs.items():
                if k not in param_names:
                    raise InvalidError(
                        f"Modal batched function {func_name} got unexpected keyword argument {k} in one invocation in the batch."  # noqa
                    )
                if k in kwargs_by_inputs[i]:
                    raise InvalidError(
                        f"Modal batched function {func_name} got multiple values for argument {k} in one invocation in the batch."  # noqa
                    )
                kwargs_by_inputs[i][k] = v

        formatted_kwargs = {
            param_name: [kwargs[param_name] for kwargs in kwargs_by_inputs] for param_name in param_names
        }
        return (), formatted_kwargs

    def call_finalized_function(self) -> Any:
        logger.debug(f"Starting input {self.input_ids}")
        args, kwargs = self._args_and_kwargs()
        res = self.finalized_function.callable(*args, **kwargs)
        logger.debug(f"Finished input {self.input_ids}")
        return res

    def validate_output_data(self, data: Any) -> List[Any]:
        if not self._is_batched:
            return [data]

        function_name = self.finalized_function.callable.__name__
        if not isinstance(data, list):
            raise InvalidError(f"Output of batched function {function_name} must be a list.")
        if len(data) != len(self.input_ids):
            raise InvalidError(
                f"Output of batched function {function_name} must be a list of equal length as its inputs."
            )
        return data


class InputSlots:
    """A semaphore that allows dynamically adjusting the concurrency."""

    active: int
    value: int
    waiter: Optional[asyncio.Future]
    closed: bool

    def __init__(self, value: int) -> None:
        self.active = 0
        self.value = value
        self.waiter = None
        self.closed = False

    async def acquire(self) -> None:
        if self.active < self.value:
            self.active += 1
        elif self.waiter is None:
            self.waiter = asyncio.get_running_loop().create_future()
            await self.waiter
        else:
            raise RuntimeError("Concurrent waiters are not supported.")

    def _wake_waiter(self) -> None:
        if self.active < self.value and self.waiter is not None:
            if not self.waiter.cancelled():  # could have been cancelled during interpreter shutdown
                self.waiter.set_result(None)
            self.waiter = None
            self.active += 1

    def release(self) -> None:
        self.active -= 1
        self._wake_waiter()

    def set_value(self, value: int) -> None:
        if self.closed:
            return
        self.value = value
        self._wake_waiter()

    async def close(self) -> None:
        self.closed = True
        for _ in range(self.value):
            await self.acquire()


class _ContainerIOManager:
    """Synchronizes all RPC calls and network operations for a running container.

    TODO: maybe we shouldn't synchronize the whole class.
    Then we could potentially move a bunch of the global functions onto it.
    """

    task_id: str
    function_id: str
    app_id: str
    function_def: api_pb2.Function
    checkpoint_id: Optional[str]

    calls_completed: int
    total_user_time: float
    current_input_id: Optional[str]
    current_inputs: Dict[str, IOContext]  # input_id -> IOContext
    current_input_started_at: Optional[float]

    _target_concurrency: int
    _max_concurrency: int
    _concurrency_loop: Optional[asyncio.Task]
    _input_slots: InputSlots

    _environment_name: str
    _heartbeat_loop: Optional[asyncio.Task]
    _heartbeat_condition: Optional[asyncio.Condition]
    _waiting_for_memory_snapshot: bool

    _is_interactivity_enabled: bool
    _fetching_inputs: bool

    _client: _Client

    _GENERATOR_STOP_SENTINEL: ClassVar[Sentinel] = Sentinel()
    _singleton: ClassVar[Optional["_ContainerIOManager"]] = None

    def _init(self, container_args: api_pb2.ContainerArguments, client: _Client):
        self.task_id = container_args.task_id
        self.function_id = container_args.function_id
        self.app_id = container_args.app_id
        self.function_def = container_args.function_def
        self.checkpoint_id = container_args.checkpoint_id or None

        self.calls_completed = 0
        self.total_user_time = 0.0
        self.current_input_id = None
        self.current_inputs = {}
        self.current_input_started_at = None

        if container_args.function_def.pty_info.pty_type == api_pb2.PTYInfo.PTY_TYPE_SHELL:
            target_concurrency = 1
            max_concurrency = 1
        else:
            target_concurrency = container_args.function_def.target_concurrent_inputs or 1
            max_concurrency = container_args.function_def.max_concurrent_inputs or target_concurrency

        self._target_concurrency = target_concurrency
        self._max_concurrency = max_concurrency
        self._concurrency_loop = None
        self._stop_concurrency_loop = False
        self._input_slots = InputSlots(target_concurrency)

        self._environment_name = container_args.environment_name
        self._heartbeat_loop = None
        self._heartbeat_condition = None
        self._waiting_for_memory_snapshot = False

        self._is_interactivity_enabled = False
        self._fetching_inputs = True

        self._client = client
        assert isinstance(self._client, _Client)

    @property
    def heartbeat_condition(self) -> asyncio.Condition:
        # ensures that heartbeat condition isn't assigned to an event loop until it's used for the first time
        # (On Python 3.9 and below it would be assigned to the current thread's event loop on creation)
        if self._heartbeat_condition is None:
            self._heartbeat_condition = asyncio.Condition()
        return self._heartbeat_condition

    def __new__(cls, container_args: api_pb2.ContainerArguments, client: _Client) -> "_ContainerIOManager":
        cls._singleton = super().__new__(cls)
        cls._singleton._init(container_args, client)
        return cls._singleton

    @classmethod
    def _reset_singleton(cls):
        """Only used for tests."""
        cls._singleton = None

    async def _run_heartbeat_loop(self):
        while 1:
            t0 = time.monotonic()
            try:
                if await self._heartbeat_handle_cancellations():
                    # got a cancellation event, fine to start another heartbeat immediately
                    # since the cancellation queue should be empty on the worker server
                    # however, we wait at least 1s to prevent short-circuiting the heartbeat loop
                    # in case there is ever a bug. This means it will take at least 1s between
                    # two subsequent cancellations on the same task at the moment
                    await asyncio.sleep(1.0)
                    continue
            except ClientClosed:
                logger.info("Stopping heartbeat loop due to client shutdown")
                break
            except Exception as exc:
                # don't stop heartbeat loop if there are transient exceptions!
                time_elapsed = time.monotonic() - t0
                error = exc
                logger.warning(f"Heartbeat attempt failed ({time_elapsed=}, {error=})")

            heartbeat_duration = time.monotonic() - t0
            time_until_next_hearbeat = max(0.0, HEARTBEAT_INTERVAL - heartbeat_duration)
            await asyncio.sleep(time_until_next_hearbeat)

    async def _heartbeat_handle_cancellations(self) -> bool:
        # Return True if a cancellation event was received, in that case
        # we shouldn't wait too long for another heartbeat
        async with self.heartbeat_condition:
            # Continuously wait until `waiting_for_memory_snapshot` is false.
            # TODO(matt): Verify that a `while` is necessary over an `if`. Spurious
            # wakeups could allow execution to continue despite `_waiting_for_memory_snapshot`
            # being true.
            while self._waiting_for_memory_snapshot:
                await self.heartbeat_condition.wait()

            request = api_pb2.ContainerHeartbeatRequest(canceled_inputs_return_outputs_v2=True)
            response = await retry_transient_errors(
                self._client.stub.ContainerHeartbeat, request, attempt_timeout=HEARTBEAT_TIMEOUT
            )

        if response.HasField("cancel_input_event"):
            # response.cancel_input_event.terminate_containers is never set, the server gets the worker to handle it.
            input_ids_to_cancel = response.cancel_input_event.input_ids
            if input_ids_to_cancel:
                if self._max_concurrency > 1:
                    for input_id in input_ids_to_cancel:
                        if input_id in self.current_inputs:
                            self.current_inputs[input_id].cancel()

                elif self.current_input_id and self.current_input_id in input_ids_to_cancel:
                    # This goes to a registered signal handler for sync Modal functions, or to the
                    # `SignalHandlingEventLoop` for async functions.
                    #
                    # We only send this signal on functions that do not have concurrent inputs enabled.
                    # This allows us to do fine-grained input cancellation. On sync functions, the
                    # SIGUSR1 signal should interrupt the main thread where user code is running,
                    # raising an InputCancellation() exception. On async functions, the signal should
                    # reach a handler in SignalHandlingEventLoop, which cancels the task.
                    logger.warning(f"Received a cancellation signal while processing input {self.current_input_id}")
                    os.kill(os.getpid(), signal.SIGUSR1)
            return True
        return False

    @asynccontextmanager
    async def heartbeats(self, wait_for_mem_snap: bool) -> AsyncGenerator[None, None]:
        async with TaskContext() as tc:
            self._heartbeat_loop = t = tc.create_task(self._run_heartbeat_loop())
            t.set_name("heartbeat loop")
            self._waiting_for_memory_snapshot = wait_for_mem_snap
            try:
                yield
            finally:
                t.cancel()

    def stop_heartbeat(self):
        if self._heartbeat_loop:
            self._heartbeat_loop.cancel()

    @asynccontextmanager
    async def dynamic_concurrency_manager(self) -> AsyncGenerator[None, None]:
        async with TaskContext() as tc:
            self._concurrency_loop = t = tc.create_task(self._dynamic_concurrency_loop())
            t.set_name("dynamic concurrency loop")
            try:
                yield
            finally:
                t.cancel()

    async def _dynamic_concurrency_loop(self):
        logger.debug(f"Starting dynamic concurrency loop for task {self.task_id}")
        while not self._stop_concurrency_loop:
            try:
                request = api_pb2.FunctionGetDynamicConcurrencyRequest(
                    function_id=self.function_id,
                    target_concurrency=self._target_concurrency,
                    max_concurrency=self._max_concurrency,
                )
                resp = await retry_transient_errors(
                    self._client.stub.FunctionGetDynamicConcurrency,
                    request,
                    attempt_timeout=DYNAMIC_CONCURRENCY_TIMEOUT_SECS,
                )
                if resp.concurrency != self._input_slots.value and not self._stop_concurrency_loop:
                    logger.debug(f"Dynamic concurrency set from {self._input_slots.value} to {resp.concurrency}")
                    self._input_slots.set_value(resp.concurrency)

            except Exception as exc:
                logger.debug(f"Failed to get dynamic concurrency for task {self.task_id}, {exc}")

            await asyncio.sleep(DYNAMIC_CONCURRENCY_INTERVAL_SECS)

    async def get_app_objects(self) -> RunningApp:
        req = api_pb2.AppGetObjectsRequest(app_id=self.app_id, include_unindexed=True)
        resp = await retry_transient_errors(self._client.stub.AppGetObjects, req)
        logger.debug(f"AppGetObjects received {len(resp.items)} objects for app {self.app_id}")

        tag_to_object_id = {}
        object_handle_metadata = {}
        for item in resp.items:
            handle_metadata: Optional[Message] = get_proto_oneof(item.object, "handle_metadata_oneof")
            object_handle_metadata[item.object.object_id] = handle_metadata
            if item.tag:
                tag_to_object_id[item.tag] = item.object.object_id

        return RunningApp(
            self.app_id,
            environment_name=self._environment_name,
            tag_to_object_id=tag_to_object_id,
            object_handle_metadata=object_handle_metadata,
            client=self._client,
        )

    async def get_serialized_function(self) -> Tuple[Optional[Any], Optional[Callable[..., Any]]]:
        # Fetch the serialized function definition
        request = api_pb2.FunctionGetSerializedRequest(function_id=self.function_id)
        response = await self._client.stub.FunctionGetSerialized(request)
        if response.function_serialized:
            fun = self.deserialize(response.function_serialized)
        else:
            fun = None

        if response.class_serialized:
            cls = self.deserialize(response.class_serialized)
        else:
            cls = None

        return cls, fun

    def serialize(self, obj: Any) -> bytes:
        return serialize(obj)

    def deserialize(self, data: bytes) -> Any:
        return deserialize(data, self._client)

    @synchronizer.no_io_translation
    def serialize_data_format(self, obj: Any, data_format: int) -> bytes:
        return serialize_data_format(obj, data_format)

    async def format_blob_data(self, data: bytes) -> Dict[str, Any]:
        return (
            {"data_blob_id": await blob_upload(data, self._client.stub)}
            if len(data) > MAX_OBJECT_SIZE_BYTES
            else {"data": data}
        )

    async def get_data_in(self, function_call_id: str) -> AsyncIterator[Any]:
        """Read from the `data_in` stream of a function call."""
        async for data in _stream_function_call_data(self._client, function_call_id, "data_in"):
            yield data

    async def put_data_out(
        self,
        function_call_id: str,
        start_index: int,
        data_format: int,
        messages_bytes: List[Any],
    ) -> None:
        """Put data onto the `data_out` stream of a function call.

        This is used for generator outputs, which includes web endpoint responses. Note that this
        was introduced as a performance optimization in client version 0.57, so older clients will
        still use the previous Postgres-backed system based on `FunctionPutOutputs()`.
        """
        data_chunks: List[api_pb2.DataChunk] = []
        for i, message_bytes in enumerate(messages_bytes):
            chunk = api_pb2.DataChunk(data_format=data_format, index=start_index + i)  # type: ignore
            if len(message_bytes) > MAX_OBJECT_SIZE_BYTES:
                chunk.data_blob_id = await blob_upload(message_bytes, self._client.stub)
            else:
                chunk.data = message_bytes
            data_chunks.append(chunk)

        req = api_pb2.FunctionCallPutDataRequest(function_call_id=function_call_id, data_chunks=data_chunks)
        await retry_transient_errors(self._client.stub.FunctionCallPutDataOut, req)

    async def generator_output_task(self, function_call_id: str, data_format: int, message_rx: asyncio.Queue) -> None:
        """Task that feeds generator outputs into a function call's `data_out` stream."""
        index = 1
        received_sentinel = False
        while not received_sentinel:
            message = await message_rx.get()
            if message is self._GENERATOR_STOP_SENTINEL:
                break
            # ASGI 'http.response.start' and 'http.response.body' msgs are observed to be separated by 1ms.
            # If we don't sleep here for 1ms we end up with an extra call to .put_data_out().
            if index == 1:
                await asyncio.sleep(0.001)
            messages_bytes = [serialize_data_format(message, data_format)]
            total_size = len(messages_bytes[0]) + 512
            while total_size < 16 * 1024 * 1024:  # 16 MiB, maximum size in a single message
                try:
                    message = message_rx.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if message is self._GENERATOR_STOP_SENTINEL:
                    received_sentinel = True
                    break
                else:
                    messages_bytes.append(serialize_data_format(message, data_format))
                    total_size += len(messages_bytes[-1]) + 512  # 512 bytes for estimated framing overhead
            await self.put_data_out(function_call_id, index, data_format, messages_bytes)
            index += len(messages_bytes)

    async def _queue_create(self, size: int) -> asyncio.Queue:
        """Create a queue, on the synchronicity event loop (needed on Python 3.8 and 3.9)."""
        return asyncio.Queue(size)

    async def _queue_put(self, queue: asyncio.Queue, value: Any) -> None:
        """Put a value onto a queue, using the synchronicity event loop."""
        await queue.put(value)

    def get_average_call_time(self) -> float:
        if self.calls_completed == 0:
            return 0

        return self.total_user_time / self.calls_completed

    def get_max_inputs_to_fetch(self):
        if self.calls_completed == 0:
            return 1

        return math.ceil(RTT_S / max(self.get_average_call_time(), 1e-6))

    @synchronizer.no_io_translation
    async def _generate_inputs(
        self,
        batch_max_size: int,
        batch_wait_ms: int,
    ) -> AsyncIterator[List[Tuple[str, str, api_pb2.FunctionInput]]]:
        request = api_pb2.FunctionGetInputsRequest(function_id=self.function_id)
        iteration = 0
        while self._fetching_inputs:
            await self._input_slots.acquire()

            request.average_call_time = self.get_average_call_time()
            request.max_values = self.get_max_inputs_to_fetch()  # Deprecated; remove.
            request.input_concurrency = self.get_input_concurrency()
            request.batch_max_size, request.batch_linger_ms = batch_max_size, batch_wait_ms

            yielded = False
            try:
                # If number of active inputs is at max queue size, this will block.
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
                    # for input cancellations and concurrency logic we currently assume
                    # that there is no input buffering in the container
                    assert 0 < len(response.inputs) <= max(1, request.batch_max_size)
                    inputs = []
                    final_input_received = False
                    for item in response.inputs:
                        if item.kill_switch:
                            logger.debug(f"Task {self.task_id} input kill signal input.")
                            return

                        inputs.append((item.input_id, item.function_call_id, item.input))
                        if item.input.final_input:
                            if request.batch_max_size > 0:
                                logger.debug(f"Task {self.task_id} Final input not expected in batch input stream")
                            final_input_received = True
                            break

                    # If yielded, allow input slots to be released via exit_context
                    yield inputs
                    yielded = True

                    # We only support max_inputs = 1 at the moment
                    if final_input_received or self.function_def.max_inputs == 1:
                        return
            finally:
                if not yielded:
                    self._input_slots.release()

    @synchronizer.no_io_translation
    async def run_inputs_outputs(
        self,
        finalized_functions: Dict[str, FinalizedFunction],
        batch_max_size: int = 0,
        batch_wait_ms: int = 0,
    ) -> AsyncIterator[IOContext]:
        # Ensure we do not fetch new inputs when container is too busy.
        # Before trying to fetch an input, acquire an input slot:
        # - if no input is fetched, release the input slot.
        # - or, when the output for the fetched input is sent, release the input slot.
        dynamic_concurrency_manager = (
            self.dynamic_concurrency_manager() if self._max_concurrency > self._target_concurrency else AsyncExitStack()
        )
        async with dynamic_concurrency_manager:
            async for inputs in self._generate_inputs(batch_max_size, batch_wait_ms):
                io_context = await IOContext.create(self._client, finalized_functions, inputs, batch_max_size > 0)
                for input_id in io_context.input_ids:
                    self.current_inputs[input_id] = io_context

                self.current_input_id, self.current_input_started_at = io_context.input_ids[0], time.time()
                yield io_context
                self.current_input_id, self.current_input_started_at = (None, None)

            # collect all active input slots, meaning all inputs have wrapped up.
            await self._input_slots.close()

    @synchronizer.no_io_translation
    async def _push_outputs(
        self,
        io_context: IOContext,
        started_at: float,
        data_format: "modal_proto.api_pb2.DataFormat.ValueType",
        results: List[api_pb2.GenericResult],
    ) -> None:
        output_created_at = time.time()
        outputs = [
            api_pb2.FunctionPutOutputsItem(
                input_id=input_id,
                input_started_at=started_at,
                output_created_at=output_created_at,
                result=result,
                data_format=data_format,
            )
            for input_id, result in zip(io_context.input_ids, results)
        ]
        await retry_transient_errors(
            self._client.stub.FunctionPutOutputs,
            api_pb2.FunctionPutOutputsRequest(outputs=outputs),
            additional_status_codes=[Status.RESOURCE_EXHAUSTED],
            max_retries=None,  # Retry indefinitely, trying every 1s.
        )

    def serialize_exception(self, exc: BaseException) -> bytes:
        try:
            return self.serialize(exc)
        except Exception as serialization_exc:
            # We can't always serialize exceptions.
            err = f"Failed to serialize exception {exc} of type {type(exc)}: {serialization_exc}"
            logger.info(err)
            return self.serialize(SerializationError(err))

    def serialize_traceback(self, exc: BaseException) -> Tuple[Optional[bytes], Optional[bytes]]:
        serialized_tb, tb_line_cache = None, None

        try:
            tb_dict, line_cache = extract_traceback(exc, self.task_id)
            serialized_tb = self.serialize(tb_dict)
            tb_line_cache = self.serialize(line_cache)
        except Exception:
            logger.info("Failed to serialize exception traceback.")

        return serialized_tb, tb_line_cache

    @asynccontextmanager
    async def handle_user_exception(self) -> AsyncGenerator[None, None]:
        """Sets the task as failed in a way where it's not retried.

        Used for handling exceptions from container lifecycle methods at the moment, which should
        trigger a task failure state.
        """
        try:
            yield
        except KeyboardInterrupt:
            # Send no task result in case we get sigint:ed by the runner
            # The status of the input should have been handled externally already in that case
            raise
        except BaseException as exc:
            if isinstance(exc, ImportError):
                # Catches errors raised by global scope imports
                check_fastapi_pydantic_compatibility(exc)

            # Since this is on a different thread, sys.exc_info() can't find the exception in the stack.
            print_exception(type(exc), exc, exc.__traceback__)

            serialized_tb, tb_line_cache = self.serialize_traceback(exc)

            result = api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                data=self.serialize_exception(exc),
                exception=repr(exc),
                traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                serialized_tb=serialized_tb or b"",
                tb_line_cache=tb_line_cache or b"",
            )

            req = api_pb2.TaskResultRequest(result=result)
            await retry_transient_errors(self._client.stub.TaskResult, req)

            # Shut down the task gracefully
            raise UserException()

    @asynccontextmanager
    async def handle_input_exception(
        self,
        io_context: IOContext,
        started_at: float,
    ) -> AsyncGenerator[None, None]:
        """Handle an exception while processing a function input."""
        try:
            yield
        except (KeyboardInterrupt, GeneratorExit):
            # We need to explicitly reraise these BaseExceptions to not handle them in the catch-all:
            # 1. KeyboardInterrupt can end up here even though this runs on non-main thread, since the
            #    code block yielded to could be sending back a main thread exception
            # 2. GeneratorExit - raised if this (async) generator is garbage collected while waiting
            #    for the yield. Typically on event loop shutdown
            raise
        except (InputCancellation, asyncio.CancelledError):
            # Create terminated outputs for these inputs to signal that the cancellations have been completed.
            results = [
                api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
                for _ in io_context.input_ids
            ]
            await self._push_outputs(
                io_context=io_context,
                started_at=started_at,
                data_format=api_pb2.DATA_FORMAT_PICKLE,
                results=results,
            )
            self.exit_context(started_at, io_context.input_ids)
            logger.warning(f"Successfully canceled input {io_context.input_ids}")
            return
        except BaseException as exc:
            if isinstance(exc, ImportError):
                # Catches errors raised by imports from within function body
                check_fastapi_pydantic_compatibility(exc)

            # print exception so it's logged
            print_exception(*sys.exc_info())

            serialized_tb, tb_line_cache = self.serialize_traceback(exc)

            # Note: we're not serializing the traceback since it contains
            # local references that means we can't unpickle it. We *are*
            # serializing the exception, which may have some issues (there
            # was an earlier note about it that it might not be possible
            # to unpickle it in some cases). Let's watch out for issues.

            repr_exc = repr(exc)
            if len(repr_exc) >= MAX_OBJECT_SIZE_BYTES:
                # We prevent large exception messages to avoid
                # unhandled exceptions causing inf loops
                # and just send backa trimmed version
                trimmed_bytes = len(repr_exc) - MAX_OBJECT_SIZE_BYTES - 1000
                repr_exc = repr_exc[: MAX_OBJECT_SIZE_BYTES - 1000]
                repr_exc = f"{repr_exc}...\nTrimmed {trimmed_bytes} bytes from original exception"

            data: bytes = self.serialize_exception(exc) or b""
            data_result_part = await self.format_blob_data(data)
            results = [
                api_pb2.GenericResult(
                    status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                    exception=repr_exc,
                    traceback=traceback.format_exc(),
                    serialized_tb=serialized_tb or b"",
                    tb_line_cache=tb_line_cache or b"",
                    **data_result_part,
                )
                for _ in io_context.input_ids
            ]
            await self._push_outputs(
                io_context=io_context,
                started_at=started_at,
                data_format=api_pb2.DATA_FORMAT_PICKLE,
                results=results,
            )
            self.exit_context(started_at, io_context.input_ids)

    def exit_context(self, started_at, input_ids: List[str]):
        self.total_user_time += time.time() - started_at
        self.calls_completed += 1

        for input_id in input_ids:
            self.current_inputs.pop(input_id)

        self._input_slots.release()

    @synchronizer.no_io_translation
    async def push_outputs(
        self,
        io_context: IOContext,
        started_at: float,
        data: Any,
        data_format: "modal_proto.api_pb2.DataFormat.ValueType",
    ) -> None:
        data = io_context.validate_output_data(data)
        formatted_data = await asyncio.gather(
            *[self.format_blob_data(self.serialize_data_format(d, data_format)) for d in data]
        )
        results = [
            api_pb2.GenericResult(
                status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                **d,
            )
            for d in formatted_data
        ]
        await self._push_outputs(
            io_context=io_context,
            started_at=started_at,
            data_format=data_format,
            results=results,
        )
        self.exit_context(started_at, io_context.input_ids)

    async def memory_restore(self) -> None:
        # Busy-wait for restore. `/__modal/restore-state.json` is created
        # by the worker process with updates to the container config.
        restored_path = Path(config.get("restore_state_path"))
        start = time.perf_counter()
        while not restored_path.exists():
            logger.debug(f"Waiting for restore (elapsed={time.perf_counter() - start:.3f}s)")
            await asyncio.sleep(0.01)
            continue

        logger.debug("Container: restored")

        # Look for state file and create new client with updated credentials.
        # State data is serialized with key-value pairs, example: {"task_id": "tk-000"}
        with restored_path.open("r") as file:
            restored_state = json.load(file)

        # Start a debugger if the worker tells us to
        if int(restored_state.get("snapshot_debug", 0)):
            logger.debug("Entering snapshot debugger")
            breakpoint()

        # Local ContainerIOManager state.
        for key in ["task_id", "function_id"]:
            if value := restored_state.get(key):
                logger.debug(f"Updating ContainerIOManager.{key} = {value}")
                setattr(self, key, restored_state[key])

        # Env vars and global state.
        for key, value in restored_state.items():
            # Empty string indicates that value does not need to be updated.
            if value != "":
                config.override_locally(key, value)

        # Restore input to default state.
        self.current_input_id = None
        self.current_inputs = {}
        self.current_input_started_at = None
        self._client = await _Client.from_env()

    async def memory_snapshot(self) -> None:
        """Message server indicating that function is ready to be checkpointed."""
        if self.checkpoint_id:
            logger.debug(f"Checkpoint ID: {self.checkpoint_id} (Memory Snapshot ID)")
        else:
            logger.debug("No checkpoint ID provided (Memory Snapshot ID)")

        # Pause heartbeats since they keep the client connection open which causes the snapshotter to crash
        async with self.heartbeat_condition:
            # Notify the heartbeat loop that the snapshot phase has begun in order to
            # prevent it from sending heartbeat RPCs
            self._waiting_for_memory_snapshot = True
            self.heartbeat_condition.notify_all()

            await self._client.stub.ContainerCheckpoint(
                api_pb2.ContainerCheckpointRequest(checkpoint_id=self.checkpoint_id or "")
            )

            await self._client._close(prep_for_restore=True)

            logger.debug("Memory snapshot request sent. Connection closed.")
            await self.memory_restore()
            # Turn heartbeats back on. This is safe since the snapshot RPC
            # and the restore phase has finished.
            self._waiting_for_memory_snapshot = False
            self.heartbeat_condition.notify_all()

    async def volume_commit(self, volume_ids: List[str]) -> None:
        """
        Perform volume commit for given `volume_ids`.
        Only used on container exit to persist uncommitted changes on behalf of user.
        """
        if not volume_ids:
            return
        await asyncify(os.sync)()
        results = await asyncio.gather(
            *[
                retry_transient_errors(
                    self._client.stub.VolumeCommit,
                    api_pb2.VolumeCommitRequest(volume_id=v_id),
                    max_retries=9,
                    base_delay=0.25,
                    max_delay=256,
                    delay_factor=2,
                )
                for v_id in volume_ids
            ],
            return_exceptions=True,
        )
        for volume_id, res in zip(volume_ids, results):
            if isinstance(res, Exception):
                logger.error(f"modal.Volume background commit failed for {volume_id}. Exception: {res}")
            else:
                logger.debug(f"modal.Volume background commit success for {volume_id}.")

    async def interact(self, from_breakpoint: bool = False):
        if self._is_interactivity_enabled:
            # Currently, interactivity is enabled forever
            return
        self._is_interactivity_enabled = True

        if not self.function_def.pty_info.pty_type:
            trigger = "breakpoint()" if from_breakpoint else "modal.interact()"
            raise InvalidError(f"Cannot use {trigger} without running Modal in interactive mode.")

        try:
            await self._client.stub.FunctionStartPtyShell(Empty())
        except Exception as e:
            logger.error("Failed to start PTY shell.")
            raise e

    @property
    def target_concurrency(self) -> int:
        return self._target_concurrency

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @classmethod
    def get_input_concurrency(cls) -> int:
        """
        Returns the number of usable input slots.

        If concurrency is reduced, active slots can exceed allotted slots. Returns the larger value
        in this case.
        """

        io_manager = cls._singleton
        assert io_manager
        return max(io_manager._input_slots.active, io_manager._input_slots.value)

    @classmethod
    def set_input_concurrency(cls, concurrency: int):
        """
        Edit the number of input slots.

        This disables the background loop which automatically adjusts concurrency
        within [target_concurrency, max_concurrency].
        """
        io_manager = cls._singleton
        assert io_manager
        io_manager._stop_concurrency_loop = True
        concurrency = min(concurrency, io_manager._max_concurrency)
        io_manager._input_slots.set_value(concurrency)

    @classmethod
    def stop_fetching_inputs(cls):
        assert cls._singleton
        cls._singleton._fetching_inputs = False


ContainerIOManager = synchronize_api(_ContainerIOManager)


def check_fastapi_pydantic_compatibility(exc: ImportError) -> None:
    """Add a helpful note to an exception that is likely caused by a pydantic<>fastapi version incompatibility.

    We need this becasue the legacy set of container requirements (image_builder_version=2023.12) contains a
    version of fastapi that is not forwards-compatible with pydantic 2.0+, and users commonly run into issues
    building an image that specifies a more recent version only for pydantic.
    """
    note = (
        "Please ensure that your Image contains compatible versions of fastapi and pydantic."
        " If using pydantic>=2.0, you must also install fastapi>=0.100."
    )
    name = exc.name or ""
    if name.startswith("pydantic"):
        try:
            fastapi_version = parse_major_minor_version(importlib.metadata.version("fastapi"))
            pydantic_version = parse_major_minor_version(importlib.metadata.version("pydantic"))
            if pydantic_version >= (2, 0) and fastapi_version < (0, 100):
                if sys.version_info < (3, 11):
                    # https://peps.python.org/pep-0678/
                    exc.__notes__ = [note]
                else:
                    exc.add_note(note)
        except Exception:
            # Since we're just trying to add a helpful message, don't fail here
            pass
