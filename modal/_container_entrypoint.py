# Copyright Modal Labs 2022
from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import inspect
import json
import math
import os
import pickle
import signal
import sys
import threading
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Callable, List, Optional, Set, Type

from grpclib import Status

from modal_proto import api_pb2

from ._asgi import asgi_app_wrapper, webhook_asgi_app, wsgi_app_wrapper
from ._proxy_tunnel import proxy_tunnel
from ._serialization import deserialize, deserialize_data_format, serialize, serialize_data_format
from ._traceback import extract_traceback
from ._utils.async_utils import TaskContext, asyncify, synchronize_api, synchronizer
from ._utils.blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._utils.function_utils import LocalFunctionError, is_async as get_is_async, is_global_function, method_has_params
from ._utils.grpc_utils import retry_transient_errors
from .app import _container_app, _ContainerApp, interact
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, Client, _Client
from .cls import Cls
from .config import config, logger
from .exception import InputCancellation, InvalidError
from .functions import Function, _Function, _set_current_context_ids, _stream_function_call_data
from .partial_function import _find_callables_for_obj, _PartialFunctionFlags
from .stub import _Stub

if TYPE_CHECKING:
    from types import ModuleType

MAX_OUTPUT_BATCH_SIZE: int = 49

RTT_S: float = 0.5  # conservative estimate of RTT in seconds.


class UserException(Exception):
    # Used to shut down the task gracefully
    pass


class SignalHandlingEventLoop:
    """Run an async event loop as a context manager and handle signals.

    The following signals are handled while a coroutine is running on the event loop until
    completion (and then handlers are deregistered):

    - `SIGUSR1`: converted to an async task cancellation. Note that this only affects the event
      loop, and the signal handler defined here doesn't run for sync functions.
    """

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        if sys.version_info[:2] >= (3, 9):
            self.loop.run_until_complete(self.loop.shutdown_default_executor())  # Introduced in Python 3.9
        self.loop.close()

    def run(self, coro):
        task = asyncio.ensure_future(coro, loop=self.loop)

        # Before Python 3.9 there is no argument to Task.cancel
        if sys.version_info[:2] >= (3, 9):
            self.loop.add_signal_handler(signal.SIGUSR1, task.cancel, "Input was cancelled by user")
        else:
            self.loop.add_signal_handler(signal.SIGUSR1, task.cancel)

        try:
            return self.loop.run_until_complete(task)
        finally:
            self.loop.remove_signal_handler(signal.SIGUSR1)


class _FunctionIOManager:
    """Synchronizes all RPC calls and network operations for a running container.

    TODO: maybe we shouldn't synchronize the whole class.
    Then we could potentially move a bunch of the global functions onto it.
    """

    _GENERATOR_STOP_SENTINEL = object()

    def __init__(self, container_args: api_pb2.ContainerArguments, client: _Client):
        self.cancelled_input_ids: Set[str] = set()
        self.task_id = container_args.task_id
        self.function_id = container_args.function_id
        self.app_id = container_args.app_id
        self.function_def = container_args.function_def
        self.checkpoint_id = container_args.checkpoint_id

        self.calls_completed = 0
        self.total_user_time: float = 0.0
        self.current_input_id: Optional[str] = None
        self.current_input_started_at: Optional[float] = None

        self._stub_name = self.function_def.stub_name
        self._input_concurrency: Optional[int] = None

        self._semaphore: Optional[asyncio.Semaphore] = None
        self._environment_name = container_args.environment_name
        self._waiting_for_checkpoint = False
        self._heartbeat_loop = None

        self._client = client
        assert isinstance(self._client, _Client)

    async def initialize_app(self) -> _ContainerApp:
        await _container_app.init(self._client, self.app_id, self._stub_name, self._environment_name, self.function_def)
        return _container_app

    async def _run_heartbeat_loop(self):
        while 1:
            t0 = time.monotonic()
            if await self._heartbeat():
                # got a cancellation event, fine to start another heartbeat immediately
                # since the cancellation queue should be empty on the worker server
                # however, we wait at least 1s to prevent short-circuiting the heartbeat loop
                # in case there is ever a bug. This means it will take at least 1s between
                # two subsequent cancellations on the same task at the moment
                time_until_next_hearbeat = 1.0
            else:
                heartbeat_duration = time.monotonic() - t0
                time_until_next_hearbeat = max(0.0, HEARTBEAT_INTERVAL - heartbeat_duration)
            await asyncio.sleep(time_until_next_hearbeat)

    async def _heartbeat(self) -> bool:
        # Return True if a cancellation event was received, in that case we shouldn't wait too long for another heartbeat

        # Don't send heartbeats for tasks waiting to be checkpointed.
        # Calling gRPC methods open new connections which block the
        # checkpointing process.
        if self._waiting_for_checkpoint:
            return False

        request = api_pb2.ContainerHeartbeatRequest(supports_graceful_input_cancellation=True)
        if self.current_input_id is not None:
            request.current_input_id = self.current_input_id
        if self.current_input_started_at is not None:
            request.current_input_started_at = self.current_input_started_at

        # TODO(erikbern): capture exceptions?
        response = await retry_transient_errors(
            self._client.stub.ContainerHeartbeat, request, attempt_timeout=HEARTBEAT_TIMEOUT
        )

        if response.HasField("cancel_input_event"):
            # Pause processing of the current input by signaling self a SIGUSR1.
            input_ids_to_cancel = response.cancel_input_event.input_ids
            if input_ids_to_cancel:
                if self._input_concurrency > 1:
                    logger.info(
                        "Shutting down task to stop some subset of inputs (concurrent functions don't support fine-grained cancellation)"
                    )
                    # This is equivalent to a task cancellation or preemption from worker code,
                    # except we do not send a SIGKILL to forcefully exit after 30 seconds.
                    #
                    # SIGINT always interrupts the main thread, but not any auxiliary threads. On a
                    # sync function without concurrent inputs, this raises a KeyboardInterrupt. When
                    # there are concurrent inputs, we cannot interrupt the thread pool, but the
                    # interpreter stops waiting for daemon threads and exits. On async functions,
                    # this signal lands outside the event loop, stopping `run_until_complete()`.
                    os.kill(os.getpid(), signal.SIGINT)

                elif self.current_input_id in input_ids_to_cancel:
                    # This goes to a registered signal handler for sync Modal functions, or to the
                    # `SignalHandlingEventLoop` for async functions.
                    #
                    # We only send this signal on functions that do not have concurrent inputs enabled.
                    # This allows us to do fine-grained input cancellation. On sync functions, the
                    # SIGUSR1 signal should interrupt the main thread where user code is running,
                    # raising an InputCancellation() exception. On async functions, the signal should
                    # reach a handler in SignalHandlingEventLoop, which cancels the task.
                    os.kill(os.getpid(), signal.SIGUSR1)
            return True
        return False

    @contextlib.asynccontextmanager
    async def heartbeats(self):
        async with TaskContext() as tc:
            self._heartbeat_loop = t = tc.create_task(self._run_heartbeat_loop())
            t.set_name("heartbeat loop")
            yield

    def stop_heartbeat(self):
        if self._heartbeat_loop:
            self._heartbeat_loop.cancel()

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

    @synchronizer.no_io_translation
    def serialize_data_format(self, obj: Any, data_format: int) -> bytes:
        return serialize_data_format(obj, data_format)

    def deserialize_data_format(self, data: bytes, data_format: int) -> Any:
        return deserialize_data_format(data, data_format, self._client)

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

    @synchronizer.no_io_translation
    async def _generate_inputs(self) -> AsyncIterator[tuple[str, str, api_pb2.FunctionInput]]:
        request = api_pb2.FunctionGetInputsRequest(function_id=self.function_id)
        eof_received = False
        iteration = 0
        while not eof_received:
            request.average_call_time = self.get_average_call_time()
            request.max_values = self.get_max_inputs_to_fetch()  # Deprecated; remove.
            request.input_concurrency = self._input_concurrency

            await self._semaphore.acquire()
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
                    # for input cancellations we currently assume there is no input buffering in the container
                    assert len(response.inputs) == 1

                    for item in response.inputs:
                        if item.kill_switch:
                            logger.debug(f"Task {self.task_id} input kill signal input.")
                            eof_received = True
                            break
                        if item.input_id in self.cancelled_input_ids:
                            continue

                        # If we got a pointer to a blob, download it from S3.
                        if item.input.WhichOneof("args_oneof") == "args_blob_id":
                            input_pb = await self.populate_input_blobs(item.input)
                        else:
                            input_pb = item.input

                        # If yielded, allow semaphore to be released via complete_call
                        yield (item.input_id, item.function_call_id, input_pb)
                        yielded = True

                        # We only support max_inputs = 1 at the moment
                        if item.input.final_input or self.function_def.max_inputs == 1:
                            eof_received = True
                            break
            finally:
                if not yielded:
                    self._semaphore.release()

    @synchronizer.no_io_translation
    async def run_inputs_outputs(self, input_concurrency: int = 1) -> AsyncIterator[tuple[str, str, Any, Any]]:
        # Ensure we do not fetch new inputs when container is too busy.
        # Before trying to fetch an input, acquire the semaphore:
        # - if no input is fetched, release the semaphore.
        # - or, when the output for the fetched input is sent, release the semaphore.
        self._input_concurrency = input_concurrency
        self._semaphore = asyncio.Semaphore(input_concurrency)

        try:
            async for input_id, function_call_id, input_pb in self._generate_inputs():
                args, kwargs = self.deserialize(input_pb.args) if input_pb.args else ((), {})
                self.current_input_id, self.current_input_started_at = (input_id, time.time())
                yield input_id, function_call_id, args, kwargs
                self.current_input_id, self.current_input_started_at = (None, None)
        finally:
            # collect all active input slots, meaning all inputs have wrapped up.
            for _ in range(input_concurrency):
                await self._semaphore.acquire()

    async def _push_output(self, input_id, started_at: float, data_format=api_pb2.DATA_FORMAT_UNSPECIFIED, **kwargs):
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
        """Sets the task as failed in a way where it's not retried.

        Used for handling exceptions from container lifecycle methods at the moment, which should
        trigger a task failure state.
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
    async def handle_input_exception(self, input_id, started_at: float) -> AsyncGenerator[None, None]:
        """Handle an exception while processing a function input."""
        try:
            yield
        except KeyboardInterrupt:
            raise
        except (InputCancellation, asyncio.CancelledError):
            # just skip creating any output for this input and keep going with the next instead
            # it should have been marked as cancelled already in the backend at this point so it
            # won't be retried
            logger.info(f"The current input ({input_id=}) was cancelled by a user request")
            await self.complete_call(started_at)
            return
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

    @synchronizer.no_io_translation
    async def push_output(self, input_id, started_at: float, data: Any, data_format: int) -> None:
        await self._push_output(
            input_id,
            started_at=started_at,
            data_format=data_format,
            status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
            data=self.serialize_data_format(data, data_format),
        )
        await self.complete_call(started_at)

    async def restore(self) -> None:
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

        # Local FunctionIOManager state.
        for key in ["task_id", "function_id"]:
            if value := restored_state.get(key):
                logger.debug(f"Updating FunctionIOManager.{key} = {value}")
                setattr(self, key, restored_state[key])

        # Env vars and global state.
        for key, value in restored_state.items():
            # Empty string indicates that value does not need to be updated.
            if value != "":
                config.override_locally(key, value)

        # Restore input to default state.
        self.current_input_id = None
        self.current_input_started_at = None

        self._client = await _Client.from_env()
        self._waiting_for_checkpoint = False

    async def checkpoint(self) -> None:
        """Message server indicating that function is ready to be checkpointed."""
        if self.checkpoint_id:
            logger.debug(f"Checkpoint ID: {self.checkpoint_id}")

        await self._client.stub.ContainerCheckpoint(
            api_pb2.ContainerCheckpointRequest(checkpoint_id=self.checkpoint_id)
        )

        self._waiting_for_checkpoint = True
        await self._client._close()

        logger.debug("Checkpointing request sent. Connection closed.")
        await self.restore()

    async def volume_commit(self, volume_ids: list[str]) -> None:
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


FunctionIOManager = synchronize_api(_FunctionIOManager)


def call_function_sync(
    function_io_manager,  #: FunctionIOManager,  TODO: this type is generated at runtime
    imp_fun: ImportedFunction,
):
    def run_input(input_id: str, function_call_id: str, args: Any, kwargs: Any) -> None:
        started_at = time.time()
        reset_context = _set_current_context_ids(input_id, function_call_id)
        with function_io_manager.handle_input_exception(input_id, started_at):
            logger.debug(f"Starting input {input_id} (sync)")
            res = imp_fun.fun(*args, **kwargs)
            logger.debug(f"Finished input {input_id} (sync)")

            # TODO(erikbern): any exception below shouldn't be considered a user exception
            if imp_fun.is_generator:
                if not inspect.isgenerator(res):
                    raise InvalidError(f"Generator function returned value of type {type(res)}")

                # Send up to this many outputs at a time.
                generator_queue: asyncio.Queue[Any] = function_io_manager._queue_create(1024)
                generator_output_task = function_io_manager.generator_output_task(
                    function_call_id,
                    imp_fun.data_format,
                    generator_queue,
                    _future=True,  # Synchronicity magic to return a future.
                )

                item_count = 0
                for value in res:
                    function_io_manager._queue_put(generator_queue, value)
                    item_count += 1

                function_io_manager._queue_put(generator_queue, _FunctionIOManager._GENERATOR_STOP_SENTINEL)
                generator_output_task.result()  # Wait to finish sending generator outputs.
                message = api_pb2.GeneratorDone(items_total=item_count)
                function_io_manager.push_output(input_id, started_at, message, api_pb2.DATA_FORMAT_GENERATOR_DONE)
            else:
                if inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                    raise InvalidError(
                        f"Sync (non-generator) function return value of type {type(res)}."
                        " You might need to use @stub.function(..., is_generator=True)."
                    )
                function_io_manager.push_output(input_id, started_at, res, imp_fun.data_format)
        reset_context()

    if imp_fun.input_concurrency > 1:
        # We can't use `concurrent.futures.ThreadPoolExecutor` here because in Python 3.11+, this
        # class has no workaround that allows us to exit the Python interpreter process without
        # waiting for the worker threads to finish. We need this behavior on SIGINT.

        import queue
        import threading

        spawned_workers = 0
        inputs: queue.Queue[Any] = queue.Queue()
        finished = threading.Event()

        def worker_thread():
            while not finished.is_set():
                try:
                    args = inputs.get(timeout=1)
                except queue.Empty:
                    continue
                try:
                    run_input(*args)
                except BaseException:
                    pass
                inputs.task_done()

        for input_id, function_call_id, args, kwargs in function_io_manager.run_inputs_outputs(
            imp_fun.input_concurrency
        ):
            if spawned_workers < imp_fun.input_concurrency:
                threading.Thread(target=worker_thread, daemon=True).start()
                spawned_workers += 1
            inputs.put((input_id, function_call_id, args, kwargs))

        finished.set()
        inputs.join()

    else:
        for input_id, function_call_id, args, kwargs in function_io_manager.run_inputs_outputs(
            imp_fun.input_concurrency
        ):
            run_input(input_id, function_call_id, args, kwargs)


async def call_function_async(
    function_io_manager,  #: FunctionIOManager,  TODO: this type is generated at runtime
    imp_fun: ImportedFunction,
):
    async def run_input(input_id: str, function_call_id: str, args: Any, kwargs: Any) -> None:
        started_at = time.time()
        reset_context = _set_current_context_ids(input_id, function_call_id)
        async with function_io_manager.handle_input_exception.aio(input_id, started_at):
            logger.debug(f"Starting input {input_id} (async)")
            res = imp_fun.fun(*args, **kwargs)
            logger.debug(f"Finished input {input_id} (async)")

            # TODO(erikbern): any exception below shouldn't be considered a user exception
            if imp_fun.is_generator:
                if not inspect.isasyncgen(res):
                    raise InvalidError(f"Async generator function returned value of type {type(res)}")

                # Send up to this many outputs at a time.
                generator_queue: asyncio.Queue[Any] = await function_io_manager._queue_create.aio(1024)
                generator_output_task = asyncio.create_task(
                    function_io_manager.generator_output_task.aio(
                        function_call_id,
                        imp_fun.data_format,
                        generator_queue,
                    )
                )

                item_count = 0
                async for value in res:
                    await function_io_manager._queue_put.aio(generator_queue, value)
                    item_count += 1

                await function_io_manager._queue_put.aio(generator_queue, _FunctionIOManager._GENERATOR_STOP_SENTINEL)
                await generator_output_task  # Wait to finish sending generator outputs.
                message = api_pb2.GeneratorDone(items_total=item_count)
                await function_io_manager.push_output.aio(
                    input_id, started_at, message, api_pb2.DATA_FORMAT_GENERATOR_DONE
                )
            else:
                if not inspect.iscoroutine(res) or inspect.isgenerator(res) or inspect.isasyncgen(res):
                    raise InvalidError(
                        f"Async (non-generator) function returned value of type {type(res)}"
                        " You might need to use @stub.function(..., is_generator=True)."
                    )
                value = await res
                await function_io_manager.push_output.aio(input_id, started_at, value, imp_fun.data_format)
        reset_context()

    if imp_fun.input_concurrency > 1:
        async with TaskContext() as execution_context:
            async for input_id, function_call_id, args, kwargs in function_io_manager.run_inputs_outputs.aio(
                imp_fun.input_concurrency
            ):
                execution_context.create_task(run_input(input_id, function_call_id, args, kwargs))
    else:
        async for input_id, function_call_id, args, kwargs in function_io_manager.run_inputs_outputs.aio(
            imp_fun.input_concurrency
        ):
            await run_input(input_id, function_call_id, args, kwargs)


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


def import_function(
    function_def: api_pb2.Function,
    ser_cls,
    ser_fun,
    ser_params: Optional[bytes],
    function_io_manager,
) -> ImportedFunction:
    # This is not in function_io_manager, so that any global scope code that runs during import
    # runs on the main thread.
    module: Optional[ModuleType] = None
    cls: Optional[Type] = None
    fun: Callable
    function: Optional[_Function] = None
    active_stub: Optional[_Stub] = None
    pty_info: api_pb2.PTYInfo = function_def.pty_info

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
            # This is a function
            cls = None
            f = getattr(module, qual_name)
            if isinstance(f, Function):
                function = synchronizer._translate_in(f)
                fun = function.get_raw_f()
                active_stub = function._stub
            else:
                fun = f
        elif len(parts) == 2:
            # This is a method on a class
            cls_name, fun_name = parts
            cls = getattr(module, cls_name)
            if isinstance(cls, Cls):
                # The cls decorator is in global scope
                _cls = synchronizer._translate_in(cls)
                fun = _cls._callables[fun_name]
                function = _cls._functions.get(fun_name)
                active_stub = _cls._stub
            else:
                # This is a raw class
                fun = getattr(cls, fun_name)
        else:
            raise InvalidError(f"Invalid function qualname {qual_name}")

    # If the cls/function decorator was applied in local scope, but the stub is global, we can look it up
    if active_stub is None and function_def.stub_name:
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
    if pty_info.pty_type == api_pb2.PTYInfo.PTY_TYPE_SHELL:
        # Concurrency doesn't apply for `modal shell`.
        input_concurrency = 1
    else:
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

    if not pty_info.pty_type:  # do not wrap PTY-enabled functions
        if function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP:
            # function returns an asgi_app, that we can use as a callable.
            asgi_app = fun()
            fun = asgi_app_wrapper(asgi_app, function_io_manager)
            is_async = True
            is_generator = True
            data_format = api_pb2.DATA_FORMAT_ASGI
        elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP:
            # function returns an wsgi_app, that we can use as a callable.
            wsgi_app = fun()
            fun = wsgi_app_wrapper(wsgi_app, function_io_manager)
            is_async = True
            is_generator = True
            data_format = api_pb2.DATA_FORMAT_ASGI
        elif function_def.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
            # function is webhook without an ASGI app. Create one for it.
            asgi_app = webhook_asgi_app(fun, function_def.webhook_config.method)
            fun = asgi_app_wrapper(asgi_app, function_io_manager)
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


async def call_lifecycle_functions(
    function_io_manager,  #: FunctionIOManager,  TODO: this type is generated at runtime
    funcs: Iterable[Callable],
) -> None:
    """Call function(s), can be sync or async, but any return values are ignored."""
    with function_io_manager.handle_user_exception():
        for func in funcs:
            # We are deprecating parameterized exit methods but want to gracefully handle old code.
            # We can remove this once the deprecation in the actual @exit decorator is enforced.
            args = (None, None, None) if method_has_params(func) else ()
            res = func(*args)
            if inspect.iscoroutine(res):
                await res


def main(container_args: api_pb2.ContainerArguments, client: Client):
    # This is a bit weird but we need both the blocking and async versions of FunctionIOManager.
    # At some point, we should fix that by having built-in support for running "user code"
    function_io_manager = FunctionIOManager(container_args, client)

    # Define a global app (need to do this before imports).
    container_app = function_io_manager.initialize_app()

    with function_io_manager.heartbeats(), SignalHandlingEventLoop() as event_loop:
        # If this is a serialized function, fetch the definition from the server
        if container_args.function_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED:
            ser_cls, ser_fun = function_io_manager.get_serialized_function()
        else:
            ser_cls, ser_fun = None, None

        # Initialize the function, importing user code.
        with function_io_manager.handle_user_exception():
            imp_fun = import_function(
                container_args.function_def, ser_cls, ser_fun, container_args.serialized_params, function_io_manager
            )

        # Hydrate all function dependencies.
        if imp_fun.function:
            dep_object_ids: list[str] = [dep.object_id for dep in container_args.function_def.object_dependencies]
            container_app.hydrate_function_deps(imp_fun.function, dep_object_ids)

        # Identify all "enter" methods that need to run before we checkpoint.
        if imp_fun.obj is not None and not imp_fun.is_auto_snapshot:
            pre_checkpoint_methods = _find_callables_for_obj(imp_fun.obj, _PartialFunctionFlags.ENTER_PRE_CHECKPOINT)
            event_loop.run(call_lifecycle_functions(function_io_manager, pre_checkpoint_methods.values()))

        # If this container is being used to create a checkpoint, checkpoint the container after
        # global imports and innitialization. Checkpointed containers run from this point onwards.
        if container_args.function_def.is_checkpointing_function:
            function_io_manager.checkpoint()

        # Install hooks for interactive functions.
        if container_args.function_def.pty_info.pty_type != api_pb2.PTYInfo.PTY_TYPE_UNSPECIFIED:

            def breakpoint_wrapper():
                # note: it would be nice to not have breakpoint_wrapper() included in the backtrace
                interact()
                import pdb

                pdb.set_trace()

            sys.breakpointhook = breakpoint_wrapper

        # Identify the "enter" methods to run after resuming from a checkpoint.
        if imp_fun.obj is not None and not imp_fun.is_auto_snapshot:
            post_checkpoint_methods = _find_callables_for_obj(imp_fun.obj, _PartialFunctionFlags.ENTER_POST_CHECKPOINT)
            event_loop.run(call_lifecycle_functions(function_io_manager, post_checkpoint_methods.values()))

        # Execute the function.
        try:
            if imp_fun.is_async:
                event_loop.run(call_function_async(function_io_manager, imp_fun))
            else:
                # Set up a signal handler for `SIGUSR1`, which gets translated to an InputCancellation
                # during function execution. This is sent to cancel inputs from the user.
                def _cancel_input_signal_handler(signum, stackframe):
                    raise InputCancellation("Input was cancelled by user")

                signal.signal(signal.SIGUSR1, _cancel_input_signal_handler)

                call_function_sync(function_io_manager, imp_fun)
        finally:
            # Run exit handlers. From this point onward, ignore all SIGINT signals that come from
            # graceful shutdowns originating on the worker, as well as stray SIGUSR1 signals that
            # may have been sent to cancel inputs.
            int_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            usr1_handler = signal.signal(signal.SIGUSR1, signal.SIG_IGN)

            try:
                # Identify "exit" methods and run them.
                if imp_fun.obj is not None and not imp_fun.is_auto_snapshot:
                    exit_methods = _find_callables_for_obj(imp_fun.obj, _PartialFunctionFlags.EXIT)
                    event_loop.run(call_lifecycle_functions(function_io_manager, exit_methods.values()))

                # Finally, commit on exit to catch uncommitted volume changes and surface background
                # commit errors.
                function_io_manager.volume_commit(
                    [v.volume_id for v in container_args.function_def.volume_mounts if v.allow_background_commits]
                )
                # Avoid "Canceling remaining unfinished task" warnings.
                function_io_manager.stop_heartbeat()

            finally:
                # Restore the original signal handler, needed for container_test hygiene since the
                # test runs `main()` multiple times in the same process.
                signal.signal(signal.SIGINT, int_handler)
                signal.signal(signal.SIGUSR1, usr1_handler)


if __name__ == "__main__":
    logger.debug("Container: starting")

    container_args = api_pb2.ContainerArguments()
    container_args.ParseFromString(base64.b64decode(sys.argv[1]))

    # Note that we're creating the client in a synchronous context, but it will be running in a separate thread.
    # This is good because if the function is long running then we the client can still send heartbeats
    # The only caveat is a bunch of calls will now cross threads, which adds a bit of overhead?
    client = Client.from_env()

    try:
        with proxy_tunnel(container_args.proxy_info):
            try:
                main(container_args, client)
            except UserException:
                logger.info("User exception caught, exiting")
    except KeyboardInterrupt:
        logger.debug("Container: interrupted")

    # Detect if any non-daemon threads are still running, which will prevent the Python interpreter
    # from shutting down. The sleep(0) here is needed for finished ThreadPoolExecutor resources to
    # shut down without triggering this warning (e.g., `@wsgi_app()`).
    time.sleep(0)
    lingering_threads: List[threading.Thread] = []
    for thread in threading.enumerate():
        current_thread = threading.get_ident()
        if thread.ident is not None and thread.ident != current_thread and not thread.daemon and thread.is_alive():
            lingering_threads.append(thread)
    if lingering_threads:
        thread_names = ", ".join(t.name for t in lingering_threads)
        logger.warning(
            f"Detected {len(lingering_threads)} background thread(s) [{thread_names}] still running after container exit. This will prevent runner shutdown for up to 30 seconds."
        )

    logger.debug("Container: done")
