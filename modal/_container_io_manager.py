# Copyright Modal Labs 2024
import asyncio
import json
import math
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Callable, ClassVar, List, Optional, Set, Tuple

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message
from grpclib import Status
from synchronicity.async_wrap import asynccontextmanager

from modal_proto import api_pb2

from ._serialization import deserialize, deserialize_data_format, serialize, serialize_data_format
from ._traceback import extract_traceback
from ._utils.async_utils import TaskContext, asyncify, synchronize_api, synchronizer
from ._utils.blob_utils import MAX_OBJECT_SIZE_BYTES, blob_download, blob_upload
from ._utils.function_utils import _stream_function_call_data
from ._utils.grpc_utils import get_proto_oneof, retry_transient_errors
from .client import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, _Client
from .config import config, logger
from .exception import InputCancellation, InvalidError
from .running_app import RunningApp

MAX_OUTPUT_BATCH_SIZE: int = 49

RTT_S: float = 0.5  # conservative estimate of RTT in seconds.


class UserException(Exception):
    """Used to shut down the task gracefully."""


class Sentinel:
    """Used to get type-stubs to work with this object."""


class _ContainerIOManager:
    """Synchronizes all RPC calls and network operations for a running container.

    TODO: maybe we shouldn't synchronize the whole class.
    Then we could potentially move a bunch of the global functions onto it.
    """

    cancelled_input_ids: Set[str]
    task_id: str
    function_id: str
    app_id: str
    function_def: api_pb2.Function
    checkpoint_id: Optional[str]

    calls_completed: int
    total_user_time: float
    current_input_id: Optional[str]
    current_input_started_at: Optional[float]

    _input_concurrency: Optional[int]
    _semaphore: Optional[asyncio.Semaphore]
    _environment_name: str
    _waiting_for_memory_snapshot: bool
    _heartbeat_loop: Optional[asyncio.Task]

    _is_interactivity_enabled: bool
    _fetching_inputs: bool

    _client: _Client

    _GENERATOR_STOP_SENTINEL: ClassVar[Sentinel] = Sentinel()
    _singleton: ClassVar[Optional["_ContainerIOManager"]] = None

    def _init(self, container_args: api_pb2.ContainerArguments, client: _Client):
        self.cancelled_input_ids = set()
        self.task_id = container_args.task_id
        self.function_id = container_args.function_id
        self.app_id = container_args.app_id
        self.function_def = container_args.function_def
        self.checkpoint_id = container_args.checkpoint_id or None

        self.calls_completed = 0
        self.total_user_time = 0.0
        self.current_input_id = None
        self.current_input_started_at = None

        self._input_concurrency = None

        self._semaphore = None
        self._environment_name = container_args.environment_name
        self._waiting_for_memory_snapshot = False
        self._heartbeat_loop = None

        self._is_interactivity_enabled = False
        self._fetching_inputs = True

        self._client = client
        assert isinstance(self._client, _Client)

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

        # Don't send heartbeats for tasks waiting to be checkpointed.
        # Calling gRPC methods open new connections which block the
        # checkpointing process.
        if self._waiting_for_memory_snapshot:
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
                        "Shutting down task to stop some subset of inputs "
                        "(concurrent functions don't support fine-grained cancellation)"
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

    @asynccontextmanager
    async def heartbeats(self) -> AsyncGenerator[None, None]:
        async with TaskContext() as tc:
            self._heartbeat_loop = t = tc.create_task(self._run_heartbeat_loop())
            t.set_name("heartbeat loop")
            try:
                yield
            finally:
                t.cancel()

    def stop_heartbeat(self):
        if self._heartbeat_loop:
            self._heartbeat_loop.cancel()

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
        )

    async def get_serialized_function(self) -> Tuple[Optional[Any], Callable]:
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
    async def _generate_inputs(self) -> AsyncIterator[Tuple[str, str, api_pb2.FunctionInput]]:
        request = api_pb2.FunctionGetInputsRequest(function_id=self.function_id)
        eof_received = False
        iteration = 0
        while not eof_received and self._fetching_inputs:
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
                    # for input cancellations and concurrency logic we currently assume
                    # that there is no input buffering in the container
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
    async def run_inputs_outputs(self, input_concurrency: int = 1) -> AsyncIterator[Tuple[str, str, str, Any, Any]]:
        # Ensure we do not fetch new inputs when container is too busy.
        # Before trying to fetch an input, acquire the semaphore:
        # - if no input is fetched, release the semaphore.
        # - or, when the output for the fetched input is sent, release the semaphore.
        self._input_concurrency = input_concurrency
        self._semaphore = asyncio.Semaphore(input_concurrency)

        async for input_id, function_call_id, input_pb in self._generate_inputs():
            args, kwargs = self.deserialize(input_pb.args) if input_pb.args else ((), {})
            self.current_input_id, self.current_input_started_at = (input_id, time.time())
            yield input_id, function_call_id, input_pb.method_name, args, kwargs
            self.current_input_id, self.current_input_started_at = (None, None)

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

    @asynccontextmanager
    async def handle_input_exception(self, input_id, started_at: float) -> AsyncGenerator[None, None]:
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
            # just skip creating any output for this input and keep going with the next instead
            # it should have been marked as cancelled already in the backend at this point so it
            # won't be retried
            logger.warning(f"The current input ({input_id=}) was cancelled by a user request")
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

            repr_exc = repr(exc)
            if len(repr_exc) >= MAX_OBJECT_SIZE_BYTES:
                # We prevent large exception messages to avoid
                # unhandled exceptions causing inf loops
                # and just send backa trimmed version
                trimmed_bytes = len(repr_exc) - MAX_OBJECT_SIZE_BYTES - 1000
                repr_exc = repr_exc[: MAX_OBJECT_SIZE_BYTES - 1000]
                repr_exc = f"{repr_exc}...\nTrimmed {trimmed_bytes} bytes from original exception"

            await self._push_output(
                input_id,
                started_at=started_at,
                data_format=api_pb2.DATA_FORMAT_PICKLE,
                status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                data=self.serialize_exception(exc),
                exception=repr_exc,
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
        self.current_input_started_at = None

        self._client = await _Client.from_env()
        self._waiting_for_memory_snapshot = False

    async def memory_snapshot(self) -> None:
        """Message server indicating that function is ready to be checkpointed."""
        if self.checkpoint_id:
            logger.debug(f"Checkpoint ID: {self.checkpoint_id} (Memory Snapshot ID)")

        await self._client.stub.ContainerCheckpoint(
            api_pb2.ContainerCheckpointRequest(checkpoint_id=self.checkpoint_id)
        )

        self._waiting_for_memory_snapshot = True
        await self._client._close(forget_credentials=True)

        logger.debug("Memory snapshot request sent. Connection closed.")
        await self.memory_restore()

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

    async def interact(self):
        if self._is_interactivity_enabled:
            # Currently, interactivity is enabled forever
            return
        self._is_interactivity_enabled = True

        if not self.function_def.pty_info:
            raise InvalidError(
                "Interactivity is not enabled in this function. "
                "Use MODAL_INTERACTIVE_FUNCTIONS=1 to enable interactivity."
            )

        if self.function_def.concurrency_limit > 1:
            print(
                "Warning: Interactivity is not supported on functions with concurrency > 1. "
                "You may experience unexpected behavior."
            )

        # todo(nathan): add warning if concurrency limit > 1. but idk how to check this here
        # todo(nathan): check if function interactivity is enabled
        try:
            await self._client.stub.FunctionStartPtyShell(Empty())
        except Exception as e:
            print("Error: Failed to start PTY shell.")
            raise e

    @classmethod
    def stop_fetching_inputs(cls):
        assert cls._singleton
        cls._singleton._fetching_inputs = False


ContainerIOManager = synchronize_api(_ContainerIOManager)
