# Copyright Modal Labs 2022
import asyncio
import inspect
import os
import pickle
import posixpath
import time
import typing
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import PurePath
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from aiostream import pipe, stream
from google.protobuf.message import Message
from grpclib import GRPCError, Status
from synchronicity.exceptions import UserCodeException

from modal import _pty
from modal._types import typechecked
from modal_proto import api_pb2
from modal_utils.async_utils import (
    queue_batch_iterator,
    synchronize_api,
    synchronizer,
    warn_if_generator_is_not_consumed,
)
from modal_utils.grpc_utils import retry_transient_errors

from ._blob_utils import (
    BLOB_MAX_PARALLELISM,
    MAX_OBJECT_SIZE_BYTES,
    blob_download,
    blob_upload,
)
from ._function_utils import FunctionInfo
from ._location import parse_cloud_provider
from ._output import OutputManager
from ._resolver import Resolver
from ._serialization import deserialize, serialize
from ._traceback import append_modal_tb
from .call_graph import InputInfo, _reconstruct_call_graph
from .client import _Client
from .config import config, logger
from .exception import (
    ExecutionError,
    InvalidError,
    RemoteError,
    TimeoutError as _TimeoutError,
    deprecation_error,
    deprecation_warning,
)
from .gpu import GPU_T, display_gpu_config, parse_gpu_config
from .image import _Image
from .mount import _get_client_mount, _Mount
from .network_file_system import _NetworkFileSystem
from .object import _Handle, _Provider
from .proxy import _Proxy
from .retries import Retries
from .schedule import Schedule
from .secret import _Secret
from .volume import _Volume

ATTEMPT_TIMEOUT_GRACE_PERIOD = 5  # seconds


if typing.TYPE_CHECKING:
    import modal.stub


def exc_with_hints(exc: BaseException):
    """mdmd:hidden"""
    if isinstance(exc, ImportError) and exc.msg == "attempted relative import with no known parent package":
        exc.msg += """\n
HINT: For relative imports to work, you might need to run your modal app as a module. Try:
- `python -m my_pkg.my_app` instead of `python my_pkg/my_app.py`
- `modal deploy my_pkg.my_app` instead of `modal deploy my_pkg/my_app.py`
"""
    elif isinstance(
        exc, RuntimeError
    ) and "CUDA error: no kernel image is available for execution on the device" in str(exc):
        msg = (
            exc.args[0]
            + """\n
HINT: This error usually indicates an outdated CUDA version. Older versions of torch (<=1.12)
come with CUDA 10.2 by default. If pinning to an older torch version, you can specify a CUDA version
manually, for example:
-  image.pip_install("torch==1.12.1+cu116", find_links="https://download.pytorch.org/whl/torch_stable.html")
"""
        )
        exc.args = (msg,)

    return exc


async def _process_result(result, stub, client=None):
    if result.WhichOneof("data_oneof") == "data_blob_id":
        data = await blob_download(result.data_blob_id, stub)
    else:
        data = result.data

    if result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
        raise _TimeoutError(result.exception)
    elif result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
        if data:
            try:
                exc = deserialize(data, client)
            except Exception as deser_exc:
                raise ExecutionError(
                    "Could not deserialize remote exception due to local error:\n"
                    + f"{deser_exc}\n"
                    + "This can happen if your local environment does not have the remote exception definitions.\n"
                    + "Here is the remote traceback:\n"
                    + f"{result.traceback}"
                )
            if not isinstance(exc, BaseException):
                raise ExecutionError(f"Got remote exception of incorrect type {type(exc)}")

            if result.serialized_tb:
                try:
                    tb_dict = deserialize(result.serialized_tb, client)
                    line_cache = deserialize(result.tb_line_cache, client)
                    append_modal_tb(exc, tb_dict, line_cache)
                except Exception:
                    pass
            uc_exc = UserCodeException(exc_with_hints(exc))
            raise uc_exc
        raise RemoteError(result.exception)

    try:
        return deserialize(data, client)
    except ModuleNotFoundError as deser_exc:
        raise ExecutionError(
            "Could not deserialize result due to error:\n"
            + f"{deser_exc}\n"
            + "This can happen if your local environment does not have a module that was used to construct the result. \n"
        )


async def _create_input(args, kwargs, client, idx=None) -> api_pb2.FunctionPutInputsItem:
    """Serialize function arguments and create a FunctionInput protobuf,
    uploading to blob storage if needed.
    """

    args_serialized = serialize((args, kwargs))

    if len(args_serialized) > MAX_OBJECT_SIZE_BYTES:
        args_blob_id = await blob_upload(args_serialized, client.stub)

        return api_pb2.FunctionPutInputsItem(
            input=api_pb2.FunctionInput(args_blob_id=args_blob_id),
            idx=idx,
        )
    else:
        return api_pb2.FunctionPutInputsItem(
            input=api_pb2.FunctionInput(args=args_serialized),
            idx=idx,
        )


@dataclass
class _OutputValue:
    # box class for distinguishing None results from non-existing/None markers
    value: Any


class _Invocation:
    """Internal client representation of a single-input call to a Modal Function or Generator"""

    def __init__(self, stub, function_call_id, client=None):
        self.stub = stub
        self.client = client  # Used by the deserializer.
        self.function_call_id = function_call_id  # TODO: remove and use only input_id

    @staticmethod
    async def create(function_id, args, kwargs, client):
        if not function_id:
            raise InvalidError(
                "The function has not been initialized.\n"
                "\n"
                "Modal functions can only be called within an app. "
                "Try calling it from another running modal function or from an app run context:\n\n"
                "with stub.run():\n"
                "    my_modal_function.call()\n"
            )
        request = api_pb2.FunctionMapRequest(
            function_id=function_id,
            parent_input_id=current_input_id(),
            function_call_type=api_pb2.FUNCTION_CALL_TYPE_UNARY,
        )
        response = await retry_transient_errors(client.stub.FunctionMap, request)

        function_call_id = response.function_call_id

        item = await _create_input(args, kwargs, client)
        request_put = api_pb2.FunctionPutInputsRequest(
            function_id=function_id, inputs=[item], function_call_id=function_call_id
        )
        inputs_response: api_pb2.FunctionPutInputsResponse = await retry_transient_errors(
            client.stub.FunctionPutInputs,
            request_put,
        )
        processed_inputs = inputs_response.inputs
        if not processed_inputs:
            raise Exception("Could not create function call - the input queue seems to be full")
        return _Invocation(client.stub, function_call_id, client)

    async def pop_function_call_outputs(self, timeout: Optional[float], clear_on_success: bool):
        t0 = time.time()
        if timeout is None:
            backend_timeout = config["outputs_timeout"]
        else:
            backend_timeout = min(config["outputs_timeout"], timeout)  # refresh backend call every 55s

        while True:
            # always execute at least one poll for results, regardless if timeout is 0
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=self.function_call_id,
                timeout=backend_timeout,
                last_entry_id="0-0",
                clear_on_success=clear_on_success,
            )
            response = await retry_transient_errors(
                self.stub.FunctionGetOutputs,
                request,
                attempt_timeout=backend_timeout + ATTEMPT_TIMEOUT_GRACE_PERIOD,
            )
            if len(response.outputs) > 0:
                for item in response.outputs:
                    yield item.result
                return

            if timeout is not None:
                # update timeout in retry loop
                backend_timeout = min(config["outputs_timeout"], t0 + timeout - time.time())
                if backend_timeout < 0:
                    break

    async def run_function(self):
        # waits indefinitely for a single result for the function, and clear the outputs buffer after
        result = (await stream.list(self.pop_function_call_outputs(timeout=None, clear_on_success=True)))[0]
        assert not result.gen_status
        return await _process_result(result, self.stub, self.client)

    async def poll_function(self, timeout: Optional[float] = None):
        # waits up to timeout for a result from a function
        # * timeout=0 means a single poll
        # * timeout=None means wait indefinitely
        # raises TimeoutError if there is no result before timeout
        # Intended to be used for future polling, and as such keeps
        # results around after returning them
        results = await stream.list(self.pop_function_call_outputs(timeout=timeout, clear_on_success=False))

        if len(results) == 0:
            raise TimeoutError()

        return await _process_result(results[0], self.stub, self.client)

    async def run_generator(self):
        last_entry_id = "0-0"
        completed = False
        try:
            while not completed:
                request = api_pb2.FunctionGetOutputsRequest(
                    function_call_id=self.function_call_id,
                    timeout=config["outputs_timeout"],
                    last_entry_id=last_entry_id,
                    clear_on_success=False,  # there could be more results
                )
                response = await retry_transient_errors(
                    self.stub.FunctionGetOutputs,
                    request,
                    attempt_timeout=config["outputs_timeout"] + ATTEMPT_TIMEOUT_GRACE_PERIOD,
                )
                if len(response.outputs) > 0:
                    last_entry_id = response.last_entry_id
                    for item in response.outputs:
                        if item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                            completed = True
                            break
                        yield await _process_result(item.result, self.stub, self.client)
        finally:
            # "ack" that we have all outputs we are interested in and let backend clear results
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=self.function_call_id,
                timeout=0,
                last_entry_id="0-0",
                clear_on_success=True,
            )
            await self.stub.FunctionGetOutputs(request)


MAP_INVOCATION_CHUNK_SIZE = 100


async def _map_invocation(
    function_id: str,
    input_stream: AsyncIterable[Any],
    kwargs: Dict[str, Any],
    client: _Client,
    is_generator: bool,
    order_outputs: bool,
    return_exceptions: bool,
    count_update_callback: Optional[Callable[[int, int], None]],
):
    request = api_pb2.FunctionMapRequest(
        function_id=function_id,
        parent_input_id=current_input_id(),
        function_call_type=api_pb2.FUNCTION_CALL_TYPE_MAP,
        return_exceptions=return_exceptions,
    )
    response = await retry_transient_errors(client.stub.FunctionMap, request)

    function_call_id = response.function_call_id

    have_all_inputs = False
    num_inputs = 0
    num_outputs = 0
    pending_outputs: Dict[str, int] = {}  # Map input_id -> next expected gen_index value
    completed_outputs: Set[str] = set()  # Set of input_ids whose outputs are complete (expecting no more values)

    input_queue: asyncio.Queue = asyncio.Queue()

    async def create_input(arg):
        nonlocal num_inputs
        idx = num_inputs
        num_inputs += 1
        item = await _create_input(arg, kwargs, client, idx=idx)
        return item

    async def drain_input_generator():
        # Parallelize uploading blobs
        proto_input_stream = input_stream | pipe.map(create_input, ordered=True, task_limit=BLOB_MAX_PARALLELISM)
        async with proto_input_stream.stream() as streamer:
            async for item in streamer:
                await input_queue.put(item)

        # close queue iterator
        await input_queue.put(None)
        yield

    async def pump_inputs():
        nonlocal have_all_inputs
        async for items in queue_batch_iterator(input_queue, MAP_INVOCATION_CHUNK_SIZE):
            request = api_pb2.FunctionPutInputsRequest(
                function_id=function_id, inputs=items, function_call_id=function_call_id
            )
            logger.debug(
                f"Pushing {len(items)} inputs to server. Num queued inputs awaiting push is {input_queue.qsize()}."
            )
            resp = await retry_transient_errors(
                client.stub.FunctionPutInputs,
                request,
                max_retries=None,
                max_delay=10,
                additional_status_codes=[Status.RESOURCE_EXHAUSTED],
            )
            for item in resp.inputs:
                pending_outputs.setdefault(item.input_id, 0)
            logger.debug(
                f"Successfully pushed {len(items)} inputs to server. Num queued inputs awaiting push is {input_queue.qsize()}."
            )

        have_all_inputs = True
        yield

    async def get_all_outputs():
        nonlocal num_inputs, num_outputs, have_all_inputs
        last_entry_id = "0-0"
        while not have_all_inputs or len(pending_outputs) > len(completed_outputs):
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=function_call_id,
                timeout=config["outputs_timeout"],
                last_entry_id=last_entry_id,
                clear_on_success=False,
            )
            response = await retry_transient_errors(
                client.stub.FunctionGetOutputs,
                request,
                max_retries=20,
                attempt_timeout=config["outputs_timeout"] + ATTEMPT_TIMEOUT_GRACE_PERIOD,
            )

            if len(response.outputs) == 0:
                continue

            last_entry_id = response.last_entry_id
            for item in response.outputs:
                pending_outputs.setdefault(item.input_id, 0)
                if item.input_id in completed_outputs or (
                    item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_INCOMPLETE
                    and item.gen_index < pending_outputs[item.input_id]
                ):
                    # If this input is already completed, or if it's a generator output and we've already seen a later
                    # output, it means the output has already been processed and was received again due
                    # to a duplicate output enqueue on the server.
                    continue

                if is_generator:
                    # Mark this input completed if the generator completed successfully, or it crashed (exception, timeout, etc).
                    if item.result.gen_status == api_pb2.GenericResult.GENERATOR_STATUS_COMPLETE:
                        completed_outputs.add(item.input_id)
                        num_outputs += 1
                    elif item.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                        completed_outputs.add(item.input_id)
                        num_outputs += 1
                        yield item
                    else:
                        assert pending_outputs[item.input_id] == item.gen_index
                        pending_outputs[item.input_id] += 1
                        yield item
                else:
                    completed_outputs.add(item.input_id)
                    num_outputs += 1
                    yield item

    async def get_all_outputs_and_clean_up():
        try:
            async for item in get_all_outputs():
                yield item
        finally:
            # "ack" that we have all outputs we are interested in and let backend clear results
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=function_call_id,
                timeout=0,
                last_entry_id="0-0",
                clear_on_success=True,
            )
            await retry_transient_errors(client.stub.FunctionGetOutputs, request)

    async def fetch_output(item):
        try:
            output = await _process_result(item.result, client.stub, client)
        except Exception as e:
            if return_exceptions:
                output = e
            else:
                raise e
        return (item.idx, output)

    async def poll_outputs():
        outputs = stream.iterate(get_all_outputs_and_clean_up())
        outputs_fetched = outputs | pipe.map(fetch_output, ordered=True, task_limit=BLOB_MAX_PARALLELISM)

        # map to store out-of-order outputs received
        received_outputs = {}
        output_idx = 0

        async with outputs_fetched.stream() as streamer:
            async for idx, output in streamer:
                if count_update_callback is not None:
                    count_update_callback(num_outputs, num_inputs)
                if is_generator:
                    yield _OutputValue(output)
                elif not order_outputs:
                    yield _OutputValue(output)
                else:
                    # hold on to outputs for function maps, so we can reorder them correctly.
                    received_outputs[idx] = output
                    while output_idx in received_outputs:
                        output = received_outputs.pop(output_idx)
                        yield _OutputValue(output)
                        output_idx += 1

        assert len(received_outputs) == 0

    response_gen = stream.merge(drain_input_generator(), pump_inputs(), poll_outputs())

    async with response_gen.stream() as streamer:
        async for response in streamer:
            if response is not None:
                yield response.value


# Wrapper type for api_pb2.FunctionStats
@dataclass
class FunctionStats:
    """Simple data structure storing stats for a running function."""

    backlog: int
    num_active_runners: int
    num_total_runners: int


class _FunctionHandle(_Handle, type_prefix="fu"):
    """Interact with a Modal Function of a live app."""

    _web_url: Optional[str]
    _info: Optional[FunctionInfo]
    _stub: Optional["modal.stub._Stub"]  # TODO(erikbern): remove
    _is_remote_cls_method: bool = False
    _function_name: Optional[str]

    def _initialize_from_empty(self):
        self._progress = None
        self._is_generator = None
        self._info = None
        self._web_url = None
        self._output_mgr: Optional[OutputManager] = None
        self._mute_cancellation = (
            False  # set when a user terminates the app intentionally, to prevent useless traceback spam
        )
        self._function_name = None
        self._stub = None  # TODO(erikbern): remove
        self._self_obj = None

    def _initialize_from_local(self, stub, info: FunctionInfo):
        # note that this is not a full hydration of the function, as it doesn't yet get an object_id etc.
        self._stub = stub  # TODO(erikbern): remove
        self._info = info

    def _hydrate_metadata(self, metadata: Message):
        # makes function usable
        assert isinstance(metadata, (api_pb2.Function, api_pb2.FunctionHandleMetadata))
        self._is_generator = metadata.function_type == api_pb2.Function.FUNCTION_TYPE_GENERATOR
        self._web_url = metadata.web_url
        self._function_name = metadata.function_name

    async def _make_bound_function_handle(self, *args: Iterable[Any], **kwargs: Dict[str, Any]) -> "_FunctionHandle":
        assert self.is_hydrated(), "Cannot make bound function handle from unhydrated handle."

        if len(args) + len(kwargs) == 0:
            # short circuit if no args, don't need a special object.
            return self

        new_handle = _FunctionHandle._new()
        new_handle._initialize_from_local(self._stub, self._info)

        serialized_params = pickle.dumps((args, kwargs))
        req = api_pb2.FunctionBindParamsRequest(
            function_id=self._object_id,
            serialized_params=serialized_params,
        )
        response = await self._client.stub.FunctionBindParams(req)
        new_handle._hydrate(response.bound_function_id, self._client, response.handle_metadata)
        new_handle._is_remote_cls_method = True
        return new_handle

    def _get_is_remote_cls_method(self):
        return self._is_remote_cls_method

    def _get_info(self):
        return self._info

    def _get_self_obj(self):
        return self._self_obj

    def _get_metadata(self):
        return api_pb2.FunctionHandleMetadata(
            function_name=self._function_name,
            function_type=api_pb2.Function.FUNCTION_TYPE_GENERATOR
            if self._is_generator
            else api_pb2.Function.FUNCTION_TYPE_FUNCTION,
            web_url=self._web_url,
        )

    def _set_mute_cancellation(self, value: bool = True):
        self._mute_cancellation = value

    def _set_output_mgr(self, output_mgr: OutputManager):
        self._output_mgr = output_mgr

    def _get_function(self) -> "_Function":
        # TODO(erikbern): don't use the stub here, just return the function, should be 1:1
        return self._stub[self._info.get_tag()]

    @property
    def web_url(self) -> str:
        """URL of a Function running as a web endpoint."""
        return self._web_url

    @property
    def is_generator(self) -> bool:
        return self._is_generator

    def _track_function_invocation(self):
        if self._stub and self._stub.app:
            self._stub.app.track_function_invocation()

    async def _map(self, input_stream: AsyncIterable[Any], order_outputs: bool, return_exceptions: bool, kwargs={}):
        if self._web_url:
            raise InvalidError(
                "A web endpoint function cannot be directly invoked for parallel remote execution. "
                f"Invoke this function via its web url '{self._web_url}' or call it locally: {self._function_name}()."
            )

        if order_outputs and self._is_generator:
            raise ValueError("Can't return ordered results for a generator")

        count_update_callback = (
            self._output_mgr.function_progress_callback(self._function_name, total=None) if self._output_mgr else None
        )

        self._track_function_invocation()
        async for item in _map_invocation(
            self._object_id,
            input_stream,
            kwargs,
            self._client,
            self._is_generator,
            order_outputs,
            return_exceptions,
            count_update_callback,
        ):
            yield item

    @warn_if_generator_is_not_consumed
    async def map(
        self,
        *input_iterators,  # one input iterator per argument in the mapped-over function/generator
        kwargs={},  # any extra keyword arguments for the function
        order_outputs=None,  # defaults to True for regular functions, False for generators
        return_exceptions=False,  # whether to propogate exceptions (False) or aggregate them in the results list (True)
    ) -> AsyncGenerator[Any, None]:
        """Parallel map over a set of inputs.

        Takes one iterator argument per argument in the function being mapped over.

        Example:
        ```python
        @stub.function()
        def my_func(a):
            return a ** 2

        @stub.local_entrypoint()
        def main():
            assert list(my_func.map([1, 2, 3, 4])) == [1, 4, 9, 16]
        ```

        If applied to a `stub.function`, `map()` returns one result per input and the output order
        is guaranteed to be the same as the input order. Set `order_outputs=False` to return results
        in the order that they are completed instead.

        If applied to a `stub.generator`, the results are returned as they are finished and can be
        out of order. By yielding zero or more than once, mapping over generators can also be used
        as a "flat map".

        `return_exceptions` can be used to treat exceptions as successful results:
        ```python
        @stub.function()
        def my_func(a):
            if a == 2:
                raise Exception("ohno")
            return a ** 2

        @stub.local_entrypoint()
        def main():
            # [0, 1, UserCodeException(Exception('ohno'))]
            print(list(my_func.map(range(3), return_exceptions=True)))
        ```
        """
        if order_outputs is None:
            order_outputs = not self._is_generator

        input_stream = stream.zip(*(stream.iterate(it) for it in input_iterators))
        async for item in self._map(input_stream, order_outputs, return_exceptions, kwargs):
            yield item

    async def for_each(self, *input_iterators, kwargs={}, ignore_exceptions=False):
        """Execute function for all outputs, ignoring outputs

        Convenient alias for `.map()` in cases where the function just needs to be called.
        as the caller doesn't have to consume the generator to process the inputs.
        """
        async for _ in self.map(
            *input_iterators, kwargs=kwargs, order_outputs=False, return_exceptions=ignore_exceptions
        ):
            pass

    @warn_if_generator_is_not_consumed
    async def starmap(
        self, input_iterator, kwargs={}, order_outputs=None, return_exceptions=False
    ) -> AsyncGenerator[typing.Any, None]:
        """Like `map` but spreads arguments over multiple function arguments

        Assumes every input is a sequence (e.g. a tuple).

        Example:
        ```python
        @stub.function()
        def my_func(a, b):
            return a + b

        @stub.local_entrypoint()
        def main():
            assert list(my_func.starmap([(1, 2), (3, 4)])) == [3, 7]
        ```
        """
        if order_outputs is None:
            order_outputs = not self._is_generator

        input_stream = stream.iterate(input_iterator)
        async for item in self._map(input_stream, order_outputs, return_exceptions, kwargs):
            yield item

    async def _call_function(self, args, kwargs):
        self._track_function_invocation()
        invocation = await _Invocation.create(self._object_id, args, kwargs, self._client)
        try:
            return await invocation.run_function()
        except asyncio.CancelledError:
            # this can happen if the user terminates a program, triggering a cancellation cascade
            if not self._mute_cancellation:
                raise

    async def _call_function_nowait(self, args, kwargs):
        self._track_function_invocation()
        return await _Invocation.create(self._object_id, args, kwargs, self._client)

    @warn_if_generator_is_not_consumed
    async def _call_generator(self, args, kwargs):
        self._track_function_invocation()
        invocation = await _Invocation.create(self._object_id, args, kwargs, self._client)
        async for res in invocation.run_generator():
            yield res

    async def _call_generator_nowait(self, args, kwargs):
        self._track_function_invocation()
        return await _Invocation.create(self._object_id, args, kwargs, self._client)

    def call(self, *args, **kwargs) -> Awaitable[Any]:  # TODO: Generics/TypeVars
        """
        Calls the function remotely, executing it with the given arguments and returning the execution's result.
        """
        if self._web_url:
            raise InvalidError(
                "A web endpoint function cannot be invoked for remote execution with `.call`. "
                f"Invoke this function via its web url '{self._web_url}' or call it locally: {self._function_name}()."
            )
        if self._is_generator:
            return self._call_generator(args, kwargs)  # type: ignore
        else:
            return self._call_function(args, kwargs)

    @synchronizer.nowrap
    def __call__(self, *args, **kwargs) -> Any:  # TODO: Generics/TypeVars
        if self._get_is_remote_cls_method():  # TODO(elias): change parametrization so this is isn't needed
            return self.call(*args, **kwargs)

        info = self._get_info()
        if not info:
            msg = (
                "The definition for this function is missing so it is not possible to invoke it locally. "
                "If this function was retrieved via `Function.lookup` you need to use `.call()`."
            )
            raise AttributeError(msg)

        self_obj = self._get_self_obj()
        if self_obj:
            # This is a method on a class, so bind the self to the function
            fun = info.raw_f.__get__(self_obj)
        else:
            fun = info.raw_f
        return fun(*args, **kwargs)

    async def spawn(self, *args, **kwargs) -> Optional["_FunctionCall"]:
        """Calls the function with the given arguments, without waiting for the results.

        Returns a `modal.functions.FunctionCall` object, that can later be polled or waited for using `.get(timeout=...)`.
        Conceptually similar to `multiprocessing.pool.apply_async`, or a Future/Promise in other contexts.

        *Note:* `.spawn()` on a modal generator function does call and execute the generator, but does not currently
        return a function handle for polling the result.
        """
        if self._is_generator:
            await self._call_generator_nowait(args, kwargs)
            return None

        invocation = await self._call_function_nowait(args, kwargs)
        return _FunctionCall._new_hydrated(invocation.function_call_id, invocation.client, None)

    def get_raw_f(self) -> Callable[..., Any]:
        """Return the inner Python object wrapped by this Modal Function."""
        if not self._info:
            raise AttributeError("_info has not been set on this FunctionHandle and not available in this context")

        return self._info.raw_f

    async def get_current_stats(self) -> FunctionStats:
        """Return a `FunctionStats` object describing the current function's queue and runner counts."""

        resp = await self._client.stub.FunctionGetCurrentStats(
            api_pb2.FunctionGetCurrentStatsRequest(function_id=self._object_id)
        )
        return FunctionStats(
            backlog=resp.backlog, num_active_runners=resp.num_active_tasks, num_total_runners=resp.num_total_tasks
        )

    def bind_obj(self, obj, objtype) -> "_FunctionHandle":
        # This is needed to bind "self" to methods for direct __call__
        self._self_obj = obj

        # TODO(erikbern): we're mutating self directly here, as opposed to returning a different _FunctionHandle
        # We should fix this in the future since it probably precludes using classmethods/staticmethods
        return self

    def __get__(self, obj, objtype=None) -> "_FunctionHandle":
        deprecation_warning(
            date(2023, 5, 9),
            "Using the `@stub.function` decorator on methods is deprecated."
            " Use the @method decorator instead."
            " See https://modal.com/docs/guide/lifecycle-functions",
        )
        return self.bind_obj(obj, objtype)


FunctionHandle = synchronize_api(_FunctionHandle)


class _Function(_Provider[_FunctionHandle]):
    """Functions are the basic units of serverless execution on Modal.

    Generally, you will not construct a `Function` directly. Instead, use the
    `@stub.function()` decorator on the `Stub` object for your application.
    """

    # TODO: more type annotations
    _secrets: Collection[_Secret]
    _info: FunctionInfo
    _mounts: Collection[_Mount]
    _network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem]
    _allow_cross_region_volumes: bool
    _volumes: Dict[Union[str, os.PathLike], _Volume]
    _image: Optional[_Image]
    _gpu: Optional[GPU_T]
    _cloud: Optional[str]
    _handle: _FunctionHandle
    _stub: "modal.stub._Stub"
    _is_builder_function: bool
    _retry_policy: Optional[api_pb2.FunctionRetryPolicy]

    @staticmethod
    def from_args(
        info: FunctionInfo,
        stub,
        image=None,
        secret: Optional[_Secret] = None,
        secrets: Collection[_Secret] = (),
        schedule: Optional[Schedule] = None,
        is_generator=False,
        gpu: GPU_T = None,
        # TODO: maybe break this out into a separate decorator for notebooks.
        mounts: Collection[_Mount] = (),
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        allow_cross_region_volumes: bool = False,
        volumes: Dict[Union[str, os.PathLike], _Volume] = {},
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        memory: Optional[int] = None,
        proxy: Optional[_Proxy] = None,
        retries: Optional[Union[int, Retries]] = None,
        timeout: Optional[int] = None,
        concurrency_limit: Optional[int] = None,
        container_idle_timeout: Optional[int] = None,
        cpu: Optional[float] = None,
        keep_warm: Optional[int] = None,
        interactive: bool = False,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        is_builder_function: bool = False,
        cls: Optional[type] = None,
    ) -> None:
        """mdmd:hidden"""
        handle = _FunctionHandle._new()
        handle._initialize_from_local(stub, info)

        tag = info.get_tag()

        if stub is not None and stub.app is not None:
            # If the container is running, and we recognize this function, hydrate it
            # TODO(erikbern): later when we merge apps and stubs, there should be no separate objects on the app,
            # and there should be no need to "steal" ids
            running_handle = stub.app._tag_to_object.get(tag)
            if running_handle is not None:
                handle._hydrate_from_other(running_handle)

        raw_f = info.raw_f
        assert callable(raw_f)
        if schedule is not None:
            if not info.is_nullary():
                raise InvalidError(
                    f"Function {raw_f} has a schedule, so it needs to support calling it with no arguments"
                )

        all_mounts = [
            _get_client_mount(),  # client
            *mounts,  # explicit mounts
        ]
        # TODO (elias): Clean up mount logic, this is quite messy:
        if stub:
            all_mounts.extend(stub._get_deduplicated_function_mounts(info.get_mounts()))  # implicit mounts
        else:
            all_mounts.extend(info.get_mounts().values())  # this would typically only happen for builder functions

        if secret:
            secrets = [secret, *secrets]

        if isinstance(retries, int):
            retry_policy = Retries(
                max_retries=retries,
                initial_delay=1.0,
                backoff_coefficient=1.0,
            )._to_proto()
        elif isinstance(retries, Retries):
            retry_policy = retries._to_proto()
        elif retries is None:
            retry_policy = None
        else:
            raise InvalidError(
                f"Function {raw_f} retries must be an integer or instance of modal.Retries. Found: {type(retries)}"
            )

        gpu_config = parse_gpu_config(gpu)

        if proxy:
            # HACK: remove this once we stop using ssh tunnels for this.
            if image:
                image = image.apt_install("autossh")

        if interactive and concurrency_limit and concurrency_limit > 1:
            warnings.warn(
                "Interactive functions require `concurrency_limit=1`. The concurrency limit will be overridden."
            )
            concurrency_limit = 1

        if keep_warm is True:
            deprecation_error(
                date(2023, 3, 3),
                "Setting `keep_warm=True` is deprecated. Pass an explicit warm pool size instead, e.g. `keep_warm=2`.",
            )

        if not cloud and not is_builder_function:
            cloud = config.get("default_cloud")
        if cloud:
            cloud_provider = parse_cloud_provider(cloud)
        else:
            cloud_provider = None

        panel_items = [
            str(i)
            for i in [
                *mounts,
                image,
                *secrets,
                *network_file_systems.values(),
                *volumes.values(),
            ]
        ]
        if gpu:
            panel_items.append(display_gpu_config(gpu))
        if cloud:
            panel_items.append(f"Cloud({cloud.upper()})")

        if is_generator and webhook_config:
            if webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION:
                raise InvalidError(
                    """Webhooks cannot be generators. If you want a streaming response, see https://modal.com/docs/guide/streaming-endpoints
                    """
                )
            else:
                raise InvalidError("Webhooks cannot be generators")

        async def _preload(resolver: Resolver, existing_object_id: Optional[str]) -> _FunctionHandle:
            if is_generator:
                function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
            else:
                function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

            req = api_pb2.FunctionPrecreateRequest(
                app_id=resolver.app_id,
                function_name=info.function_name,
                function_type=function_type,
                webhook_config=webhook_config,
                existing_function_id=existing_object_id,
            )
            response = await resolver.client.stub.FunctionPrecreate(req)
            # Update the precreated function handle (todo: hack until we merge providers/handles)
            handle._hydrate(response.function_id, resolver.client, response.handle_metadata)
            return handle

        async def _load(resolver: Resolver, existing_object_id: Optional[str]) -> _FunctionHandle:
            # TODO: should we really join recursively here? Maybe it's better to move this logic to the app class?
            status_row = resolver.add_status_row()
            status_row.message(f"Creating {tag}...")

            if proxy:
                proxy_id = (await resolver.load(proxy)).object_id
            else:
                proxy_id = None

            # Mount point path validation for volumes and shared volumes
            def _validate_mount_points(
                display_name: str, volume_likes: Dict[Union[str, os.PathLike], Union[_Volume, _NetworkFileSystem]]
            ) -> List[Tuple[str, Union[_Volume, _NetworkFileSystem]]]:
                validated = []
                for path, vol in volume_likes.items():
                    path = PurePath(path).as_posix()
                    abs_path = posixpath.abspath(path)

                    if path != abs_path:
                        raise InvalidError(f"{display_name} {path} must be a canonical, absolute path.")
                    elif abs_path == "/":
                        raise InvalidError(f"{display_name} {path} cannot be mounted into root directory.")
                    elif abs_path == "/root":
                        raise InvalidError(f"{display_name} {path} cannot be mounted at '/root'.")
                    elif abs_path == "/tmp":
                        raise InvalidError(f"{display_name} {path} cannot be mounted at '/tmp'.")
                    validated.append((path, vol))
                return validated

            async def _load_ids(providers) -> typing.List[str]:
                loaded_handles = await asyncio.gather(*[resolver.load(provider) for provider in providers])
                return [handle.object_id for handle in loaded_handles]

            async def image_loader():
                if image is not None:
                    if not isinstance(image, _Image):
                        raise InvalidError(f"Expected modal.Image object. Got {type(image)}.")
                    image_id = (await resolver.load(image)).object_id
                else:
                    image_id = None  # Happens if it's a notebook function
                return image_id

            # validation
            if not isinstance(network_file_systems, dict):
                raise InvalidError(
                    "network_file_systems must be a dict[str, NetworkFileSystem] where the keys are paths"
                )
            validated_network_file_systems = _validate_mount_points("Shared volume", network_file_systems)

            async def network_file_system_loader():
                network_file_system_mounts = []
                volume_ids = await _load_ids([vol for _, vol in validated_network_file_systems])
                # Relies on dicts being ordered (true as of Python 3.6).
                for ((path, _), volume_id) in zip(validated_network_file_systems, volume_ids):
                    network_file_system_mounts.append(
                        api_pb2.SharedVolumeMount(
                            mount_path=path,
                            shared_volume_id=volume_id,
                            allow_cross_region=allow_cross_region_volumes,
                        )
                    )
                return network_file_system_mounts

            if not isinstance(volumes, dict):
                raise InvalidError("volumes must be a dict[str, Volume] where the keys are paths")
            validated_volumes = _validate_mount_points("Volume", volumes)
            # We don't support mounting a volume in more than one location
            volume_to_paths: Dict[_Volume, List[str]] = {}
            for (path, volume) in validated_volumes:
                volume_to_paths.setdefault(volume, []).append(path)
            for paths in volume_to_paths.values():
                if len(paths) > 1:
                    conflicting = ", ".join(paths)
                    raise InvalidError(
                        f"The same Volume cannot be mounted in multiple locations for the same function: {conflicting}"
                    )

            async def volume_loader():
                volume_mounts = []
                volume_ids = await _load_ids([vol for _, vol in validated_volumes])
                # Relies on dicts being ordered (true as of Python 3.6).
                for ((path, _), volume_id) in zip(validated_volumes, volume_ids):
                    volume_mounts.append(
                        api_pb2.VolumeMount(
                            mount_path=path,
                            volume_id=volume_id,
                        )
                    )
                return volume_mounts

            if is_generator:
                function_type = api_pb2.Function.FUNCTION_TYPE_GENERATOR
            else:
                function_type = api_pb2.Function.FUNCTION_TYPE_FUNCTION

            if cpu is not None and cpu < 0.25:
                raise InvalidError(f"Invalid fractional CPU value {cpu}. Cannot have less than 0.25 CPU resources.")
            milli_cpu = int(1000 * cpu) if cpu is not None else None

            if interactive:
                pty_info = _pty.get_pty_info()
            else:
                pty_info = None

            if info.is_serialized():
                # Use cloudpickle. Used when working w/ Jupyter notebooks.
                # serialize at _load time, not function decoration time
                # otherwise we can't capture a surrounding class for lifetime methods etc.
                function_serialized = info.serialized_function()
                class_serialized = serialize(cls) if cls is not None else None
            else:
                function_serialized = None
                class_serialized = None

            stub_name = ""
            if stub and stub.name:
                stub_name = stub.name

            mount_ids, secret_ids, image_id, network_file_system_mounts, volume_mounts = await asyncio.gather(
                _load_ids(all_mounts),
                _load_ids(secrets),
                image_loader(),
                network_file_system_loader(),
                volume_loader(),
            )

            # Create function remotely
            function_definition = api_pb2.Function(
                module_name=info.module_name,
                function_name=info.function_name,
                mount_ids=mount_ids,
                secret_ids=secret_ids,
                image_id=image_id,
                definition_type=info.definition_type,
                function_serialized=function_serialized,
                class_serialized=class_serialized,
                function_type=function_type,
                resources=api_pb2.Resources(milli_cpu=milli_cpu, gpu_config=gpu_config, memory_mb=memory),
                webhook_config=webhook_config,
                shared_volume_mounts=network_file_system_mounts,
                volume_mounts=volume_mounts,
                proxy_id=proxy_id,
                retry_policy=retry_policy,
                timeout_secs=timeout,
                task_idle_timeout_secs=container_idle_timeout,
                concurrency_limit=concurrency_limit,
                pty_info=pty_info,
                cloud_provider=cloud_provider,
                warm_pool_size=keep_warm,
                runtime=config.get("function_runtime"),
                stub_name=stub_name,
                is_builder_function=is_builder_function,
            )
            request = api_pb2.FunctionCreateRequest(
                app_id=resolver.app_id,
                function=function_definition,
                schedule=schedule.proto_message if schedule is not None else None,
                existing_function_id=existing_object_id,
            )
            try:
                response = await resolver.client.stub.FunctionCreate(request)
            except GRPCError as exc:
                if exc.status == Status.INVALID_ARGUMENT:
                    raise InvalidError(exc.message)
                if exc.status == Status.FAILED_PRECONDITION:
                    raise InvalidError(exc.message)
                raise

            if response.function.web_url:
                # Ensure terms used here match terms used in modal.com/docs/guide/webhook-urls doc.
                if response.function.web_url_info.truncated:
                    suffix = " [grey70](label truncated)[/grey70]"
                elif response.function.web_url_info.has_unique_hash:
                    suffix = " [grey70](label includes conflict-avoidance hash)[/grey70]"
                elif response.function.web_url_info.label_stolen:
                    suffix = " [grey70](label stolen)[/grey70]"
                else:
                    suffix = ""
                # TODO: this is only printed when we're showing progress. Maybe move this somewhere else.
                status_row.finish(f"Created {tag} => [magenta underline]{response.web_url}[/magenta underline]{suffix}")
            else:
                status_row.finish(f"Created {tag}.")

            handle._hydrate(response.function_id, resolver.client, response.handle_metadata)
            return handle

        rep = f"Function({tag})"
        obj = _Function._from_loader(_load, rep, preload=_preload)
        obj._handle = handle
        # TODO(erikbern): almost all of these are only needed because of modal.cli.run.shell
        obj._allow_cross_region_volumes = allow_cross_region_volumes
        obj._cloud = cloud
        obj._image = image
        obj._info = info
        obj._gpu = gpu
        obj._gpu_config = gpu_config
        obj._mounts = mounts
        obj._panel_items = panel_items
        obj._raw_f = raw_f
        obj._secrets = secrets
        obj._network_file_systems = network_file_systems
        obj._tag = tag
        obj._all_mounts = all_mounts  # needed for modal.serve file watching
        return obj

    def get_panel_items(self) -> List[str]:
        return self._panel_items

    @property
    def tag(self):
        """mdmd:hidden"""
        return self._tag

    def get_build_def(self):
        """mdmd:hidden"""
        # Used to check whether we should rebuild an image using run_function
        # Plaintext source and arg definition for the function, so it's part of the image
        # hash. We can't use the cloudpickle hash because it's not very stable.
        kwargs = dict(
            secrets=repr(self._secrets),
            gpu_config=repr(self._gpu_config),
            mounts=repr(self._mounts),
            network_file_systems=repr(self._network_file_systems),
        )
        return f"{inspect.getsource(self._raw_f)}\n{repr(kwargs)}"


Function = synchronize_api(_Function)


class _FunctionCall(_Handle, type_prefix="fc"):
    """A reference to an executed function call.

    Constructed using `.spawn(...)` on a Modal function with the same
    arguments that a function normally takes. Acts as a reference to
    an ongoing function call that can be passed around and used to
    poll or fetch function results at some later time.

    Conceptually similar to a Future/Promise/AsyncResult in other contexts and languages.
    """

    def _invocation(self):
        assert self._client
        return _Invocation(self._client.stub, self.object_id, self._client)

    async def get(self, timeout: Optional[float] = None):
        """Gets the result of the function call

        Raises `TimeoutError` if no results are returned within `timeout` seconds.
        Setting `timeout` to None (the default) waits indefinitely until there is a result
        """
        return await self._invocation().poll_function(timeout=timeout)

    async def get_call_graph(self) -> List[InputInfo]:
        """Returns a structure representing the call graph from a given root
        call ID, along with the status of execution for each node.

        See [`modal.call_graph`](/docs/reference/modal.call_graph) reference page
        for documentation on the structure of the returned `InputInfo` items.
        """
        assert self._client and self._client.stub
        request = api_pb2.FunctionGetCallGraphRequest(function_call_id=self.object_id)
        response = await retry_transient_errors(self._client.stub.FunctionGetCallGraph, request)
        return _reconstruct_call_graph(response)

    async def cancel(self):
        request = api_pb2.FunctionCallCancelRequest(function_call_id=self.object_id)
        assert self._client and self._client.stub
        await self._client.stub.FunctionCallCancel(request)


FunctionCall = synchronize_api(_FunctionCall)


async def _gather(*function_calls: _FunctionCall):
    """Wait until all Modal function calls have results before returning

    Accepts a variable number of FunctionCall objects as returned by `Function.spawn()`.

    Returns a list of results from each function call, or raises an exception
    of the first failing function call.

    E.g.

    ```python notest
    function_call_1 = slow_func_1.spawn()
    function_call_2 = slow_func_2.spawn()

    result_1, result_2 = gather(function_call_1, function_call_2)
    ```
    """
    try:
        return await asyncio.gather(*[fc.get() for fc in function_calls])
    except Exception as exc:
        # TODO: kill all running function calls
        raise exc


gather = synchronize_api(_gather)


_current_input_id: Optional[str] = None


def current_input_id() -> str:
    """Returns the input ID for the currently processed input.

    Can only be called from Modal function (i.e. in a container context).

    ```python
    from modal import current_input_id

    @stub.function()
    def process_stuff():
        print(f"Starting to process {current_input_id()}")
    ```
    """
    global _current_input_id
    return _current_input_id


def _set_current_input_id(input_id: Optional[str]):
    global _current_input_id
    _current_input_id = input_id


class _PartialFunction:
    """Intermediate function, produced by @method or @web_endpoint"""

    @staticmethod
    def initialize_cls(user_cls: type, function_handles: Dict[str, _FunctionHandle]):
        user_cls._modal_function_handles = function_handles

    def __init__(
        self,
        raw_f: Callable[..., Any],
        webhook_config: Optional[api_pb2.WebhookConfig] = None,
        is_generator: Optional[bool] = None,
    ):
        self.raw_f = raw_f
        self.webhook_config = webhook_config
        self.is_generator = is_generator
        self.wrapped = False  # Make sure that this was converted into a FunctionHandle

    def __get__(self, obj, objtype=None) -> _FunctionHandle:
        k = self.raw_f.__name__
        if obj:  # Cls().fun
            function_handle = obj._modal_function_handles[k]
        else:  # Cls.fun
            function_handle = objtype._modal_function_handles[k]
        return function_handle.bind_obj(obj, objtype)

    def __del__(self):
        if self.wrapped is False:
            logger.warning(
                f"Method or web function {self.raw_f} was never turned into a function."
                " Did you forget a @stub.function or @stub.cls decorator?"
            )


PartialFunction = synchronize_api(_PartialFunction)


def _method(
    *,
    # Set this to True if it's a non-generator function returning
    # a [sync/async] generator object
    is_generator: Optional[bool] = None,
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Decorator for methods that should be transformed into a Modal Function registered against this class's stub.

    **Usage:**

    ```python
    @stub.cls(cpu=8)
    class MyCls:

        @modal.method()
        def f(self):
            ...
    ```
    """

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        return _PartialFunction(raw_f, is_generator=is_generator)

    return wrapper


@typechecked
def _web_endpoint(
    method: str = "GET",  # REST method for the created endpoint.
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Register a basic web endpoint with this application.

    This is the simple way to create a web endpoint on Modal. The function
    behaves as a [FastAPI](https://fastapi.tiangolo.com/) handler and should
    return a response object to the caller.

    Endpoints created with `@stub.web_endpoint` are meant to be simple, single
    request handlers and automatically have
    [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) enabled.
    For more flexibility, use `@stub.asgi_app`.

    To learn how to use Modal with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).

    All webhook requests have a 150s maximum request time for the HTTP request itself. However, the underlying functions can
    run for longer and return results to the caller on completion.

    The two `wait_for_response` modes for webhooks are as follows:
    * `wait_for_response=True` - tries to fulfill the request on the original URL, but returns a 302 redirect after ~150s to a result URL (original URL with an added `__modal_function_id=...` query parameter)
    * `wait_for_response=False` - immediately returns a 202 ACCEPTED response with a JSON payload: `{"result_url": "..."}` containing the result "redirect" URL from above (which in turn redirects to itself every ~150s)
    """
    if not isinstance(method, str):
        raise InvalidError(
            f"Unexpected argument {method} of type {type(method)} for `method` parameter. "
            "Add empty parens to the decorator, e.g. @web_endpoint() if there are no arguments. "
            "Otherwise, pass an argument of type `str`: @web_endpoint(method='POST')"
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _FunctionHandle):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@stub.function()\n@stub.web_endpoint()\ndef my_webhook():\n    ..."
            )
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        # self._loose_webhook_configs.add(raw_f)

        return _PartialFunction(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_FUNCTION,
                method=method,
                requested_suffix=label,
                async_mode=_response_mode,
            ),
        )

    return wrapper


@typechecked
def _asgi_app(
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Register an ASGI app with this application.

    Asynchronous Server Gateway Interface (ASGI) is a standard for Python
    synchronous and asynchronous apps, supported by all popular Python web
    libraries. This is an advanced decorator that gives full flexibility in
    defining one or more web endpoints on Modal.

    To learn how to use Modal with popular web frameworks, see the
    [guide on web endpoints](https://modal.com/docs/guide/webhooks).

    The two `wait_for_response` modes for webhooks are as follows:
    * wait_for_response=True - tries to fulfill the request on the original URL, but returns a 302 redirect after ~150s to a result URL (original URL with an added `__modal_function_id=fc-1234abcd` query parameter)
    * wait_for_response=False - immediately returns a 202 ACCEPTED response with a JSON payload: `{"result_url": "..."}` containing the result "redirect" url from above (which in turn redirects to itself every 150s)
    """
    if label and not isinstance(label, str):
        raise InvalidError(
            f"Unexpected argument {label} of type {type(label)} for `label` parameter. "
            "Add empty parens to the decorator, e.g. @asgi_app() if there are no arguments. "
            "Otherwise, pass an argument of type `str`: @asgi_app(label='mylabel')"
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        # self._loose_webhook_configs.add(raw_f)

        return _PartialFunction(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_ASGI_APP,
                requested_suffix=label,
                async_mode=_response_mode,
            ),
        )

    return wrapper


@typechecked
def _wsgi_app(
    label: Optional[str] = None,  # Label for created endpoint. Final subdomain will be <workspace>--<label>.modal.run.
    wait_for_response: bool = True,  # Whether requests should wait for and return the function response.
) -> Callable[[Callable[..., Any]], _PartialFunction]:
    """Register a WSGI app with this application.

    See documentation for [`asgi_app`](/docs/reference/modal.asgi_app).
    """
    if label and not isinstance(label, str):
        raise InvalidError(
            f"Unexpected argument {label} of type {type(label)} for `label` parameter. "
            "Add empty parens to the decorator, e.g. @wsgi_app() if there are no arguments. "
            "Otherwise, pass an argument of type `str`: @wsgi_app(label='mylabel')"
        )

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if not wait_for_response:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_TRIGGER
        else:
            _response_mode = api_pb2.WEBHOOK_ASYNC_MODE_AUTO  # the default

        # self._loose_webhook_configs.add(raw_f)

        return _PartialFunction(
            raw_f,
            api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_WSGI_APP,
                requested_suffix=label,
                async_mode=_response_mode,
            ),
        )

    return wrapper


method = synchronize_api(_method)
web_endpoint = synchronize_api(_web_endpoint)
asgi_app = synchronize_api(_asgi_app)
wsgi_app = synchronize_api(_wsgi_app)
