# Copyright Modal Labs 2024
import asyncio
import time
import typing
from dataclasses import dataclass
from typing import Any, Callable, Optional

from grpclib import Status

from modal._runtime.execution_context import current_input_id
from modal._utils.async_utils import (
    AsyncOrSyncIterable,
    aclosing,
    async_map_ordered,
    async_merge,
    async_zip,
    queue_batch_iterator,
    sync_or_async_iter,
    synchronize_api,
    synchronizer,
    warn_if_generator_is_not_consumed,
)
from modal._utils.blob_utils import BLOB_MAX_PARALLELISM
from modal._utils.function_utils import (
    ATTEMPT_TIMEOUT_GRACE_PERIOD,
    OUTPUTS_TIMEOUT,
    _create_input,
    _process_result,
)
from modal._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, RetryWarningMessage, retry_transient_errors
from modal.config import logger
from modal_proto import api_pb2

if typing.TYPE_CHECKING:
    import modal.client

# pump_inputs should retry if it receives any of the standard retryable codes plus RESOURCE_EXHAUSTED.
PUMP_INPUTS_RETRYABLE_GRPC_STATUS_CODES = RETRYABLE_GRPC_STATUS_CODES + [Status.RESOURCE_EXHAUSTED]
PUMP_INPUTS_MAX_RETRIES = 8
PUMP_INPUTS_MAX_RETRY_DELAY=15

class _SynchronizedQueue:
    """mdmd:hidden"""

    # small wrapper around asyncio.Queue to make it cross-thread compatible through synchronicity
    async def init(self):
        # in Python 3.8 the asyncio.Queue is bound to the event loop on creation
        # so it needs to be created in a synchronicity-wrapped init method
        self.q = asyncio.Queue()

    @synchronizer.no_io_translation
    async def put(self, item):
        await self.q.put(item)

    @synchronizer.no_io_translation
    async def get(self):
        return await self.q.get()


SynchronizedQueue = synchronize_api(_SynchronizedQueue)


@dataclass
class _OutputValue:
    # box class for distinguishing None results from non-existing/None markers
    value: Any


MAP_INVOCATION_CHUNK_SIZE = 49

if typing.TYPE_CHECKING:
    import modal.functions


async def _map_invocation(
    function: "modal.functions._Function",
    raw_input_queue: _SynchronizedQueue,
    client: "modal.client._Client",
    order_outputs: bool,
    return_exceptions: bool,
    count_update_callback: Optional[Callable[[int, int], None]],
):
    assert client.stub
    request = api_pb2.FunctionMapRequest(
        function_id=function.object_id,
        parent_input_id=current_input_id() or "",
        function_call_type=api_pb2.FUNCTION_CALL_TYPE_MAP,
        return_exceptions=return_exceptions,
    )
    response = await retry_transient_errors(client.stub.FunctionMap, request)

    function_call_id = response.function_call_id

    have_all_inputs = False
    num_inputs = 0
    num_outputs = 0

    def count_update():
        if count_update_callback is not None:
            count_update_callback(num_outputs, num_inputs)

    pending_outputs: dict[str, int] = {}  # Map input_id -> next expected gen_index value
    completed_outputs: set[str] = set()  # Set of input_ids whose outputs are complete (expecting no more values)

    input_queue: asyncio.Queue = asyncio.Queue()

    async def create_input(argskwargs):
        nonlocal num_inputs
        idx = num_inputs
        num_inputs += 1
        (args, kwargs) = argskwargs
        return await _create_input(args, kwargs, client, idx=idx, method_name=function._use_method_name)

    async def input_iter():
        while 1:
            raw_input = await raw_input_queue.get()
            if raw_input is None:  # end of input sentinel
                break
            yield raw_input  # args, kwargs

    async def drain_input_generator():
        # Parallelize uploading blobs
        async with aclosing(
            async_map_ordered(input_iter(), create_input, concurrency=BLOB_MAX_PARALLELISM)
        ) as streamer:
            async for item in streamer:
                await input_queue.put(item)

        # close queue iterator
        await input_queue.put(None)
        yield

    async def pump_inputs():
        assert client.stub
        nonlocal have_all_inputs, num_inputs
        async for items in queue_batch_iterator(input_queue, MAP_INVOCATION_CHUNK_SIZE):
            request = api_pb2.FunctionPutInputsRequest(
                function_id=function.object_id, inputs=items, function_call_id=function_call_id
            )
            logger.debug(
                f"Pushing {len(items)} inputs to server. Num queued inputs awaiting push is {input_queue.qsize()}."
            )
            # with 8 retries we log the warning below about every 30 seconds which isn't too spammy.
            retry_warning_message = RetryWarningMessage(
                message=f"Warning: map progress for function {function._function_name} is limited."
                " Common bottlenecks include slow iteration over results, or function backlogs.",
                warning_interval=8,
                errors_to_warn_for=[Status.RESOURCE_EXHAUSTED])
            resp = await retry_transient_errors(
                client.stub.FunctionPutInputs,
                request,
                max_retries=None,
                max_delay=PUMP_INPUTS_MAX_RETRY_DELAY,
                additional_status_codes=[Status.RESOURCE_EXHAUSTED],
                retry_warning_message=retry_warning_message)
            count_update()
            for item in resp.inputs:
                pending_outputs.setdefault(item.input_id, 0)
            logger.debug(
                f"Successfully pushed {len(items)} inputs to server. "
                f"Num queued inputs awaiting push is {input_queue.qsize()}."
            )

        have_all_inputs = True
        yield

    async def get_all_outputs():
        assert client.stub
        nonlocal num_inputs, num_outputs, have_all_inputs
        last_entry_id = "0-0"
        while not have_all_inputs or len(pending_outputs) > len(completed_outputs):
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=function_call_id,
                timeout=OUTPUTS_TIMEOUT,
                last_entry_id=last_entry_id,
                clear_on_success=False,
                requested_at=time.time(),
            )
            response = await retry_transient_errors(
                client.stub.FunctionGetOutputs,
                request,
                max_retries=20,
                attempt_timeout=OUTPUTS_TIMEOUT + ATTEMPT_TIMEOUT_GRACE_PERIOD,
            )

            if len(response.outputs) == 0:
                continue

            last_entry_id = response.last_entry_id
            for item in response.outputs:
                pending_outputs.setdefault(item.input_id, 0)
                if item.input_id in completed_outputs:
                    # If this input is already completed, it means the output has already been
                    # processed and was received again due to a duplicate.
                    continue
                completed_outputs.add(item.input_id)
                num_outputs += 1
                yield item

    async def get_all_outputs_and_clean_up():
        assert client.stub
        try:
            async with aclosing(get_all_outputs()) as output_items:
                async for item in output_items:
                    yield item
        finally:
            # "ack" that we have all outputs we are interested in and let backend clear results
            request = api_pb2.FunctionGetOutputsRequest(
                function_call_id=function_call_id,
                timeout=0,
                last_entry_id="0-0",
                clear_on_success=True,
                requested_at=time.time(),
            )
            await retry_transient_errors(client.stub.FunctionGetOutputs, request)

    async def fetch_output(item: api_pb2.FunctionGetOutputsItem) -> tuple[int, Any]:
        try:
            output = await _process_result(item.result, item.data_format, client.stub, client)
        except Exception as e:
            if return_exceptions:
                output = e
            else:
                raise e
        return (item.idx, output)

    async def poll_outputs():
        # map to store out-of-order outputs received
        received_outputs = {}
        output_idx = 0

        async with aclosing(
            async_map_ordered(get_all_outputs_and_clean_up(), fetch_output, concurrency=BLOB_MAX_PARALLELISM)
        ) as streamer:
            async for idx, output in streamer:
                count_update()
                if not order_outputs:
                    yield _OutputValue(output)
                else:
                    # hold on to outputs for function maps, so we can reorder them correctly.
                    received_outputs[idx] = output
                    while output_idx in received_outputs:
                        output = received_outputs.pop(output_idx)
                        yield _OutputValue(output)
                        output_idx += 1

        assert len(received_outputs) == 0

    async with aclosing(async_merge(drain_input_generator(), pump_inputs(), poll_outputs())) as streamer:
        async for response in streamer:
            if response is not None:
                yield response.value


@warn_if_generator_is_not_consumed(function_name="Function.map")
def _map_sync(
    self,
    *input_iterators: typing.Iterable[Any],  # one input iterator per argument in the mapped-over function/generator
    kwargs={},  # any extra keyword arguments for the function
    order_outputs: bool = True,  # return outputs in order
    return_exceptions: bool = False,  # propagate exceptions (False) or aggregate them in the results list (True)
) -> AsyncOrSyncIterable:
    """Parallel map over a set of inputs.

    Takes one iterator argument per argument in the function being mapped over.

    Example:
    ```python
    @app.function()
    def my_func(a):
        return a ** 2


    @app.local_entrypoint()
    def main():
        assert list(my_func.map([1, 2, 3, 4])) == [1, 4, 9, 16]
    ```

    If applied to a `stub.function`, `map()` returns one result per input and the output order
    is guaranteed to be the same as the input order. Set `order_outputs=False` to return results
    in the order that they are completed instead.

    `return_exceptions` can be used to treat exceptions as successful results:

    ```python
    @app.function()
    def my_func(a):
        if a == 2:
            raise Exception("ohno")
        return a ** 2


    @app.local_entrypoint()
    def main():
        # [0, 1, UserCodeException(Exception('ohno'))]
        print(list(my_func.map(range(3), return_exceptions=True)))
    ```
    """

    return AsyncOrSyncIterable(
        _map_async(
            self, *input_iterators, kwargs=kwargs, order_outputs=order_outputs, return_exceptions=return_exceptions
        ),
        nested_async_message=(
            "You can't iter(Function.map()) or Function.for_each() from an async function. "
            "Use async for ... Function.map.aio() or Function.for_each.aio() instead."
        ),
    )


@warn_if_generator_is_not_consumed(function_name="Function.map.aio")
async def _map_async(
    self: "modal.functions.Function",
    *input_iterators: typing.Union[
        typing.Iterable[Any], typing.AsyncIterable[Any]
    ],  # one input iterator per argument in the mapped-over function/generator
    kwargs={},  # any extra keyword arguments for the function
    order_outputs: bool = True,  # return outputs in order
    return_exceptions: bool = False,  # propagate exceptions (False) or aggregate them in the results list (True)
) -> typing.AsyncGenerator[Any, None]:
    """mdmd:hidden
    This runs in an event loop on the main thread

    It concurrently feeds new input to the input queue and yields available outputs
    to the caller.
    Note that since the iterator(s) can block, it's a bit opaque how often the event
    loop decides to get a new input vs how often it will emit a new output.
    We could make this explicit as an improvement or even let users decide what they
    prefer: throughput (prioritize queueing inputs) or latency (prioritize yielding results)
    """
    raw_input_queue: Any = SynchronizedQueue()  # type: ignore
    raw_input_queue.init()

    async def feed_queue():
        # This runs in a main thread event loop, so it doesn't block the synchronizer loop
        async with aclosing(async_zip(*[sync_or_async_iter(it) for it in input_iterators])) as streamer:
            async for args in streamer:
                await raw_input_queue.put.aio((args, kwargs))
        await raw_input_queue.put.aio(None)  # end-of-input sentinel
        if False:
            # make this a never yielding generator so we can async_merge it below
            # this is important so any exception raised in feed_queue will be propagated
            yield

    # note that `map()` and `map.aio()` are not synchronicity-wrapped, since
    # they accept executable code in the form of
    # iterators that we don't want to run inside the synchronicity thread.
    # Instead, we delegate to `._map()` with a safer Queue as input
    async with aclosing(
        async_merge(self._map.aio(raw_input_queue, order_outputs, return_exceptions), feed_queue())
    ) as map_output_stream:
        async for output in map_output_stream:
            yield output


def _for_each_sync(self, *input_iterators, kwargs={}, ignore_exceptions: bool = False):
    """Execute function for all inputs, ignoring outputs.

    Convenient alias for `.map()` in cases where the function just needs to be called.
    as the caller doesn't have to consume the generator to process the inputs.
    """
    # TODO(erikbern): it would be better if this is more like a map_spawn that immediately exits
    # rather than iterating over the result
    for _ in self.map(*input_iterators, kwargs=kwargs, order_outputs=False, return_exceptions=ignore_exceptions):
        pass


async def _for_each_async(self, *input_iterators, kwargs={}, ignore_exceptions: bool = False):
    async for _ in self.map.aio(  # type: ignore
        *input_iterators, kwargs=kwargs, order_outputs=False, return_exceptions=ignore_exceptions
    ):
        pass


@warn_if_generator_is_not_consumed(function_name="Function.starmap")
async def _starmap_async(
    self,
    input_iterator: typing.Union[typing.Iterable[typing.Sequence[Any]], typing.AsyncIterable[typing.Sequence[Any]]],
    kwargs={},
    order_outputs: bool = True,
    return_exceptions: bool = False,
) -> typing.AsyncIterable[Any]:
    raw_input_queue: Any = SynchronizedQueue()  # type: ignore
    raw_input_queue.init()

    async def feed_queue():
        # This runs in a main thread event loop, so it doesn't block the synchronizer loop
        async with aclosing(sync_or_async_iter(input_iterator)) as streamer:
            async for args in streamer:
                await raw_input_queue.put.aio((args, kwargs))
        await raw_input_queue.put.aio(None)  # end-of-input sentinel

    feed_input_task = asyncio.create_task(feed_queue())
    try:
        async for output in self._map.aio(raw_input_queue, order_outputs, return_exceptions):  # type: ignore[reportFunctionMemberAccess]
            yield output
    finally:
        feed_input_task.cancel()  # should only be needed in case of exceptions


@warn_if_generator_is_not_consumed(function_name="Function.starmap.aio")
def _starmap_sync(
    self,
    input_iterator: typing.Iterable[typing.Sequence[Any]],
    kwargs={},
    order_outputs: bool = True,
    return_exceptions: bool = False,
) -> AsyncOrSyncIterable:
    """Like `map`, but spreads arguments over multiple function arguments.

    Assumes every input is a sequence (e.g. a tuple).

    Example:
    ```python
    @app.function()
    def my_func(a, b):
        return a + b


    @app.local_entrypoint()
    def main():
        assert list(my_func.starmap([(1, 2), (3, 4)])) == [3, 7]
    ```
    """
    return AsyncOrSyncIterable(
        _starmap_async(
            self, input_iterator, kwargs=kwargs, order_outputs=order_outputs, return_exceptions=return_exceptions
        ),
        nested_async_message=(
            "You can't run Function.map() or Function.for_each() from an async function. "
            "Use Function.map.aio()/Function.for_each.aio() instead."
        ),
    )
