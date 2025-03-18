# Copyright Modal Labs 2025
import pytest

from modal._utils.async_utils import TimestampPriorityQueue
from modal.parallel_map import _MapItemsManager, _MapItemState
from modal_proto import api_pb2
from test.supports.map_item_test_utils import (
    InputJwtData,
    assert_context_is,
    assert_retry_item_is,
    result_failure,
    result_internal_failure,
    result_success,
)

retry_policy = api_pb2.FunctionRetryPolicy(
    backoff_coefficient=1.0,
    initial_delay_ms=500,
    max_delay_ms=500,
    retries=2,
)
retry_queue: TimestampPriorityQueue
manager: _MapItemsManager
now_seconds = 1738439812
count = 10


@pytest.fixture(autouse=True)
def reset_state():
    global retry_queue, manager
    retry_queue = TimestampPriorityQueue()
    manager = _MapItemsManager(
        retry_policy=retry_policy,
        function_call_invocation_type=api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue=retry_queue,
        sync_client_retries_enabled=True
    )


async def add_items():
    put_items = [
        api_pb2.FunctionPutInputsItem(idx=i, input=api_pb2.FunctionInput(args=f"{i}".encode())) for i in range(count)
    ]
    await manager.add_items(put_items)
    assert len(manager) == count
    for i in range(count):
        ctx = manager.get_item_context(i)
        assert_context_is(ctx, _MapItemState.SENDING, 0, None, None, f"{i}".encode())


async def handle_put_inputs_response(state: _MapItemState):
    response_items = [
        api_pb2.FunctionPutInputsResponseItem(idx=i, input_id=f"in-{i}", input_jwt=InputJwtData.of(i, 0).to_jwt())
        for i in range(count)
    ]
    manager.handle_put_inputs_response(response_items)
    if state == _MapItemState.COMPLETE:
        assert len(manager) == 0
    else:
        for i in range(count):
            ctx = manager.get_item_context(i)
            assert_context_is(ctx, state, 0, f"in-{i}", InputJwtData.of(i, 0), f"{i}".encode())


def get_input_jwts_waiting_for_output(retry_count: int):
    assert [InputJwtData.from_jwt(input_jwt) for input_jwt in manager.get_input_jwts_waiting_for_output()] == [
        InputJwtData.of(i, retry_count) for i in range(count)
    ]


async def handle_get_outputs_response(
    result: api_pb2.GenericResult,
    state: _MapItemState,
    retry_count: int,
    output_is_complete: bool,
    include_input_jwt: bool = True,
):
    for i in range(count):
        _output_is_complete = await manager.handle_get_outputs_response(
            api_pb2.FunctionGetOutputsItem(idx=i, result=result, retry_count=retry_count), now_seconds
        )
        assert _output_is_complete == output_is_complete
        ctx = manager.get_item_context(i)
        if state == _MapItemState.COMPLETE:
            assert ctx is None
        else:
            input_jwt = InputJwtData.of(i, retry_count) if include_input_jwt else None
            # we add 1 to the retry count because it gets incremented during handling of the response
            assert_context_is(ctx, state, retry_count + 1, f"in-{i}", input_jwt, f"{i}".encode())
    if state == _MapItemState.COMPLETE:
        assert len(manager) == 0
    else:
        assert len(manager) == count
    if state == _MapItemState.WAITING_TO_RETRY:
        assert len(retry_queue) == count
    else:
        assert len(retry_queue) == 0


async def prepare_items_for_retry(retry_count: int):
    retry_items: list[api_pb2.FunctionRetryInputsItem] = await manager.prepare_items_for_retry(
        [i for i in range(count)]
    )
    for i in range(count):
        assert_retry_item_is(retry_items[i], InputJwtData.of(i, retry_count - 1), retry_count, f"{i}".encode())


def handle_retry_response(retry_count: int):
    response_items = [InputJwtData.of(i, retry_count).to_jwt() for i in range(count)]
    manager.handle_retry_response(response_items)
    for i in range(count):
        ctx = manager.get_item_context(i)
        assert_context_is(
            ctx,
            _MapItemState.WAITING_FOR_OUTPUT,
            retry_count,
            f"in-{i}",
            InputJwtData.of(i, retry_count),
            f"{i}".encode(),
        )


@pytest.mark.asyncio
async def test_happy_path():
    # pump_inputs - retry count 0
    await add_items()
    await handle_put_inputs_response(_MapItemState.WAITING_FOR_OUTPUT)
    # get_all_outputs
    get_input_jwts_waiting_for_output(0)
    await handle_get_outputs_response(result_success, _MapItemState.COMPLETE, 0, True)


@pytest.mark.asyncio
async def test_retry():
    # pump_inputs - retry count 0
    await add_items()
    await handle_put_inputs_response(_MapItemState.WAITING_FOR_OUTPUT)

    # get_all_outputs - retry count 0
    get_input_jwts_waiting_for_output(0)
    await handle_get_outputs_response(result_failure, _MapItemState.WAITING_TO_RETRY, 0, False)

    # retry_inputs - retry count 1
    await prepare_items_for_retry(1)
    await retry_queue.clear()
    handle_retry_response(1)

    # get_all_outputs - retry count 1
    get_input_jwts_waiting_for_output(1)
    await handle_get_outputs_response(result_success, _MapItemState.COMPLETE, 1, True)


@pytest.mark.asyncio
async def test_retry_lost_input():
    # pump_inputs - retry count 0
    await add_items()
    await handle_put_inputs_response(_MapItemState.WAITING_FOR_OUTPUT)

    # get_all_outputs - retry count 0
    get_input_jwts_waiting_for_output(0)
    await handle_get_outputs_response(result_internal_failure, _MapItemState.WAITING_TO_RETRY, 0, False)

    # retry_inputs - retry count 1
    await prepare_items_for_retry(1)
    await retry_queue.clear()
    handle_retry_response(1)

    # get_all_outputs - retry count 1
    get_input_jwts_waiting_for_output(1)
    await handle_get_outputs_response(result_success, _MapItemState.COMPLETE, 1, True)


@pytest.mark.asyncio
async def test_duplicate_succcesful_outputs():
    # pump_inputs - retry count 0
    await add_items()
    await handle_put_inputs_response(_MapItemState.WAITING_FOR_OUTPUT)

    # get_all_outputs - retry count 0
    get_input_jwts_waiting_for_output(0)
    await handle_get_outputs_response(result_success, _MapItemState.COMPLETE, 0, True)

    # get_all_outputs - retry count 0 (duplicate)
    # No items should be waiting for output since we already processed all the outputs
    assert manager.get_input_jwts_waiting_for_output() == []
    await handle_get_outputs_response(result_success, _MapItemState.COMPLETE, 0, False)


@pytest.mark.asyncio
async def test_duplicate_failed_outputs():
    # pump_inputs - retry count 0
    await add_items()
    await handle_put_inputs_response(_MapItemState.WAITING_FOR_OUTPUT)

    # get_all_outputs - retry_count 0
    get_input_jwts_waiting_for_output(0)
    await handle_get_outputs_response(result_failure, _MapItemState.WAITING_TO_RETRY, 0, False)

    # get_all_outputs - retry_count 0 (duplicate)
    # No items should be waiting for output since we already processed all the outputs
    assert manager.get_input_jwts_waiting_for_output() == []
    await handle_get_outputs_response(result_failure, _MapItemState.WAITING_TO_RETRY, 0, False)


@pytest.mark.asyncio
async def test_get_outputs_completes_before_put_inputs():
    # There is a race condition where we can send inputs to the server with PutInputs, but before it returns,
    # a call to GetOutputs executing in a coroutine fetches the output and completes. Ensure we handle this
    # properly.
    manager = _MapItemsManager(
        retry_policy=retry_policy,
        function_call_invocation_type=api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue=retry_queue,
        sync_client_retries_enabled=True,
    )
    # pump_inputs - retry_count 0 - send request
    await add_items()

    # get_all_outputs - retry_count 0
    # Verify there are no input_jwts waiting for output yet. The input_jwt is returned in the PutInputsResponse,
    # which we have not received yet.
    assert manager.get_input_jwts_waiting_for_output() == []
    await handle_get_outputs_response(result_success, _MapItemState.COMPLETE, 0, True)

    # pump_inputs - retry_count 0 - receive response
    await handle_put_inputs_response(_MapItemState.COMPLETE)


@pytest.mark.asyncio
async def test_get_outputs_completes_before_function_retry():
    # pump_inputs - retry_count 0
    await add_items()
    await handle_put_inputs_response(_MapItemState.WAITING_FOR_OUTPUT)

    # get_all_outputs - retry_count 0
    get_input_jwts_waiting_for_output(0)
    await handle_get_outputs_response(result_failure, _MapItemState.WAITING_TO_RETRY, 0, False)

    # First retry fails

    # retry_inputs - retry_count 1
    await prepare_items_for_retry(1)
    await retry_queue.clear()

    # get_all_outputs - retry_count 1
    # The retry call has not returned yet, so there are not input_jwts waiting for output.
    assert manager.get_input_jwts_waiting_for_output() == []
    await handle_get_outputs_response(result_failure, _MapItemState.WAITING_TO_RETRY, 1, False, False)

    # retry_inputs -  retry_count 1 - handle response
    response_items = [InputJwtData.of(i, 1).to_jwt() for i in range(count)]
    manager.handle_retry_response(response_items)
    for i in range(count):
        # Even though this the response for retry attempt 1, the retry count will be 2 because the above call to
        # handle_get_outputs_response would have bumped the count. The jwt will still be for retry attempt 1.
        assert_context_is(
            manager.get_item_context(i),
            _MapItemState.WAITING_FOR_OUTPUT,
            2,
            f"in-{i}",
            InputJwtData.of(i, 1),
            f"{i}".encode(),
        )

    # Second retry succeeds

    # retry_inputs - retry_count 2
    await prepare_items_for_retry(2)
    await retry_queue.clear()

    # get_all_outputs - retry_count 2
    # The retry call has not returned yet, so there are not input_jwts waiting for output.
    assert manager.get_input_jwts_waiting_for_output() == []
    await handle_get_outputs_response(result_success, _MapItemState.COMPLETE, 2, True)

    # retry_inputs - retry_count 2 - handle response
    response_items = [InputJwtData.of(i, 2).to_jwt() for i in range(count)]
    manager.handle_retry_response(response_items)
    assert len(manager) == 0
