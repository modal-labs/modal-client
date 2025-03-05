# Copyright Modal Labs 2025
import asyncio
import pytest

from modal._utils.async_utils import TimestampPriorityQueue
from modal.parallel_map import _MapItemContext, _MapItemState
from modal.retries import RetryManager
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

now_seconds = 1738439812
input_data = api_pb2.FunctionInput(args=b"test")


@pytest.fixture
def retry_queue():
    return TimestampPriorityQueue()


def test_ctx_initial_state():
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    assert_context_is(ctx, _MapItemState.SENDING, 0, None, None, input_data.args)


@pytest.mark.asyncio
async def test_successful_output(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == True
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)


@pytest.mark.asyncio
async def test_failed_output_no_retries(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=0)
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == True
    assert_context_is(ctx, _MapItemState.COMPLETE, 1, "in-0", input_jwt_data, input_data.args)
    assert retry_queue.empty()


@pytest.mark.asyncio
async def test_failed_output_retries_then_succeeds(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=2)
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data_0 = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data_0.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data_0, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, "in-0", input_jwt_data_0, input_data.args)
    assert len(retry_queue) == 1

    # Retry input
    retry_item = await ctx.prepare_item_for_retry()
    assert_retry_item_is(retry_item, input_jwt_data_0, 1, input_data.args)
    assert_context_is(ctx, _MapItemState.RETRYING, 1, "in-0", None, input_data.args)
    await retry_queue.clear()

    input_jwt_data_1 = InputJwtData.of(0, 1)
    ctx.handle_retry_response(input_jwt_data_1.to_jwt())
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 1, "in-0", input_jwt_data_1, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=1),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 2, "in-0", input_jwt_data_1, input_data.args)
    assert len(retry_queue) == 1

    # Retry input
    retry_item = await ctx.prepare_item_for_retry()
    assert_retry_item_is(retry_item, input_jwt_data_1, 2, input_data.args)
    assert_context_is(ctx, _MapItemState.RETRYING, 2, "in-0", None, input_data.args)
    await retry_queue.clear()

    input_jwt_data_2 = InputJwtData.of(0, 2)
    ctx.handle_retry_response(input_jwt_data_2.to_jwt())
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 2, "in-0", input_jwt_data_2, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success, retry_count=2),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == True
    assert_context_is(ctx, _MapItemState.COMPLETE, 2, "in-0", input_jwt_data_2, input_data.args)


@pytest.mark.asyncio
async def test_lost_input_retries_then_succeeds(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=1)
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data_0 = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data_0.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data_0, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_internal_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, "in-0", input_jwt_data_0, input_data.args)
    assert len(retry_queue) == 1

    # Retry input
    retry_item = await ctx.prepare_item_for_retry()
    assert_retry_item_is(retry_item, input_jwt_data_0, 1, input_data.args)
    assert_context_is(ctx, _MapItemState.RETRYING, 1, "in-0", None, input_data.args)

    input_jwt_data_1 = InputJwtData.of(0, 1)
    ctx.handle_retry_response(input_jwt_data_1.to_jwt())
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 1, "in-0", input_jwt_data_1, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success, retry_count=1),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == True
    assert_context_is(ctx, _MapItemState.COMPLETE, 1, "in-0", input_jwt_data_1, input_data.args)


@pytest.mark.asyncio
async def test_failed_output_exhausts_retries(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=1)
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data_0 = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data_0.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data_0, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, "in-0", input_jwt_data_0, input_data.args)
    assert len(retry_queue) == 1

    # Retry input
    retry_item = await ctx.prepare_item_for_retry()
    assert_retry_item_is(retry_item, input_jwt_data_0, 1, input_data.args)
    assert_context_is(ctx, _MapItemState.RETRYING, 1, "in-0", None, input_data.args)

    input_jwt_data_1 = InputJwtData.of(0, 1)
    ctx.handle_retry_response(input_jwt_data_1.to_jwt())
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 1, "in-0", input_jwt_data_1, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=1),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == True
    assert_context_is(ctx, _MapItemState.COMPLETE, 2, "in-0", input_jwt_data_1, input_data.args)


@pytest.mark.asyncio
async def test_get_successful_output_before_put_inputs_completes(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == True
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, None, None, input_data.args)
    assert retry_queue.empty()

    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)


@pytest.mark.asyncio
async def test_get_failed_output_before_put_inputs_completes(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, None, None, input_data.args)
    assert len(retry_queue) == 1

    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, "in-0", input_jwt_data, input_data.args)
    assert len(retry_queue) == 1

@pytest.mark.asyncio
async def test_retry_failed_output_before_put_inputs_completes(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, None, None, input_data.args)
    assert len(retry_queue) == 1

    # Retry input
    # prepare_item_for_retry should block waiting for put_inputs to set the input_jwt. Create a task for it,
    # so it can be in a blocked state, while we continue doing other things.
    task = asyncio.create_task(ctx.prepare_item_for_retry())
    # Sleep to allow prepare_item_for_retry to start and block on input_jwt.
    await asyncio.sleep(0.1)
    # Assert that the task is still blocked waiting for input_jwt.
    assert not task.done()
    assert_context_is(ctx, _MapItemState.RETRYING, 1, None, None, input_data.args)

    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.RETRYING, 1, "in-0", input_jwt_data, input_data.args)
    assert len(retry_queue) == 1

    # Retry input
    # The task should now be able to complete, since input_jwt has been set.
    retry_item = await task
    assert_retry_item_is(retry_item, input_jwt_data, 1, input_data.args)
    assert_context_is(ctx, _MapItemState.RETRYING, 1, "in-0", None, input_data.args)
    await retry_queue.clear()

    input_jwt_data_1 = InputJwtData.of(0, 1)
    ctx.handle_retry_response(input_jwt_data_1.to_jwt())
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 1, "in-0", input_jwt_data_1, input_data.args)

@pytest.mark.asyncio
async def test_ignore_duplicate_failed_output(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == False
    # The retry count is now incremented to 1
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    # We get a duplicate output
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == False
    # The output should have been ignored because it has retry count 0, but the ctx is on retry count 1.
    # Assert that state has not changed since.
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, "in-0", input_jwt_data, input_data.args)

@pytest.mark.asyncio
async def test_ignore_duplicate_successful_output(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == True
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    # We get a duplicate output
    changed_to_complete = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    # Changed to complete should be false since it was already complete - there was no change.
    assert changed_to_complete == False
    # The output should have been ignored because it is already complete.
    # Assert that state has not changed since.
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)
