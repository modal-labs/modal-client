# Copyright Modal Labs 2025
import pytest

from modal._utils.async_utils import TimestampPriorityQueue
from modal.parallel_map import _MapItemContext, _MapItemState, _OutputType
from modal.retries import RetryManager
from modal_proto import api_pb2
from test.supports.map_item_test_utils import (
    InputJwtData,
    assert_context_is,
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


@pytest.mark.asyncio
async def test_ctx_initial_state():
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
    output_type = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert output_type == _OutputType.SUCCESSFUL_COMPLETION
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)


@pytest.mark.asyncio
async def test_failed_output_zero_retries(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=0)
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy), is_input_plane_instance=True)
    input_jwt_data = InputJwtData.of(0, 0)
    ctx.handle_map_start_or_continue_response(input_jwt_data.to_jwt())
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, None, input_jwt_data, input_data.args)

    # Get outputs
    output_type = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert output_type == _OutputType.FAILED_COMPLETION
    assert_context_is(ctx, _MapItemState.COMPLETE, 1, None, input_jwt_data, input_data.args)
    assert retry_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [result_failure, result_internal_failure])
async def test_control_plane_failed_output_not_retried(retry_queue, result):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=3)
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    output_type = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert output_type == _OutputType.FAILED_COMPLETION
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)
    assert retry_queue.empty()


@pytest.mark.asyncio
async def test_get_successful_output_before_put_inputs_completes(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)

    # Get outputs
    output_type = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert output_type == _OutputType.SUCCESSFUL_COMPLETION
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, None, None, input_data.args)
    assert retry_queue.empty()

    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)


@pytest.mark.asyncio
async def test_ignore_stale_failed_output(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy), is_input_plane_instance=True)
    input_jwt_data = InputJwtData.of(0, 0)
    ctx.handle_map_start_or_continue_response(input_jwt_data.to_jwt())
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, None, input_jwt_data, input_data.args)

    # Get outputs
    output_type = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert output_type == _OutputType.RETRYING
    # The retry count is now incremented to 1
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, None, input_jwt_data, input_data.args)

    # Get outputs
    # We get a duplicate output
    output_type = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert output_type == _OutputType.STALE_RETRY_DUPLICATE
    # The output should have been ignored because it has retry count 0, but the ctx is on retry count 1.
    # Assert that state has not changed since.
    assert_context_is(ctx, _MapItemState.WAITING_TO_RETRY, 1, None, input_jwt_data, input_data.args)


@pytest.mark.asyncio
async def test_ignore_duplicate_successful_output(retry_queue):
    ctx = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    input_jwt_data = InputJwtData.of(0, 0)
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt=input_jwt_data.to_jwt())
    ctx.handle_put_inputs_response(response_item)
    assert_context_is(ctx, _MapItemState.WAITING_FOR_OUTPUT, 0, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    output_type = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert output_type == _OutputType.SUCCESSFUL_COMPLETION
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)

    # Get outputs
    # We get a duplicate output
    output_type = await ctx.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success, retry_count=0),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    # Output type should be duplicate since it was already complete.
    assert output_type == _OutputType.ALREADY_COMPLETE_DUPLICATE
    # The output should have been ignored because it is already complete.
    # Assert that state has not changed since.
    assert_context_is(ctx, _MapItemState.COMPLETE, 0, "in-0", input_jwt_data, input_data.args)
