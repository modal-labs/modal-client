# Copyright Modal Labs 2025
import pytest

from modal._utils.async_utils import TimestampPriorityQueue
from modal.parallel_map import _MapItemsManager, _MapItemState
from modal_proto import api_pb2

retry_policy = api_pb2.FunctionRetryPolicy(
    backoff_coefficient=1.0,
    initial_delay_ms=500,
    max_delay_ms=500,
    retries=2,
)

retry_queue: TimestampPriorityQueue

result_success = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
result_failure = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE)
result_internal_failure = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE)
now_seconds = 1738439812


@pytest.fixture(autouse=True)
def reset_retry_queue():
    global retry_queue
    retry_queue = TimestampPriorityQueue()


@pytest.mark.asyncio
async def test_happy_path():
    # Test putting inputs, and getting sucessful outputs. Verify context has proper values throughout.
    count = 10
    manager = _MapItemsManager(
        retry_policy=retry_policy,
        function_call_invocation_type=api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue=retry_queue,
    )
    # Put inputs
    put_items = [api_pb2.FunctionPutInputsItem(idx=i, input=api_pb2.FunctionInput(args=b"{i}")) for i in range(count)]
    await manager.add_items(put_items)
    assert len(manager) == count
    for i in range(count):
        ctx = manager.get_item_context(i)
        assert ctx.state == _MapItemState.SENDING
        assert ctx.input == put_items[i].input
    response_items = [
        api_pb2.FunctionPutInputsResponseItem(idx=i, input_id=f"in-{i}", input_jwt=f"jwt-{i}") for i in range(count)
    ]
    manager.handle_put_inputs_response(response_items)
    for i in range(count):
        ctx = manager.get_item_context(i)
        assert ctx.state == _MapItemState.WAITING_FOR_OUTPUT
        assert await ctx.input_id == response_items[i].input_id
        assert await ctx.input_jwt == response_items[i].input_jwt

    # Get outputs
    assert [await i.input_jwt for i in manager.get_items_waiting_for_output()] == [f"jwt-{i}" for i in range(count)]
    for i in range(count):
        output_is_complete = await manager.handle_get_outputs_response(
            api_pb2.FunctionGetOutputsItem(idx=i, result=result_success), now_seconds
        )
        assert output_is_complete == True
    # Verify handle_get_outputs_response removed the item from the manager
    assert len(manager) == 0


@pytest.mark.asyncio
async def test_retry():
    count = 10
    manager = _MapItemsManager(
        retry_policy=retry_policy,
        function_call_invocation_type=api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue=retry_queue,
    )
    # Put inputs
    put_items = [api_pb2.FunctionPutInputsItem(idx=i, input=api_pb2.FunctionInput(args=b"{i}")) for i in range(count)]
    await manager.add_items(put_items)
    assert len(manager) == count
    for i in range(count):
        ctx = manager.get_item_context(i)
        assert ctx.state == _MapItemState.SENDING
        assert ctx.input == put_items[i].input
    response_items = [
        api_pb2.FunctionPutInputsResponseItem(idx=i, input_id=f"in-{i}", input_jwt=f"jwt-{i}-0") for i in range(count)
    ]
    manager.handle_put_inputs_response(response_items)
    for i in range(count):
        ctx = manager.get_item_context(i)
        assert ctx.state == _MapItemState.WAITING_FOR_OUTPUT
        assert await ctx.input_id == response_items[i].input_id
        assert await ctx.input_jwt == response_items[i].input_jwt

    # Get outputs
    assert [await i.input_jwt for i in manager.get_items_waiting_for_output()] == [f"jwt-{i}-0" for i in range(count)]
    for i in range(count):
        output_is_complete = await manager.handle_get_outputs_response(
            api_pb2.FunctionGetOutputsItem(idx=i, result=result_failure), now_seconds
        )
        assert output_is_complete == False
    # all inputs should still be in the manager, and waiting for retry
    assert len(manager) == count
    for i in range(count):
        assert manager.get_item_context(i).state == _MapItemState.WAITING_TO_RETRY

    # Retry lost input
    retry_items: list[api_pb2.FunctionRetryInputsItem] = manager.get_items_for_retry([i for i in range(count)])
    assert len(retry_items) == count
    for i in range(count):
        assert retry_items[i].input_jwt == f"jwt-{i}-0"
        assert retry_items[i].input == put_items[i].input
        assert retry_items[i].retry_count == 1

    # Update the jwt to something new. It will be different because the redis entry id will have changed.
    response_items = [api_pb2.FunctionRetryInputsResponseItem(idx=i, input_jwt=f"jwt-{i}-1") for i in range(count)]
    manager.handle_retry_response(response_items)
    for i in range(count):
        ctx = manager.get_item_context(i)
        assert ctx.state == _MapItemState.WAITING_FOR_OUTPUT
        assert await ctx.input_id == f"in-{i}"
        # Make sure we have the updated jwt and not the old one
        assert await ctx.input_jwt == f"jwt-{i}-1"

    # Get outputs
    assert [await i.input_jwt for i in manager.get_items_waiting_for_output()] == [f"jwt-{i}-1" for i in range(count)]
    for i in range(count):
        output_is_complete = await manager.handle_get_outputs_response(
            api_pb2.FunctionGetOutputsItem(idx=i, result=result_success), now_seconds
        )
        assert output_is_complete == True

    # handle_output should have removed the item from the manager
    assert len(manager) == 0


@pytest.mark.asyncio
async def test_retry_lost_input():
    manager = _MapItemsManager(
        retry_policy=retry_policy,
        function_call_invocation_type=api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue=retry_queue,
    )
    # Put inputs
    put_item = api_pb2.FunctionPutInputsItem(idx=0, input=api_pb2.FunctionInput(args=b"0"))
    await manager.add_items([put_item])
    manager.handle_put_inputs_response(
        [api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="in-0", input_jwt="jwt-0")]
    )
    # Get output that reports a lost input
    output_is_complete = await manager.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_internal_failure), now_seconds
    )
    assert output_is_complete == False
    # Assert item is waiting to be retried. Because it is lost, it will be retried immediately.
    idx = await retry_queue.get()
    assert idx == 0
    # Retry lost input
    retry_items: list[api_pb2.FunctionRetryInputsItem] = manager.get_items_for_retry([idx])
    assert len(retry_items) == 1
    retry_item = retry_items[0]
    assert retry_item.input_jwt == "jwt-0"
    assert retry_item.input == put_item.input
    assert retry_item.retry_count == 0
    # The response will have a different input_jwt because the redis entry id will have changed
    response_item = api_pb2.FunctionRetryInputsResponseItem(idx=0, input_jwt="jwt-1")
    manager.handle_retry_response([response_item])
    ctx = manager.get_item_context(0)
    assert ctx.state == _MapItemState.WAITING_FOR_OUTPUT
    assert await ctx.input_id == "in-0"
    assert await ctx.input_jwt == "jwt-1"
    # Get succcessful output
    output_is_complete = await manager.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success), now_seconds
    )
    assert output_is_complete == True


# TODO(ryan): Add tests for:
# - Ensure duplicate outputs are ignored
# - If before a call to PutInputs returns, we completely process the output,
#   make sure we don't put context for it in the manager.
