import pytest

from modal._utils.async_utils import TimestampPriorityQueue
from modal.parallel_map import _MapItemContext, _MapItemState
from modal.retries import RetryManager
from modal_proto import api_pb2

retry_policy = api_pb2.FunctionRetryPolicy(
    backoff_coefficient=1.0,
    initial_delay_ms=500,
    max_delay_ms=500,
    retries=2,
)

result_success = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
result_failure = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE)
result_internal_failure = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE)
now_seconds = 1738439812
input_data = api_pb2.FunctionInput(args=b"test")


@pytest.fixture
def retry_queue():
    return TimestampPriorityQueue()


def test_map_item_context_initial_state():
    map_item_context = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    assert map_item_context.state == _MapItemState.SENDING
    assert map_item_context.input == input_data
    assert map_item_context.retry_manager.retry_count == 0
    assert map_item_context.input_id.done() == False
    assert map_item_context.input_jwt.done() == False


@pytest.mark.asyncio
async def test_successful_output(retry_queue):
    map_item_context = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="input-0", input_jwt="jwt-0")
    map_item_context.handle_put_inputs_response(response_item)
    assert map_item_context.state == _MapItemState.WAITING_FOR_OUTPUT
    # We call result here rather than await because we want to test that the result has been set already.
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == True
    assert map_item_context.state == _MapItemState.COMPLETE
    assert map_item_context.retry_manager.retry_count == 0
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"


@pytest.mark.asyncio
async def test_failed_output_no_retries(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=0)
    map_item_context = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="input-0", input_jwt="jwt-0")
    map_item_context.handle_put_inputs_response(response_item)
    assert map_item_context.state == _MapItemState.WAITING_FOR_OUTPUT
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == True
    assert map_item_context.state == _MapItemState.COMPLETE
    assert map_item_context.retry_manager.retry_count == 1
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"
    assert retry_queue.empty()


@pytest.mark.asyncio
async def test_failed_output_retries_then_succeeds(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=1)
    map_item_context = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="input-0", input_jwt="jwt-0")
    map_item_context.handle_put_inputs_response(response_item)
    assert map_item_context.state == _MapItemState.WAITING_FOR_OUTPUT
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert map_item_context.state == _MapItemState.WAITING_TO_RETRY
    assert map_item_context.retry_manager.retry_count == 1
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"
    assert len(retry_queue) == 1

    # Retry input
    retry_item = await map_item_context.prepare_item_for_retry()
    assert retry_item.input_jwt == "jwt-0"
    assert retry_item.input == input_data
    assert retry_item.retry_count == 1
    assert map_item_context.state == _MapItemState.RETRYING

    map_item_context.handle_retry_response(api_pb2.FunctionRetryInputsResponseItem(idx=0, input_jwt="jwt-1"))
    assert map_item_context.state == _MapItemState.WAITING_FOR_OUTPUT
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-1"

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success, retry_count=1),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == True
    assert map_item_context.state == _MapItemState.COMPLETE
    # retry count is incremented only on failures
    assert map_item_context.retry_manager.retry_count == 1
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-1"


@pytest.mark.asyncio
async def test_lost_input_retries_then_succeeds(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=1)
    map_item_context = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="input-0", input_jwt="jwt-0")
    map_item_context.handle_put_inputs_response(response_item)
    assert map_item_context.state == _MapItemState.WAITING_FOR_OUTPUT
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_internal_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert map_item_context.state == _MapItemState.WAITING_TO_RETRY
    assert map_item_context.retry_manager.retry_count == 1
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"
    assert len(retry_queue) == 1

    # Retry input
    retry_item = await map_item_context.prepare_item_for_retry()
    assert retry_item.input_jwt == "jwt-0"
    assert retry_item.input == input_data
    assert retry_item.retry_count == 1
    assert map_item_context.state == _MapItemState.RETRYING

    map_item_context.handle_retry_response(api_pb2.FunctionRetryInputsResponseItem(idx=0, input_jwt="jwt-1"))
    assert map_item_context.state == _MapItemState.WAITING_FOR_OUTPUT
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-1"

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success, retry_count=1),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == True
    assert map_item_context.state == _MapItemState.COMPLETE
    # retry count is incremented only on failures
    assert map_item_context.retry_manager.retry_count == 1
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-1"


@pytest.mark.asyncio
async def test_failed_output_exhausts_retries(retry_queue):
    retry_policy = api_pb2.FunctionRetryPolicy(retries=1)
    map_item_context = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))
    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="input-0", input_jwt="jwt-0")
    map_item_context.handle_put_inputs_response(response_item)
    assert map_item_context.state == _MapItemState.WAITING_FOR_OUTPUT
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert map_item_context.state == _MapItemState.WAITING_TO_RETRY
    assert map_item_context.retry_manager.retry_count == 1
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"
    assert len(retry_queue) == 1

    # Retry input
    retry_item = await map_item_context.prepare_item_for_retry()
    assert retry_item.input_jwt == "jwt-0"
    assert retry_item.input == input_data
    assert retry_item.retry_count == 1
    assert map_item_context.state == _MapItemState.RETRYING

    map_item_context.handle_retry_response(api_pb2.FunctionRetryInputsResponseItem(idx=0, input_jwt="jwt-1"))
    assert map_item_context.state == _MapItemState.WAITING_FOR_OUTPUT
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-1"

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_failure, retry_count=1),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )
    assert changed_to_complete == True
    assert map_item_context.state == _MapItemState.COMPLETE
    # retry count is incremented only on failures
    assert map_item_context.retry_manager.retry_count == 2
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-1"


@pytest.mark.asyncio
async def test_get_successful_output_before_put_inputs_completes(retry_queue):
    map_item_context = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == True
    assert map_item_context.state == _MapItemState.COMPLETE
    assert map_item_context.retry_manager.retry_count == 0
    assert map_item_context.input_id.done() == False
    assert map_item_context.input_jwt.done() == False
    assert retry_queue.empty()

    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="input-0", input_jwt="jwt-0")
    map_item_context.handle_put_inputs_response(response_item)
    assert map_item_context.state == _MapItemState.COMPLETE
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"


@pytest.mark.asyncio
async def test_get_failed_output_before_put_inputs_completes(retry_queue):
    map_item_context = _MapItemContext(input=input_data, retry_manager=RetryManager(retry_policy))

    # Get outputs
    changed_to_complete = await map_item_context.handle_get_outputs_response(
        api_pb2.FunctionGetOutputsItem(idx=0, result=result_success),
        now_seconds,
        api_pb2.FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
        retry_queue,
    )

    assert changed_to_complete == False
    assert map_item_context.state == _MapItemState.WAITING_TO_RETRY
    assert map_item_context.retry_manager.retry_count == 1
    assert map_item_context.input_id.done() == False
    assert map_item_context.input_jwt.done() == False
    assert retry_queue.empty()

    # Put inputs
    response_item = api_pb2.FunctionPutInputsResponseItem(idx=0, input_id="input-0", input_jwt="jwt-0")
    map_item_context.handle_put_inputs_response(response_item)
    assert map_item_context.state == _MapItemState.WAITING_TO_RETRY
    assert map_item_context.input_id.result() == "input-0"
    assert map_item_context.input_jwt.result() == "jwt-0"


# TODO(ryan): Add test for retrying before put inputs completes. Need to check that we await
# for put inputs to return before retrying.
