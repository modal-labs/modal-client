# Copyright Modal Labs 2024
import pytest

import modal
from modal import App
from modal.exception import RemoteError
from modal.retries import RetryManager
from modal_proto import api_pb2

function_call_count = 0


@pytest.fixture(autouse=True)
def reset_function_call_count(monkeypatch):
    # Set default retry delay to something small so we don't slow down tests
    monkeypatch.setattr("modal.retries.MIN_INPUT_RETRY_DELAY_MS", 0.00001)
    global function_call_count
    function_call_count = 0


class FunctionCallCountException(Exception):
    """
    An exception which lets us report to the test how many times a function was called.
    """

    def __init__(self, function_call_count):
        self.function_call_count = function_call_count


def counting_function(return_success_on_retry_count: int):
    """
    A function that updates the global function_call_count counter each time it is called.

    """
    global function_call_count
    function_call_count += 1
    if function_call_count < return_success_on_retry_count:
        raise FunctionCallCountException(function_call_count)
    return function_call_count


@pytest.fixture
def setup_app_and_function(servicer):
    app = App()
    servicer.function_body(counting_function)
    retries = modal.Retries(
        max_retries=3,
        backoff_coefficient=1.0,
        initial_delay=0,
    )
    f = app.function(retries=retries)(counting_function)
    return app, f


def fetch_input_plane_request_counts(ctx):
    retried_requests = 0
    first_time_requests = 0
    for request in ctx.get_requests("MapStartOrContinue"):
        for item in request.items:
            if item.attempt_token == "":
                first_time_requests += 1
            else:
                retried_requests += 1
    return first_time_requests, retried_requests


@pytest.fixture
def setup_app_and_function_inputplane(servicer):
    app = App()
    servicer.function_body(counting_function)
    # Custom retries are not supported for inputplane functions yet.
    f = app.function(experimental_options={"input_plane_region": "us-east"})(counting_function)
    return app, f


def test_all_retries_fail_raises_error(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    with app.run(client=client):
        with pytest.raises(FunctionCallCountException) as exc_info:
            # The client should give up after the 4th call.
            f.remote(5)
        # Assert the function was called 4 times - the original call plus 3 retries
        assert exc_info.value.function_call_count == 4


def test_failures_followed_by_success(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    with servicer.intercept() as ctx:
        with app.run(client=client):
            function_call_count = f.remote(3)
            assert function_call_count == 3

    assert len(ctx.get_requests("FunctionRetryInputs")) == 2


def test_no_retries_when_first_call_succeeds(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    with app.run(client=client):
        function_call_count = f.remote(1)
        assert function_call_count == 1


def test_no_retries_when_call_cancelled(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    with servicer.intercept() as ctx:
        ctx.add_response(
            "FunctionGetOutputs",
            api_pb2.FunctionGetOutputsResponse(
                outputs=[
                    api_pb2.FunctionGetOutputsItem(
                        result=api_pb2.GenericResult(
                            status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED, exception="cancelled"
                        ),
                    )
                ]
            ),
        )

        with app.run(client=client):
            with pytest.raises(RemoteError, match="cancelled"):
                f.remote(1)

        assert not ctx.get_requests("FunctionRetryInputs")  # no retries


def test_no_retries_when_client_retries_disabled(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = False
    app, f = setup_app_and_function
    with app.run(client=client):
        with pytest.raises(FunctionCallCountException) as exc_info:
            f.remote(2)
        assert exc_info.value.function_call_count == 1


def test_retry_delay_ms():
    with pytest.raises(ValueError):
        RetryManager._retry_delay_ms(0, api_pb2.FunctionRetryPolicy())

    retry_policy = api_pb2.FunctionRetryPolicy(retries=2, backoff_coefficient=3, initial_delay_ms=2000)
    assert RetryManager._retry_delay_ms(1, retry_policy) == 2000

    retry_policy = api_pb2.FunctionRetryPolicy(retries=2, backoff_coefficient=3, initial_delay_ms=2000)
    assert RetryManager._retry_delay_ms(2, retry_policy) == 6000


def test_lost_inputs_retried(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    # The client should retry if it receives a internal failure status.
    servicer.failure_status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE

    with app.run(client=client):
        f.remote(10)
        # Assert the function was called 10 times
        assert function_call_count == 10


def test_map_fails_immediately_without_retries(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = False
    app, f = setup_app_and_function
    with app.run(client=client):
        with pytest.raises(FunctionCallCountException) as exc_info:
            list(f.map([999, 999, 999]))
        assert exc_info.value.function_call_count == 1


def test_map_all_retries_fail_raises_error(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    with servicer.intercept() as ctx:
        with app.run(client=client):
            with pytest.raises(FunctionCallCountException) as exc_info:
                list(f.map([999]))
            assert exc_info.value.function_call_count == 4
    assert len(ctx.get_requests("FunctionRetryInputs")) == 3


# TODO(ben-okeefe): Add when there is a retry policy for inputplane functions.
# def test_map_all_retries_fail_raises_error_inputplane(client, setup_app_and_function_inputplane, servicer):
#     servicer.sync_client_retries_enabled = True
#     app, f = setup_app_and_function_inputplane
#     with servicer.intercept() as ctx:
#         with app.run(client=client):
#             with pytest.raises(FunctionCallCountException) as exc_info:
#                 list(f.map([999]))
#             assert exc_info.value.function_call_count == 4

#     first_time_requests, retried_requests = fetch_input_plane_request_counts(ctx)
#     assert first_time_requests == 1
#     assert retried_requests == 3


def test_map_failures_followed_by_success(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    with servicer.intercept() as ctx:
        with app.run(client=client):
            results = list(f.map([3, 3, 3]))
            assert set(results) == {3, 4, 5}

    assert len(ctx.get_requests("FunctionRetryInputs")) == 2


# TODO(ben-okeefe): Add when there is a retry policy for inputplane functions.
# def test_map_failures_followed_by_success_inputplane(client, setup_app_and_function_inputplane, servicer):
#     servicer.sync_client_retries_enabled = True
#     app, f = setup_app_and_function_inputplane
#     with servicer.intercept() as ctx:
#         with app.run(client=client):
#             results = list(f.map([3, 3, 3]))
#             assert set(results) == {3, 4, 5}

#     first_time_requests, retried_requests = fetch_input_plane_request_counts(ctx)
#     assert first_time_requests == 3
#     assert retried_requests == 2


def test_map_no_retries_when_first_call_succeeds(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    with app.run(client=client):
        results = list(f.map([1, 1, 1]))
        assert set(results) == {1, 2, 3}


def test_map_no_retries_when_first_call_succeeds_inputplane(client, setup_app_and_function_inputplane, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function_inputplane
    with servicer.intercept() as ctx:
        with app.run(client=client):
            results = list(f.map([1, 1, 1]))
            assert set(results) == {1, 2, 3}

    first_time_requests, retried_requests = fetch_input_plane_request_counts(ctx)
    assert first_time_requests == 3
    assert retried_requests == 0


def test_map_lost_inputs_retried(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    # The client should retry if it receives a internal failure status.
    servicer.failure_status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE

    with app.run(client=client):
        results = list(f.map([3, 3, 3]))
        assert set(results) == {3, 4, 5}


def test_map_lost_inputs_retried_inputplane(client, setup_app_and_function_inputplane, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function_inputplane
    # The client should retry if it receives a internal failure status.
    servicer.failure_status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE

    with app.run(client=client):
        results = list(f.map([3, 3, 3]))
        assert set(results) == {3, 4, 5}


def test_map_cancelled_inputs_not_retried(client, setup_app_and_function, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function
    # The client should retry if it receives a internal failure status.
    servicer.failure_status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE

    with servicer.intercept() as ctx:

        async def FunctionGetOutputs(servicer, stream):
            # don't send response until an input arrives - otherwise it could cause a race
            await servicer.function_call_inputs_update_event.wait()
            await stream.send_message(
                api_pb2.FunctionGetOutputsResponse(
                    outputs=[
                        api_pb2.FunctionGetOutputsItem(
                            result=api_pb2.GenericResult(
                                status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED, exception="cancelled"
                            ),
                        )
                    ]
                )
            )

        ctx.set_responder("FunctionGetOutputs", FunctionGetOutputs)

        with app.run(client=client):
            with pytest.raises(RemoteError, match="cancelled"):
                list(f.map([3, 3, 3]))

        assert ctx.get_requests("FunctionRetryInputs") == []


def test_map_cancelled_inputs_not_retried_inputplane(client, setup_app_and_function_inputplane, servicer):
    servicer.sync_client_retries_enabled = True
    app, f = setup_app_and_function_inputplane
    # The client should retry if it receives a internal failure status.
    servicer.failure_status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE

    with servicer.intercept() as ctx:

        async def MapAwait(servicer, stream):
            # don't send response until an input arrives - otherwise it could cause a race
            await servicer.function_call_inputs_update_event.wait()
            await stream.send_message(
                api_pb2.MapAwaitResponse(
                    outputs=[
                        api_pb2.FunctionGetOutputsItem(
                            idx=1,
                            result=api_pb2.GenericResult(
                                status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED, exception="cancelled"
                            ),
                        )
                    ],
                    last_entry_id="1",
                )
            )

        ctx.set_responder("MapAwait", MapAwait)

        with app.run(client=client):
            with pytest.raises(RemoteError, match="cancelled"):
                list(f.map([3, 3, 3]))

        _, retried_requests = fetch_input_plane_request_counts(ctx)
        assert retried_requests == 0
