# Copyright Modal Labs 2024
import asyncio
import collections
import pytest
import threading

from grpclib import GRPCError, Status

import modal
from modal import App, parallel_map
from modal._serialization import serialize_data_format
from modal.exception import InternalFailure, RemoteError
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
    app = App(include_source=False)
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


class _FakeInputPlane:
    """Creates map attempts like the input plane: retrying the token of attempt n creates attempt n + 1."""

    def __init__(self):
        self.attempts: list[tuple[int, int]] = []
        self._created: dict[tuple[int, int], asyncio.Event] = collections.defaultdict(asyncio.Event)

    def create(self, item: api_pb2.MapStartOrContinueItem) -> tuple[int, int]:
        attempt = (item.input.idx, int(item.attempt_token.split(":")[1]) + 1 if item.attempt_token else 0)
        self.attempts.append(attempt)
        self._created[attempt].set()
        return attempt

    async def wait_created(self, idx: int, n: int):
        await self._created[(idx, n)].wait()

    async def start_or_continue(self, _, stream):
        request = await stream.recv_message()
        await stream.send_message(self.response([self.create(item) for item in request.items]))

    @staticmethod
    def response(attempts: list[tuple[int, int]]) -> api_pb2.MapStartOrContinueResponse:
        return api_pb2.MapStartOrContinueResponse(
            map_token="map",
            attempt_tokens=[f"{idx}:{n}" for idx, n in attempts],
            max_inputs_outstanding=1000,
            retry_policy=api_pb2.FunctionRetryPolicy(),
        )

    @staticmethod
    def output(idx: int, n: int, value=None) -> api_pb2.FunctionGetOutputsItem:
        if value is None:
            result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE)
        else:
            data = serialize_data_format(value, api_pb2.DATA_FORMAT_PICKLE)
            result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS, data=data)
        return api_pb2.FunctionGetOutputsItem(
            idx=idx, retry_count=n, result=result, data_format=api_pb2.DATA_FORMAT_PICKLE
        )


@pytest.fixture
def setup_app_and_function_inputplane(servicer):
    app = App(include_source=False)
    servicer.function_body(counting_function)
    f = app.function(routing_region="us-east", retries=modal.Retries(max_retries=3, initial_delay=0.01))(
        counting_function
    )
    return app, f


def test_no_retries_when_first_call_succeeds(client, setup_app_and_function, servicer):
    app, f = setup_app_and_function
    with app.run(client=client):
        function_call_count = f.remote(1)
        assert function_call_count == 1


def test_failure_not_retried(client, setup_app_and_function, servicer):
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


def test_lost_inputs_not_retried(client, setup_app_and_function, servicer):
    app, f = setup_app_and_function
    servicer.failure_status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE

    with servicer.intercept() as ctx:
        with app.run(client=client):
            with pytest.raises(InternalFailure):
                f.remote(10)
    assert function_call_count == 1
    assert ctx.get_requests("FunctionRetryInputs") == []


def test_map_fails_immediately_without_retries(client, setup_app_and_function, servicer):
    app, f = setup_app_and_function
    with app.run(client=client):
        with pytest.raises(FunctionCallCountException) as exc_info:
            list(f.map([999, 999, 999]))
        assert exc_info.value.function_call_count == 1


def test_map_all_retries_fail_raises_error_inputplane(client, setup_app_and_function_inputplane, servicer):
    app, f = setup_app_and_function_inputplane
    with servicer.intercept() as ctx:
        with app.run(client=client):
            with pytest.raises(FunctionCallCountException) as exc_info:
                list(f.map([999]))
            assert exc_info.value.function_call_count == 4

    first_time_requests, retried_requests = fetch_input_plane_request_counts(ctx)
    assert first_time_requests == 1
    assert retried_requests == 3


def test_map_failures_followed_by_success_inputplane(client, setup_app_and_function_inputplane, servicer):
    app, f = setup_app_and_function_inputplane
    with servicer.intercept() as ctx:
        with app.run(client=client):
            results = list(f.map([3, 3, 3]))
            assert set(results) == {3, 4, 5}

    first_time_requests, retried_requests = fetch_input_plane_request_counts(ctx)
    assert first_time_requests == 3
    assert retried_requests == 2


def test_map_no_retries_when_first_call_succeeds(client, setup_app_and_function, servicer):
    app, f = setup_app_and_function
    with app.run(client=client):
        results = list(f.map([1, 1, 1]))
        assert set(results) == {1, 2, 3}


def test_map_no_retries_when_first_call_succeeds_inputplane(client, setup_app_and_function_inputplane, servicer):
    app, f = setup_app_and_function_inputplane
    with servicer.intercept() as ctx:
        with app.run(client=client):
            results = list(f.map([1, 1, 1]))
            assert set(results) == {1, 2, 3}

    first_time_requests, retried_requests = fetch_input_plane_request_counts(ctx)
    assert first_time_requests == 3
    assert retried_requests == 0


def test_map_lost_inputs_not_retried(client, setup_app_and_function, servicer):
    app, f = setup_app_and_function
    servicer.failure_status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE

    with servicer.intercept() as ctx:
        with app.run(client=client):
            with pytest.raises(InternalFailure):
                list(f.map([3, 3, 3]))
    assert ctx.get_requests("FunctionRetryInputs") == []


@pytest.mark.parametrize(
    "inputs, results, calls", [([21, 31], [21, 31], 5), ([51, 52], [51, 52], 10), ([91], [InternalFailure], 8)]
)
def test_map_lost_inputs_retried_inputplane(
    client, setup_app_and_function_inputplane, servicer, inputs, results, calls
):
    app, f = setup_app_and_function_inputplane
    # The client should retry if it receives a internal failure status.
    servicer.failure_status = api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE
    attempts: collections.Counter[int] = collections.Counter()

    def succeed_on_attempt_given_by_tens_digit(x):
        attempts[x] += 1
        if attempts[x] < x // 10:
            raise Exception("container crashed")
        return x

    servicer.function_body(succeed_on_attempt_given_by_tens_digit)
    with app.run(client=client):
        outputs = list(f.map(inputs, return_exceptions=True))
    assert [InternalFailure if isinstance(o, InternalFailure) else o for o in outputs] == results
    assert sum(attempts.values()) == calls


@pytest.mark.timeout(15)
def test_map_retry_after_lost_acknowledgment(client, setup_app_and_function_inputplane, servicer):
    # The server creates attempt 1 but its response is lost, and resending is rejected for having too many
    # uncollected outputs until the client polls again. Polling must continue, and attempt 2 must use attempt 1's token.
    app, f = setup_app_and_function_inputplane
    plane = _FakeInputPlane()
    trimmed = asyncio.Event()
    delivered = 0

    async def start_or_continue(_, stream):
        request = await stream.recv_message()
        item = request.items[0]
        if item.attempt_token != "1:0":
            attempt = plane.create(item)
        elif (1, 1) not in plane.attempts:
            plane.create(item)
            raise GRPCError(Status.UNAVAILABLE, "response lost")
        elif not trimmed.is_set():
            raise GRPCError(Status.RESOURCE_EXHAUSTED, "Map has too many uncollected outputs.")
        else:
            attempt = (1, 1)
        await stream.send_message(plane.response([attempt]))

    async def await_outputs(_, stream):
        nonlocal delivered
        await stream.recv_message()
        if delivered == 2:
            trimmed.set()
        await plane.wait_created(1, delivered)
        output = plane.output(1, delivered, value="ok" if delivered == 2 else None)
        delivered += 1
        await stream.send_message(api_pb2.MapAwaitResponse(outputs=[output], last_entry_id=str(delivered)))

    with servicer.intercept() as ctx:
        ctx.set_responder("MapStartOrContinue", start_or_continue)
        ctx.set_responder("MapAwait", await_outputs)
        with app.run(client=client):
            assert list(f.map([1])) == ["ok"]


@pytest.mark.timeout(15)
def test_map_lost_input_check_during_output_batch(client, setup_app_and_function_inputplane, servicer, monkeypatch):
    # The fake reports inputs as lost once the check's last_entry_id is past their outputs.
    app, f = setup_app_and_function_inputplane
    monkeypatch.setattr("modal.parallel_map.MAP_CHECK_INPUTS_INTERVAL", 0.001)
    plane = _FakeInputPlane()
    processing_batch, checked_during_batch = threading.Event(), threading.Event()
    cursors_during_batch: list[str] = []
    polls = 0
    handle_output = parallel_map._MapItemsManager.handle_get_outputs_response

    # Hold the batch open until a check runs, as a full output queue would.
    async def check_during_batch(self, item, now_seconds):
        if not processing_batch.is_set():
            processing_batch.set()
            await asyncio.get_running_loop().run_in_executor(None, checked_during_batch.wait, 10)
        return await handle_output(self, item, now_seconds)

    async def await_outputs(_, stream):
        nonlocal polls
        await stream.recv_message()
        n = min(polls, 1)
        polls += 1
        await plane.wait_created(1, n)
        await plane.wait_created(2, n)
        outputs = [plane.output(idx, n, value=value if n == 0 else "retried") for idx, value in [(1, "a"), (2, "b")]]
        await stream.send_message(api_pb2.MapAwaitResponse(outputs=outputs, last_entry_id=str(2 * polls)))

    async def check_inputs(_, stream):
        request = await stream.recv_message()
        if processing_batch.is_set() and not cursors_during_batch:
            cursors_during_batch.append(request.last_entry_id)
            checked_during_batch.set()
        lost = [request.last_entry_id == "2"] * len(request.attempt_tokens)
        await stream.send_message(api_pb2.MapCheckInputsResponse(lost=lost))

    monkeypatch.setattr(parallel_map._MapItemsManager, "handle_get_outputs_response", check_during_batch)
    with servicer.intercept() as ctx:
        ctx.set_responder("MapStartOrContinue", plane.start_or_continue)
        ctx.set_responder("MapAwait", await_outputs)
        ctx.set_responder("MapCheckInputs", check_inputs)
        with app.run(client=client):
            assert list(f.map(["x", "y"])) == ["a", "b"]

    assert cursors_during_batch == [""]
    assert sorted(plane.attempts) == [(1, 0), (2, 0)]


@pytest.mark.timeout(15)
def test_map_ignores_lost_input_check_for_a_replaced_attempt(
    client, setup_app_and_function_inputplane, servicer, monkeypatch
):
    app, f = setup_app_and_function_inputplane
    monkeypatch.setattr("modal.parallel_map.MAP_CHECK_INPUTS_INTERVAL", 0.001)
    plane = _FakeInputPlane()
    stale_check_sent, stale_check_answered, next_check_sent = asyncio.Event(), asyncio.Event(), asyncio.Event()
    polls = 0

    async def check_inputs(_, stream):
        request = await stream.recv_message()
        lost = [False] * len(request.attempt_tokens)
        if stale_check_answered.is_set():
            next_check_sent.set()
        elif list(request.attempt_tokens) == ["1:0"]:
            # Report attempt 0 as lost only after it has already failed and been retried.
            stale_check_sent.set()
            await plane.wait_created(1, 1)
            lost = [True]
            stale_check_answered.set()
        await stream.send_message(api_pb2.MapCheckInputsResponse(lost=lost))

    async def await_outputs(_, stream):
        nonlocal polls
        await stream.recv_message()
        polls += 1
        if polls == 1:
            await stale_check_sent.wait()
            output = plane.output(1, 0)
        else:
            await next_check_sent.wait()
            output = plane.output(*plane.attempts[-1], value="ok")
        await stream.send_message(api_pb2.MapAwaitResponse(outputs=[output], last_entry_id=str(polls)))

    with servicer.intercept() as ctx:
        ctx.set_responder("MapStartOrContinue", plane.start_or_continue)
        ctx.set_responder("MapAwait", await_outputs)
        ctx.set_responder("MapCheckInputs", check_inputs)
        with app.run(client=client):
            assert list(f.map(["x"])) == ["ok"]

    assert plane.attempts == [(1, 0), (1, 1)]


def test_map_internal_failures_preserve_user_retries(client, setup_app_and_function_inputplane, servicer):
    app, f = setup_app_and_function_inputplane

    def fail_internally_then_raise(success_after):
        servicer.failure_status = (
            api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE
            if function_call_count < 3
            else api_pb2.GenericResult.GENERIC_STATUS_FAILURE
        )
        return counting_function(success_after)

    servicer.function_body(fail_internally_then_raise)
    with app.run(client=client):
        assert list(f.map([7])) == [7]


@pytest.mark.timeout(15)
def test_map_stops_after_repeated_lost_input_checks(client, setup_app_and_function_inputplane, servicer, monkeypatch):
    app, f = setup_app_and_function_inputplane
    monkeypatch.setattr("modal.parallel_map.MAP_CHECK_INPUTS_INTERVAL", 0.001)
    servicer.function_is_running = True  # MapAwait returns no outputs.

    async def all_lost(_, stream):
        request = await stream.recv_message()
        await stream.send_message(api_pb2.MapCheckInputsResponse(lost=[True] * len(request.attempt_tokens)))

    with servicer.intercept() as ctx:
        ctx.set_responder("MapCheckInputs", all_lost)
        with app.run(client=client):
            with pytest.raises(InternalFailure, match="Server has lost track of input"):
                list(f.map([1]))
        assert fetch_input_plane_request_counts(ctx) == (1, 7)


def test_map_cancelled_inputs_not_retried_inputplane(client, setup_app_and_function_inputplane, servicer):
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
