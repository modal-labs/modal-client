# Copyright Modal Labs 2024
import pytest

import modal
from modal import App
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


def counting_function(return_success_on_attempt_number: int):
    """
    A function that updates the global function_call_count counter each time it is called.

    """
    global function_call_count
    function_call_count += 1
    if function_call_count < return_success_on_attempt_number:
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


def test_all_retries_fail_raises_error(client, setup_app_and_function, monkeypatch):
    monkeypatch.setenv("MODAL_CLIENT_RETRIES", "true")
    app, f = setup_app_and_function
    with app.run(client=client):
        with pytest.raises(FunctionCallCountException) as exc_info:
            f.remote(5)
        assert exc_info.value.function_call_count == 4


def test_failures_followed_by_success(client, setup_app_and_function, monkeypatch):
    monkeypatch.setenv("MODAL_CLIENT_RETRIES", "true")
    app, f = setup_app_and_function
    with app.run(client=client):
        function_call_count = f.remote(3)
        assert function_call_count == 3


def test_no_retries_when_first_call_succeeds(client, setup_app_and_function, monkeypatch):
    monkeypatch.setenv("MODAL_CLIENT_RETRIES", "true")
    app, f = setup_app_and_function
    with app.run(client=client):
        function_call_count = f.remote(1)
        assert function_call_count == 1


def test_retry_dealy_ms():
    with pytest.raises(ValueError):
        RetryManager._retry_delay_ms(0, api_pb2.FunctionRetryPolicy())

    retry_policy = api_pb2.FunctionRetryPolicy(retries=2, backoff_coefficient=3, initial_delay_ms=2000)
    assert RetryManager._retry_delay_ms(1, retry_policy) == 2000

    retry_policy = api_pb2.FunctionRetryPolicy(retries=2, backoff_coefficient=3, initial_delay_ms=2000)
    assert RetryManager._retry_delay_ms(2, retry_policy) == 6000


def test_map_all_retries_fail_raises_error(client, setup_app_and_function, monkeypatch):
    monkeypatch.setenv("MODAL_CLIENT_RETRIES", "true")
    app, f = setup_app_and_function
    with app.run(client=client):
        with pytest.raises(FunctionCallCountException) as exc_info:
            list(f.map([999, 999, 999]))
        assert exc_info.value.function_call_count == 10


def test_map_failures_followed_by_success(client, setup_app_and_function, monkeypatch):
    monkeypatch.setenv("MODAL_CLIENT_RETRIES", "true")
    app, f = setup_app_and_function
    with app.run(client=client):
        results = list(f.map([3, 3, 3]))
        assert set(results) == {3, 4, 5}


def test_map_no_retries_when_first_call_succeeds(client, setup_app_and_function, monkeypatch):
    monkeypatch.setenv("MODAL_CLIENT_RETRIES", "true")
    app, f = setup_app_and_function
    with app.run(client=client):
        results = list(f.map([1, 1, 1]))
        assert set(results) == {1, 2, 3}
