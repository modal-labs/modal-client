# Copyright Modal Labs 2025
import pytest

import modal
from modal import App
from modal.client import Client
from modal.exception import InternalFailure
from modal.functions import Function
from modal.retries import Retries
from modal.runner import deploy_app
from test.conftest import MockClientServicer

app = App()


@app.function(experimental_options={"input_plane_region": "us-east"})
def foo():
    return "foo"


def test_foo(client: Client, servicer: MockClientServicer):
    # This verifies that FunctionCreate returns the input_plane_region in the response, and call the input plane.
    servicer.function_body(foo.get_raw_f())
    with app.run(client=client):
        assert foo.remote() == "foo"
        assert foo._get_metadata().input_plane_url is not None
        assert foo._get_metadata().input_plane_region == "us-east"


def test_lookup_foo(client: Client, servicer: MockClientServicer):
    # This verifies that FunctionGet returns the input_plane_region in the response, and we call the input plane.
    servicer.function_body(foo.get_raw_f())
    modal.App()
    deploy_app(app, "app", client=client)
    f = Function.from_name("app", "foo").hydrate(client)
    assert f.remote() == "foo"
    assert f._get_metadata().input_plane_url is not None
    assert f._get_metadata().input_plane_region == "us-east"


def test_retry(client: Client, servicer: MockClientServicer):
    # Tell the servicer to fail once, and then succeed. The client should retry the failed attempt.
    servicer.attempt_await_failures_remaining = 1
    servicer.function_body(foo.get_raw_f())
    with app.run(client=client):
        assert foo.remote() == "foo"
    # We don't have a great way to verify the call was actually retried. We can at least check that the servicer
    # decremented the attempts_to_fail counter, which indicates that the call was retried.
    assert servicer.attempt_await_failures_remaining == 0


def test_retry_limit(client: Client, servicer: MockClientServicer, monkeypatch):
    monkeypatch.setattr("modal._functions.MAX_INTERNAL_FAILURE_COUNT", 2)
    # Tell the servicer to fail once, and then succeed. The client should retry the failed attempt.
    servicer.attempt_await_failures_remaining = 3
    with pytest.raises(InternalFailure):
        with app.run(client=client):
            foo.remote()
    # Verify that the mock server's failure counter was decremented by 2.
    assert servicer.attempt_await_failures_remaining == 1


@app.function(experimental_options={"input_plane_region": "DEADBEEF"})
def failing_function():
    raise ValueError()


def test_no_user_retry_policy(client: Client, servicer: MockClientServicer):
    servicer.function_body(failing_function.get_raw_f())
    with app.run(client=client):
        with pytest.raises(ValueError):
            failing_function.remote()
    assert servicer.attempted_retries == 0


@app.function(experimental_options={"input_plane_region": "DEADBEEF"}, retries=Retries(max_retries=2))
def failing_function2():
    raise ValueError()


def test_user_retry_policy(client: Client, servicer: MockClientServicer):
    servicer.function_body(failing_function2.get_raw_f())
    with app.run(client=client):
        with pytest.raises(ValueError):
            failing_function2.remote()
    assert servicer.attempted_retries == 2
