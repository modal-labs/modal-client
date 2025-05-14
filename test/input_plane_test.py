# Copyright Modal Labs 2025
import modal
from modal import App
from modal.functions import Function
from modal.runner import deploy_app

app = App()


@app.function(experimental_options={"input_plane_region": "us-east"})
def foo():
    return "foo"


def test_foo(client, servicer):
    # This verifies that FunctionCreate returns the input_plane_url in the response, and we then call the input plane.
    with app.run(client=client):
        assert foo.remote() == "attempt_await_bogus_response"


def test_lookup_foo(client, servicer):
    # This verifies that FunctionGet returns the input_plane_url in the response, and we then call the input plane.
    modal.App()
    deploy_app(app, "app", client=client)
    f = Function.from_name("app", "foo").hydrate(client)
    assert f.remote() == "attempt_await_bogus_response"

@app.function(experimental_options={"input_plane_region": "us-east"})
def maybe_fail():
    raise ValueError("fail")

def test_retry(client, servicer):
    # Tell the servicer to fail once, and then succeed. The client should retry the failed attempt.
    servicer.attempts_to_fail = 1
    with app.run(client=client):
        assert foo.remote() == "attempt_await_bogus_response"
    # We don't have a great way to verify the call was actually retried. We can at least check that the servicer
    # decremented the attempts_to_fail counter, which indicates that the call was retried.
    assert servicer.attempts_to_fail == 0

