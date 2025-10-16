# Copyright Modal Labs 2025
from test.conftest import MockClientServicer

import modal
from modal import App
from modal.client import Client
from modal.functions import Function
from modal.runner import deploy_app

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
