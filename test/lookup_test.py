# Copyright Modal Labs 2023
import pytest

from modal import App, Function, Volume, fastapi_endpoint, web_endpoint
from modal.exception import ExecutionError, NotFoundError, ServerWarning
from modal.runner import deploy_app
from modal_proto import api_pb2


def test_persistent_object(servicer, client):
    volume_id = Volume.create_deployed("my-volume", client=client)

    v = Volume.from_name("my-volume").hydrate(client)
    assert v.object_id == volume_id

    with pytest.raises(NotFoundError):
        Volume.from_name("bazbazbaz").hydrate(client)


def square(x):
    # This function isn't deployed anyway
    pass


def test_lookup_function(servicer, client):
    app = App()

    app.function()(square)
    deploy_app(app, "my-function", client=client)

    f = Function.from_name("my-function", "square").hydrate(client)
    assert f.object_id == "fu-1"

    # Call it using two arguments
    f = Function.from_name("my-function", "square").hydrate(client)
    assert f.object_id == "fu-1"
    with pytest.raises(NotFoundError):
        f = Function.from_name("my-function", "cube").hydrate(client)

    # Make sure we can call this function
    assert f.remote(2, 4) == 20
    assert [r for r in f.map([5, 2], [4, 3])] == [41, 13]

    # Make sure the new-style local calls raise an error
    with pytest.raises(ExecutionError):
        assert f.local(2, 4) == 20


@pytest.mark.parametrize("decorator", [web_endpoint, fastapi_endpoint])
def test_webhook_lookup(servicer, client, decorator):
    app = App()
    app.function()(decorator(method="POST")(square))
    deploy_app(app, "my-webhook", client=client)

    f = Function.from_name("my-webhook", "square").hydrate(client)
    assert f.web_url


def test_deploy_exists(servicer, client):
    with pytest.raises(NotFoundError):
        Volume.from_name("my-volume").hydrate(client)
    Volume.create_deployed("my-volume", client=client)
    v1 = Volume.from_name("my-volume").hydrate(client)
    v2 = Volume.from_name("my-volume").hydrate(client)
    assert v1.object_id == v2.object_id


def test_create_if_missing(servicer, client):
    v1 = Volume.from_name("my-volume", create_if_missing=True).hydrate(client)
    v2 = Volume.from_name("my-volume").hydrate(client)
    assert v1.object_id == v2.object_id


def test_lookup_server_warnings(servicer, client):
    app = App()

    app.function()(square)
    deploy_app(app, "my-function", client=client)

    servicer.function_get_server_warnings = [
        api_pb2.Warning(
            type=api_pb2.Warning.WARNING_TYPE_CLIENT_DEPRECATION,
            message="xyz",
        )
    ]
    with pytest.warns(ServerWarning, match="xyz"):
        Function.from_name("my-function", "square").hydrate(client)

    servicer.function_get_server_warnings = [api_pb2.Warning(message="abc")]
    with pytest.warns(ServerWarning, match="abc"):
        Function.from_name("my-function", "square").hydrate(client)
