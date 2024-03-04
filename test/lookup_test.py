# Copyright Modal Labs 2023
import pytest

from modal import Function, Stub, Volume, web_endpoint
from modal.exception import DeprecationError, ExecutionError, NotFoundError
from modal.runner import deploy_stub


def test_persistent_object(servicer, client):
    volume_id = Volume.create_deployed("my-volume", client=client)

    v: Volume = Volume.lookup("my-volume", client=client)
    assert isinstance(v, Volume)
    assert v.object_id == volume_id

    with pytest.raises(NotFoundError):
        Volume.lookup("bazbazbaz", client=client)


def square(x):
    # This function isn't deployed anyway
    pass


def test_lookup_function(servicer, client):
    stub = Stub()

    stub.function()(square)
    deploy_stub(stub, "my-function", client=client)

    f = Function.lookup("my-function", "square", client=client)
    assert f.object_id == "fu-1"

    # Call it using two arguments
    f = Function.lookup("my-function", "square", client=client)
    assert f.object_id == "fu-1"
    with pytest.raises(NotFoundError):
        f = Function.lookup("my-function", "cube", client=client)

    # Make sure we can call this function
    assert f.remote(2, 4) == 20
    assert [r for r in f.map([5, 2], [4, 3])] == [41, 13]

    # Make sure the new-style local calls raise an error
    with pytest.raises(ExecutionError):
        assert f.local(2, 4) == 20

    # Make sure the old-style local calls raise an error
    with pytest.raises(DeprecationError):
        assert f(2, 4)


def test_webhook_lookup(servicer, client):
    stub = Stub()
    stub.function()(web_endpoint(method="POST")(square))
    deploy_stub(stub, "my-webhook", client=client)

    f = Function.lookup("my-webhook", "square", client=client)
    assert f.web_url


def test_deploy_exists(servicer, client):
    with pytest.raises(NotFoundError):
        Volume.lookup("my-volume", client=client)
    Volume.create_deployed("my-volume", client=client)
    v1: Volume = Volume.lookup("my-volume", client=client)
    v2: Volume = Volume.lookup("my-volume", client=client)
    assert v1.object_id == v2.object_id


def test_create_if_missing(servicer, client):
    v1: Volume = Volume.lookup("my-volume", create_if_missing=True, client=client)
    v2: Volume = Volume.lookup("my-volume", client=client)
    assert v1.object_id == v2.object_id
