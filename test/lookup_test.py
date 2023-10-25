# Copyright Modal Labs 2023
import pytest

from modal import Function, Queue, Stub, web_endpoint
from modal.exception import DeprecationError, ExecutionError, NotFoundError
from modal.runner import deploy_stub


def test_persistent_object(servicer, client):
    stub = Stub()
    stub["q_1"] = Queue.new()
    deploy_stub(stub, "my-queue", client=client)

    q: Queue = Queue.lookup("my-queue", "q_1", client=client)
    assert isinstance(q, Queue)
    assert q.object_id == "qu-1"

    with pytest.raises(NotFoundError):
        Queue.lookup("bazbazbaz", client=client)


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
    assert not Queue._exists("my-queue", client=client)
    q1: Queue = Queue.new()
    q1._deploy("my-queue", client=client)
    assert Queue._exists("my-queue", client=client)
    q2: Queue = Queue.lookup("my-queue", client=client)
    assert q1.object_id == q2.object_id


def test_deploy_retain_id(servicer, client):
    q1: Queue = Queue.new()
    q2: Queue = Queue.new()
    q1._deploy("my-queue", client=client)
    q2._deploy("my-queue", client=client)
    assert q1.object_id == q2.object_id
