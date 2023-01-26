# Copyright Modal Labs 2022
import asyncio
import logging
import os
import pytest

from google.protobuf.empty_pb2 import Empty
from grpclib import GRPCError, Status

import modal.app
from modal import Stub
from modal.aio import AioDict, AioQueue, AioStub
from modal.exception import InvalidError
from modal_proto import api_pb2
from modal_test_support import module_1, module_2


@pytest.mark.asyncio
async def test_kwargs(servicer, aio_client):
    stub = AioStub(
        d=AioDict(),
        q=AioQueue(),
    )
    async with stub.run(client=aio_client) as app:
        await app["d"].put("foo", "bar")
        await app["q"].put("baz")
        assert await app["d"].get("foo") == "bar"
        assert await app["q"].get() == "baz"


@pytest.mark.asyncio
async def test_attrs(servicer, aio_client):
    stub = AioStub()
    stub.d = AioDict()
    stub.q = AioQueue()
    async with stub.run(client=aio_client) as app:
        await app.d.put("foo", "bar")
        await app.q.put("baz")
        assert await app.d.get("foo") == "bar"
        assert await app.q.get() == "baz"


def square(x):
    return x**2


@pytest.mark.asyncio
async def test_redeploy(servicer, aio_client):
    stub = AioStub()
    stub.function(square)

    # Deploy app
    app = await stub.deploy("my-app", client=aio_client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state[app.app_id] == api_pb2.APP_STATE_DEPLOYED

    # Redeploy, make sure all ids are the same
    app = await stub.deploy("my-app", client=aio_client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state[app.app_id] == api_pb2.APP_STATE_DEPLOYED

    # Deploy to a different name, ids should change
    app = await stub.deploy("my-app-xyz", client=aio_client)
    assert app.app_id == "ap-2"
    assert servicer.app_objects["ap-2"]["square"] == "fu-2"
    assert servicer.app_state[app.app_id] == api_pb2.APP_STATE_DEPLOYED


def dummy():
    pass


# Should exit without waiting for the "logs_timeout" grace period.
@pytest.mark.timeout(5)
def test_create_object_exception(servicer, client):
    servicer.function_create_error = True

    stub = Stub()
    stub.function(dummy)

    with pytest.raises(GRPCError) as excinfo:
        with stub.run(client=client):
            pass

    assert excinfo.value.status == Status.INTERNAL


def test_deploy_falls_back_to_app_name(servicer, client):
    named_stub = Stub(name="foo_app")
    named_stub.deploy(client=client)
    assert "foo_app" in servicer.deployed_apps


def test_deploy_uses_deployment_name_if_specified(servicer, client):
    named_stub = Stub(name="foo_app")
    named_stub.deploy("bar_app", client=client)
    assert "bar_app" in servicer.deployed_apps
    assert "foo_app" not in servicer.deployed_apps


def test_run_function_without_app_error():
    stub = Stub()
    dummy_modal = stub.function(dummy)

    with pytest.raises(InvalidError) as excinfo:
        dummy_modal.call()

    assert "stub.run" in str(excinfo.value)


def test_is_inside_basic():
    stub = Stub()
    assert stub.is_inside() is False


def test_missing_attr():
    """Trying to call a non-existent function on the Stub should produce
    an understandable error message."""

    stub = Stub()
    with pytest.raises(KeyError):
        stub.fun()


def test_same_function_name(caplog):
    stub = Stub()

    # Add first function
    with caplog.at_level(logging.WARNING):
        stub.function(module_1.square)
    assert len(caplog.records) == 0

    # Add second function: check warning
    with caplog.at_level(logging.WARNING):
        stub.function(module_2.square)
    assert len(caplog.records) == 1
    assert "module_1" in caplog.text
    assert "module_2" in caplog.text
    assert "square" in caplog.text


skip_in_github = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Broken in GitHub Actions",
)


@skip_in_github
def test_serve(client):
    stub = Stub()

    stub.wsgi(dummy)
    stub.serve(client=client, timeout=1)


@skip_in_github
def test_serve_teardown(client, servicer):
    stub = Stub()
    with modal.client.Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        stub.wsgi(dummy)
        stub.serve(client=client, timeout=1)

    disconnect_reqs = [s for s in servicer.requests if isinstance(s, api_pb2.AppClientDisconnectRequest)]
    assert len(disconnect_reqs) == 1


# Required as failing to raise could cause test to never return.
@skip_in_github
@pytest.mark.timeout(7)
def test_nested_serve_invocation(client):
    stub = Stub()

    stub.wsgi(dummy)
    with pytest.raises(InvalidError) as excinfo:
        with stub.run(client=client):
            # This nested call creates a second web endpoint!
            stub.serve(client=client)
    assert "existing" in str(excinfo.value)


def test_run_state(client, servicer):
    stub = Stub()
    with stub.run(client=client) as app:
        assert servicer.app_state[app.app_id] == api_pb2.APP_STATE_EPHEMERAL


def test_deploy_state(client, servicer):
    stub = Stub()
    app = stub.deploy("foobar", client=client)
    assert servicer.app_state[app.app_id] == api_pb2.APP_STATE_DEPLOYED


def test_detach_state(client, servicer):
    stub = Stub()
    with stub.run(client=client, detach=True) as app:
        assert servicer.app_state[app.app_id] == api_pb2.APP_STATE_DETACHED


@pytest.mark.asyncio
async def test_grpc_protocol(aio_client, servicer):
    stub = AioStub()
    async with stub.run(client=aio_client):
        await asyncio.sleep(0.01)  # wait for heartbeat
    assert len(servicer.requests) == 4
    assert isinstance(servicer.requests[0], Empty)  # ClientHello
    assert isinstance(servicer.requests[1], api_pb2.AppCreateRequest)
    assert isinstance(servicer.requests[2], api_pb2.AppHeartbeatRequest)
    assert isinstance(servicer.requests[3], api_pb2.AppClientDisconnectRequest)
