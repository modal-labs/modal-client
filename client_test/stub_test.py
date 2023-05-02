# Copyright Modal Labs 2022
import asyncio
import logging
import os

import pytest

from google.protobuf.empty_pb2 import Empty
from grpclib import GRPCError, Status

import modal.app
from modal import Client, Stub, web_endpoint, wsgi_app
from modal.aio import AioDict, AioQueue, AioStub, AioImage
from modal.exception import DeprecationError, InvalidError
from modal_proto import api_pb2
from modal_test_support import module_1, module_2


@pytest.mark.asyncio
async def test_kwargs(servicer, aio_client):
    stub = AioStub(
        d=AioDict(),
        q=AioQueue(),
    )
    async with stub.run(client=aio_client) as app:
        # TODO: interface to get type safe objects from live apps
        await app["d"].put("foo", "bar")  # type: ignore
        await app["q"].put("baz")  # type: ignore
        assert await app["d"].get("foo") == "bar"  # type: ignore
        assert await app["q"].get() == "baz"  # type: ignore


@pytest.mark.asyncio
async def test_attrs(servicer, aio_client):
    stub = AioStub()
    stub.d = AioDict()
    stub.q = AioQueue()
    async with stub.run(client=aio_client) as app:
        await app.d.put("foo", "bar")  # type: ignore
        await app.q.put("baz")  # type: ignore
        assert await app.d.get("foo") == "bar"  # type: ignore
        assert await app.q.get() == "baz"  # type: ignore


@pytest.mark.asyncio
async def test_stub_type_validation(servicer, aio_client):
    with pytest.raises(InvalidError):
        stub = AioStub(
            foo=4242,  # type: ignore
        )

    stub = AioStub()

    with pytest.raises(InvalidError) as excinfo:
        stub.bar = 4242  # type: ignore

    assert "4242" in str(excinfo.value)


def square(x):
    return x**2


@pytest.mark.asyncio
async def test_redeploy(servicer, aio_client):
    stub = AioStub()
    stub.function()(square)
    stub.image = AioImage.debian_slim().pip_install("pandas")

    # Deploy app
    app = await stub.deploy("my-app", client=aio_client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]

    # Redeploy, make sure all ids are the same
    app = await stub.deploy("my-app", client=aio_client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["square"] == "fu-1"
    assert servicer.app_state_history[app.app_id] == [
        api_pb2.APP_STATE_INITIALIZING,
        api_pb2.APP_STATE_DEPLOYED,
        api_pb2.APP_STATE_DEPLOYED,
    ]

    # Deploy to a different name, ids should change
    app = await stub.deploy("my-app-xyz", client=aio_client)
    assert app.app_id == "ap-2"
    assert servicer.app_objects["ap-2"]["square"] == "fu-2"
    assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def dummy():
    pass


# Should exit without waiting for the "logs_timeout" grace period.
@pytest.mark.timeout(5)
def test_create_object_exception(servicer, client):
    servicer.function_create_error = True

    stub = Stub()
    stub.function()(dummy)

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
    dummy_modal = stub.function()(dummy)

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
        stub.fun()  # type: ignore


def test_same_function_name(caplog):
    stub = Stub()

    # Add first function
    with caplog.at_level(logging.WARNING):
        stub.function()(module_1.square)
    assert len(caplog.records) == 0

    # Add second function: check warning
    with caplog.at_level(logging.WARNING):
        stub.function()(module_2.square)
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

    stub.function()(wsgi_app()(dummy))
    with pytest.warns(DeprecationError):
        stub.serve(client=client, timeout=1)


@skip_in_github
def test_serve_teardown(client, servicer):
    stub = Stub()
    with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        stub.function()(wsgi_app()(dummy))
        with pytest.warns(DeprecationError):
            stub.serve(client=client, timeout=1)

    disconnect_reqs = [s for s in servicer.requests if isinstance(s, api_pb2.AppClientDisconnectRequest)]
    assert len(disconnect_reqs) == 1


# Required as failing to raise could cause test to never return.
@skip_in_github
@pytest.mark.timeout(7)
def test_nested_serve_invocation(client):
    stub = Stub()

    stub.function()(wsgi_app()(dummy))
    with pytest.raises(InvalidError) as excinfo:
        with stub.run(client=client):
            # This nested call creates a second web endpoint!
            with pytest.warns(DeprecationError):
                stub.serve(client=client)
    assert "running" in str(excinfo.value)


def test_run_state(client, servicer):
    stub = Stub()
    with stub.run(client=client) as app:
        assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_EPHEMERAL]


def test_deploy_state(client, servicer):
    stub = Stub()
    app = stub.deploy("foobar", client=client)
    assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def test_detach_state(client, servicer):
    stub = Stub()
    with stub.run(client=client, detach=True) as app:
        assert servicer.app_state_history[app.app_id] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DETACHED]


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


async def web1(x):
    return {"square": x**2}


async def web2(x):
    return {"cube": x**3}


def test_registered_web_endpoints(client, servicer):
    stub = Stub()
    stub.function()(square)
    with pytest.warns(DeprecationError):
        stub.webhook(web1)
    stub.function()(web_endpoint()(web2))

    assert stub.registered_web_endpoints == ["web1", "web2"]


def test_init_types():
    with pytest.raises(InvalidError):
        # singular secret to plural argument
        Stub(secrets=modal.Secret.from_dict())  # type: ignore
    with pytest.raises(InvalidError):
        # not a Secret Object
        Stub(secrets=[{"foo": "bar"}])  # type: ignore
    with pytest.raises(InvalidError):
        # blueprint needs to use _Providers
        Stub(some_arg=5)  # type: ignore
    with pytest.raises(InvalidError):
        # should be an Image
        Stub(image=modal.Secret.from_dict())  # type: ignore

    Stub(
        image=modal.Image.debian_slim().pip_install("pandas"),
        secrets=[modal.Secret.from_dict()],
        mounts=[modal.Mount.from_local_file(__file__)],
        some_dict=modal.Dict(),
        some_queue=modal.Queue(),
    )


def test_set_image_on_stub_as_attribute():
    # TODO: do we want to deprecate this syntax? It's kind of random for image to
    #     have a reserved name in the blueprint, and being the only of the construction
    #     arguments that can be set on the instance after construction
    stub = Stub()
    custom_img = modal.Image.debian_slim().apt_install("emacs")
    stub.image = custom_img
    assert stub._get_default_image() == custom_img


@pytest.mark.asyncio
async def test_redeploy_persist(servicer, aio_client):
    stub = AioStub()
    stub.function()(square)
    stub.image = AioImage.debian_slim().pip_install("pandas")

    stub.d = AioDict()

    # Deploy app
    app = await stub.deploy("my-app", client=aio_client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["d"] == "di-0"

    stub.d = AioDict().persist("my-dict")
    # Redeploy, make sure all ids are the same
    app = await stub.deploy("my-app", client=aio_client)
    assert app.app_id == "ap-1"
    assert servicer.app_objects["ap-1"]["d"] == "di-1"
