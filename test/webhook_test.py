# Copyright Modal Labs 2022
import pathlib
import pytest
import subprocess
import sys

from fastapi.testclient import TestClient

from modal import Stub, asgi_app, web_endpoint, wsgi_app
from modal._asgi import webhook_asgi_app
from modal.app import ContainerApp
from modal.exception import InvalidError
from modal.functions import Function
from modal_proto import api_pb2

stub = Stub()


@stub.function(cpu=42)
@web_endpoint(method="PATCH")
async def f(x):
    return {"square": x**2}


@pytest.mark.asyncio
async def test_webhook(servicer, client):
    async with stub.run(client=client):
        assert f.web_url

        assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION
        assert servicer.app_functions["fu-1"].webhook_config.method == "PATCH"

        # Make sure we can call the webhooks
        # TODO: reinstate `.remote` check when direct webhook fn invocation is fixed.
        # assert await f.remote(10)
        assert await f.local(100) == {"square": 10000}

        # Make sure the container gets the app id as well
        container_app = await ContainerApp.init_container.aio(client, stub.app_id)
        container_app._associate_stub_container(stub)
        f_c = stub["f"]
        assert isinstance(f_c, Function)
        assert f_c.web_url


def test_webhook_cors():
    def handler():
        return {"message": "Hello, World!"}

    app = webhook_asgi_app(handler, method="GET")
    client = TestClient(app)
    resp = client.options(
        "/",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers["Access-Control-Allow-Origin"] == "http://example.com"

    assert client.get("/").json() == {"message": "Hello, World!"}
    assert client.post("/").status_code == 405  # Method Not Allowed


@pytest.mark.asyncio
async def test_webhook_no_docs():
    # FastAPI automatically sets docs URLs for apps, which we disable because it
    # can be unexpected for users who are unfamilar with FastAPI.
    #
    # https://fastapi.tiangolo.com/tutorial/metadata/#docs-urls

    def handler():
        return {"message": "Hello, World!"}

    app = webhook_asgi_app(handler, method="GET")
    client = TestClient(app)
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


def test_webhook_generator():
    stub = Stub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.function(serialized=True)
        @web_endpoint()
        def web_gen():
            yield None

    assert "streaming" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_webhook_forgot_function(servicer, client):
    lib_dir = pathlib.Path(__file__).parent.parent
    args = [sys.executable, "-m", "modal_test_support.webhook_forgot_function"]
    ret = subprocess.run(args, cwd=lib_dir, stderr=subprocess.PIPE)
    stderr = ret.stderr.decode()
    assert "absent_minded_function" in stderr
    assert "@stub.function" in stderr


@pytest.mark.asyncio
async def test_webhook_decorator_in_wrong_order(servicer, client):
    stub = Stub()

    with pytest.raises(InvalidError) as excinfo:

        @web_endpoint()
        @stub.function(serialized=True)
        async def g(x):
            pass

    assert "wrong order" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_asgi_wsgi(servicer, client):
    stub = Stub()

    @stub.function(serialized=True)
    @asgi_app()
    async def my_asgi(x):
        pass

    @stub.function(serialized=True)
    @wsgi_app()
    async def my_wsgi(x):
        pass

    async with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 2
    assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP
    assert servicer.app_functions["fu-2"].webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP


def test_positional_method(servicer, client):
    with pytest.raises(InvalidError, match="method="):
        web_endpoint("GET")
    with pytest.raises(InvalidError, match="label="):
        asgi_app("baz")
    with pytest.raises(InvalidError, match="label="):
        wsgi_app("baz")
