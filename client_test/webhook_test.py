# Copyright Modal Labs 2022
import pathlib
import pytest
import subprocess
import sys

from modal_proto import api_pb2
from modal.aio import AioApp, AioStub, aio_web_endpoint, aio_asgi_app, aio_wsgi_app
from modal.functions import AioFunctionHandle
from modal.exception import DeprecationError, InvalidError

stub = AioStub()


@stub.function(cpu=42)
@aio_web_endpoint(method="PATCH")
async def f(x):
    return {"square": x**2}


with pytest.warns(DeprecationError):

    @stub.function(cpu=42)
    @stub.web_endpoint(method="POST")
    async def h(x):
        return {"square": x**2}


with pytest.warns(DeprecationError):

    @stub.webhook(method="PUT", cpu=42)
    async def g(x):
        return {"square": x**2}


@pytest.mark.asyncio
async def test_webhook(servicer, aio_client):
    async with stub.run(client=aio_client) as app:
        assert f.web_url
        assert g.web_url

        assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION
        assert servicer.app_functions["fu-1"].webhook_config.method == "PATCH"
        assert servicer.app_functions["fu-2"].webhook_config.method == "POST"

        # Make sure we can call the webhooks
        # TODO: reinstate `.call` check when direct webhook fn invocation is fixed.
        # assert await f.call(10)
        assert await f(100) == {"square": 10000}

        # Make sure the container gets the app id as well
        container_app = await AioApp.init_container(aio_client, app.app_id)
        assert isinstance(container_app.f, AioFunctionHandle)
        assert isinstance(container_app.g, AioFunctionHandle)
        assert container_app.f.web_url
        assert container_app.g.web_url


def test_webhook_cors():
    from fastapi.testclient import TestClient

    from modal._asgi import webhook_asgi_app

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


def test_webhook_generator():
    stub = AioStub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.function(serialized=True)
        @aio_web_endpoint()
        def web_gen():
            yield None

    assert "StreamingResponse" in str(excinfo.value)


@pytest.mark.asyncio
async def test_webhook_forgot_function(servicer, aio_client):
    lib_dir = pathlib.Path(__file__).parent.parent
    args = [sys.executable, "-m", "modal_test_support.webhook_forgot_function"]
    ret = subprocess.run(args, cwd=lib_dir, stderr=subprocess.PIPE)
    stderr = ret.stderr.decode()
    assert "absent_minded_function" in stderr
    assert "@stub.function" in stderr


@pytest.mark.asyncio
async def test_webhook_decorator_in_wrong_order(servicer, aio_client):
    stub = AioStub()

    with pytest.raises(InvalidError) as excinfo:

        @aio_web_endpoint()
        @stub.function(serialized=True)
        async def g(x):
            pass

    assert "wrong order" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_asgi_wsgi(servicer, aio_client):
    stub = AioStub()

    @stub.function(serialized=True)
    @aio_asgi_app()
    async def my_asgi(x):
        pass

    @stub.function(serialized=True)
    @aio_wsgi_app()
    async def my_wsgi(x):
        pass

    async with stub.run(client=aio_client):
        pass

    assert len(servicer.app_functions) == 2
    assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP
    assert servicer.app_functions["fu-2"].webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP
