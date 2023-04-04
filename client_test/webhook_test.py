# Copyright Modal Labs 2022
import pytest

from modal_proto import api_pb2
from modal.aio import AioApp, AioStub
from modal.functions import AioFunctionHandle
from modal.exception import DeprecationError, InvalidError

stub = AioStub()


@stub.function(cpu=42)
@stub.web_endpoint(method="POST")
async def f(x):
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
        assert servicer.app_functions["fu-1"].webhook_config.method == "POST"
        assert servicer.app_functions["fu-2"].webhook_config.method == "PUT"

        # Make sure we can call the webhooks
        assert await f.call(10)
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
        @stub.web_endpoint()
        def web_gen():
            yield None

    assert "StreamingResponse" in str(excinfo.value)


@pytest.mark.asyncio
async def test_webhook_forgot_function(servicer, aio_client):
    stub = AioStub()

    @stub.web_endpoint()
    async def g(x):
        pass

    with pytest.raises(InvalidError) as excinfo:
        async with stub.run(client=aio_client):
            pass

    assert "@stub.function" in str(excinfo.value)

    with pytest.raises(InvalidError) as excinfo:
        await stub.deploy("webhook-test", client=aio_client)

    assert "@stub.function" in str(excinfo.value)


@pytest.mark.asyncio
async def test_webhook_decorator_in_wrong_order(servicer, aio_client):
    stub = AioStub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.web_endpoint()
        @stub.function(serialized=True)
        async def g(x):
            pass

    assert "wrong order" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_asgi_wsgi(servicer, aio_client):
    stub = AioStub()

    @stub.function(serialized=True)
    @stub.asgi_app()
    async def my_asgi(x):
        pass

    @stub.function(serialized=True)
    @stub.wsgi_app()
    async def my_wsgi(x):
        pass

    async with stub.run(client=aio_client):
        pass

    assert len(servicer.app_functions) == 2
    assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP
    assert servicer.app_functions["fu-2"].webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP
