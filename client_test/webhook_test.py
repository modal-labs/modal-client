# Copyright Modal Labs 2022
import pytest

from modal.aio import AioApp, AioStub

stub = AioStub()


@stub.webhook(method="POST")
async def f(x):
    return {"square": x**2}


@pytest.mark.asyncio
async def test_webhook(servicer, aio_client):
    async with stub.run(client=aio_client) as app:
        assert f.web_url

        # Make sure the container gets the app id as well
        container_app = await AioApp.init_container(aio_client, app.app_id)
        assert container_app.f.web_url


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
