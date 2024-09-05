# Copyright Modal Labs 2022
import pathlib
import pytest
import subprocess
import sys

from fastapi.testclient import TestClient

from modal import App, asgi_app, web_endpoint, wsgi_app
from modal._asgi import webhook_asgi_app
from modal.exception import DeprecationError, InvalidError
from modal.functions import Function
from modal.running_app import RunningApp
from modal_proto import api_pb2

app = App()


@app.function(cpu=42)
@web_endpoint(method="PATCH", docs=True)
async def f(x):
    return {"square": x**2}


@pytest.mark.asyncio
async def test_webhook(servicer, client, reset_container_app):
    async with app.run(client=client):
        assert f.web_url

        assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION
        assert servicer.app_functions["fu-1"].webhook_config.method == "PATCH"

        # Make sure we can call the webhooks
        # TODO: reinstate `.remote` check when direct webhook fn invocation is fixed.
        # assert await f.remote(10)
        assert await f.local(100) == {"square": 10000}

        # Make sure the container gets the app id as well
        container_app = RunningApp(app_id=app.app_id)
        app._init_container(client, container_app)
        assert isinstance(f, Function)
        assert f.web_url


def test_webhook_cors():
    def handler():
        return {"message": "Hello, World!"}

    app = webhook_asgi_app(handler, method="GET", docs=False)
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
    # FastAPI automatically sets docs URLs for apps, which we disable by default because it
    # can be unexpected for users who are unfamilar with FastAPI.
    #
    # https://fastapi.tiangolo.com/tutorial/metadata/#docs-urls

    def handler():
        return {"message": "Hello, World!"}

    app = webhook_asgi_app(handler, method="GET", docs=False)
    client = TestClient(app)
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


@pytest.mark.asyncio
async def test_webhook_docs():
    # By turning on docs, we should get three new routes: /docs, /redoc, and /openapi.json
    def handler():
        return {"message": "Hello, docs!"}

    app = webhook_asgi_app(handler, method="GET", docs=True)
    client = TestClient(app)
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_webhook_generator():
    app = App()

    with pytest.raises(InvalidError) as excinfo:

        @app.function(serialized=True)
        @web_endpoint()
        def web_gen():
            yield None

    assert "streaming" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_webhook_forgot_function(servicer, client):
    lib_dir = pathlib.Path(__file__).parent.parent
    args = [sys.executable, "-m", "test.supports.webhook_forgot_function"]
    ret = subprocess.run(args, cwd=lib_dir, stderr=subprocess.PIPE)
    stderr = ret.stderr.decode()
    assert "absent_minded_function" in stderr
    assert "@app.function" in stderr


@pytest.mark.asyncio
async def test_webhook_decorator_in_wrong_order(servicer, client):
    app = App()

    with pytest.raises(InvalidError) as excinfo:

        @web_endpoint()  # type: ignore
        @app.function(serialized=True)
        async def g(x):
            pass

    assert "wrong order" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_asgi_wsgi(servicer, client):
    app = App()

    @app.function(serialized=True)
    @asgi_app()
    async def my_asgi():
        pass

    @app.function(serialized=True)
    @wsgi_app()
    async def my_wsgi():
        pass

    with pytest.raises(InvalidError, match="can't have parameters"):

        @app.function(serialized=True)
        @asgi_app()
        async def my_invalid_asgi(x):
            pass

    with pytest.raises(InvalidError, match="can't have parameters"):

        @app.function(serialized=True)
        @wsgi_app()
        async def my_invalid_wsgi(x):
            pass

    with pytest.warns(DeprecationError, match="default parameters"):

        @app.function(serialized=True)
        @asgi_app()
        async def my_deprecated_default_params_asgi(x=1):
            pass

    with pytest.warns(DeprecationError, match="default parameters"):

        @app.function(serialized=True)
        @wsgi_app()
        async def my_deprecated_default_params_wsgi(x=1):
            pass

    async with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 4
    assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP
    assert servicer.app_functions["fu-2"].webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP
    assert servicer.app_functions["fu-3"].webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP
    assert servicer.app_functions["fu-4"].webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP


def test_positional_method(servicer, client):
    with pytest.raises(InvalidError, match="method="):
        web_endpoint("GET")
    with pytest.raises(InvalidError, match="label="):
        asgi_app("baz")
    with pytest.raises(InvalidError, match="label="):
        wsgi_app("baz")
