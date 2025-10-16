# Copyright Modal Labs 2022
import pathlib
import pytest
import subprocess
import sys

from fastapi.testclient import TestClient

import modal
from modal import App, asgi_app, fastapi_endpoint, wsgi_app
from modal._runtime.asgi import magic_fastapi_app
from modal.exception import InvalidError
from modal.functions import Function
from modal.running_app import RunningApp
from modal_proto import api_pb2

app = App()


@app.function(cpu=42)
@fastapi_endpoint(method="PATCH", docs=True)
async def f(x):
    return {"square": x**2}


@pytest.mark.asyncio
async def test_webhook(servicer, client, reset_container_app):
    async with app.run(client=client):
        assert f.get_web_url()

        assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION
        assert servicer.app_functions["fu-1"].webhook_config.method == "PATCH"

        # Make sure we can call the webhooks
        # TODO: reinstate `.remote` check when direct webhook fn invocation is fixed.
        # assert await f.remote(10)
        assert await f.local(100) == {"square": 10000}

        # Make sure the container gets the app id as well
        container_app = RunningApp(app.app_id)
        app._init_container(client, container_app)
        assert isinstance(f, Function)
        assert f.get_web_url()


def test_webhook_cors():
    def handler():
        return {"message": "Hello, World!"}

    app = magic_fastapi_app(handler, method="GET", docs=False)
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

    app = magic_fastapi_app(handler, method="GET", docs=False)
    client = TestClient(app)
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


@pytest.mark.asyncio
async def test_webhook_docs():
    # By turning on docs, we should get three new routes: /docs, /redoc, and /openapi.json
    def handler():
        return {"message": "Hello, docs!"}

    app = magic_fastapi_app(handler, method="GET", docs=True)
    client = TestClient(app)
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_webhook_generator():
    app = App()

    with pytest.raises(InvalidError) as excinfo:

        @app.function(serialized=True)
        @fastapi_endpoint()
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
@pytest.mark.parametrize("decorator", [fastapi_endpoint, asgi_app, wsgi_app])
async def test_webhook_decorator_in_wrong_order(decorator):
    app = App()

    with pytest.raises(InvalidError, match=decorator.__name__) as excinfo:

        @decorator()  # type: ignore
        @app.function(serialized=True)
        async def g(x):
            pass

    assert "swap the order" in str(excinfo.value).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("decorator", [fastapi_endpoint, asgi_app, wsgi_app])
async def test_webhook_decorator_on_class(decorator):
    app = App()

    with pytest.raises(InvalidError, match=decorator.__name__) as excinfo:

        @app.cls(serialized=True)
        @decorator()  # type: ignore
        class C:
            @modal.method()
            def f(self):
                pass

    assert "method instead" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_asgi_wsgi(servicer, client):
    app = App()

    @app.function(serialized=True)
    @asgi_app()
    def my_asgi():
        pass

    @app.function(serialized=True)
    @wsgi_app()
    def my_wsgi():
        pass

    with pytest.raises(InvalidError, match="can't have parameters"):

        @app.function(serialized=True)
        @asgi_app()
        def my_invalid_asgi(x):
            pass

    with pytest.raises(InvalidError, match="can't have parameters"):

        @app.function(serialized=True)
        @wsgi_app()
        def my_invalid_wsgi(x):
            pass

    with pytest.raises(InvalidError, match="can't have parameters"):

        @app.function(serialized=True)
        @asgi_app()
        def my_deprecated_default_params_asgi(x=1):
            pass

    with pytest.raises(InvalidError, match="can't have parameters"):

        @app.function(serialized=True)
        @wsgi_app()
        def my_deprecated_default_params_wsgi(x=1):
            pass

    with pytest.raises(InvalidError, match="async function"):

        @app.function(serialized=True)
        @asgi_app()
        async def my_async_asgi_function():
            pass

    with pytest.raises(InvalidError, match="async function"):

        @app.function(serialized=True)
        @wsgi_app()
        async def my_async_wsgi_function():
            pass

    async with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 2
    assert servicer.app_functions["fu-1"].webhook_config.type == api_pb2.WEBHOOK_TYPE_ASGI_APP
    assert servicer.app_functions["fu-2"].webhook_config.type == api_pb2.WEBHOOK_TYPE_WSGI_APP


def test_positional_method(servicer, client):
    with pytest.raises(InvalidError, match="method="):
        fastapi_endpoint("GET")
    with pytest.raises(InvalidError, match="label="):
        asgi_app("baz")
    with pytest.raises(InvalidError, match="label="):
        wsgi_app("baz")


@pytest.mark.asyncio
async def test_asgi_app_missing_return(servicer, client):
    """Test that forgetting to return from @asgi_app() gives a clear error."""
    from unittest.mock import MagicMock

    from modal._runtime import user_code_imports
    from modal.app import _App

    def my_asgi_no_return():
        pass

    service = user_code_imports.ImportedFunction(
        app=_App(), service_deps=None, _user_defined_callable=my_asgi_no_return
    )
    fun_def = api_pb2.Function(webhook_config=api_pb2.WebhookConfig(type=api_pb2.WEBHOOK_TYPE_ASGI_APP))

    with pytest.raises(
        InvalidError, match=r"@modal\.asgi_app\(\).+callable.+NoneType.+Did you forget to add a return statement"
    ):
        service.get_finalized_functions(fun_def, container_io_manager=MagicMock())


@pytest.mark.asyncio
async def test_wsgi_app_missing_return(servicer, client):
    """Test that forgetting to return from @wsgi_app() gives a clear error."""
    from unittest.mock import MagicMock

    from modal._runtime import user_code_imports
    from modal.app import _App

    def my_wsgi_no_return():
        pass

    service = user_code_imports.ImportedFunction(
        app=_App(), service_deps=None, _user_defined_callable=my_wsgi_no_return
    )
    fun_def = api_pb2.Function(webhook_config=api_pb2.WebhookConfig(type=api_pb2.WEBHOOK_TYPE_WSGI_APP))

    with pytest.raises(
        InvalidError, match=r"@modal\.wsgi_app\(\).+callable.+NoneType.+Did you forget to add a return statement"
    ):
        service.get_finalized_functions(fun_def, container_io_manager=MagicMock())


@pytest.mark.asyncio
async def test_dev_suffix(servicer, client, modal_config):
    modal_toml = """
    [default]
    dev_suffix = "test"
    """
    with modal_config(modal_toml):
        app = modal.App()

        @app.function(serialized=True)
        @fastapi_endpoint()
        async def f(x): ...

        with servicer.intercept() as ctx:
            async with app.run(client=client):
                ...

            request = ctx.pop_request("FunctionCreate")
            assert request.function.webhook_config.ephemeral_suffix == "test"
