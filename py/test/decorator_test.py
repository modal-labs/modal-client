# Copyright Modal Labs 2023
import pytest

from modal import App, asgi_app, fastapi_endpoint, method, sessioned, wsgi_app
from modal.exception import InvalidError


def test_local_entrypoint_forgot_parentheses():
    app = App()

    with pytest.raises(TypeError, match="positional argument"):

        @app.local_entrypoint  # type: ignore
        def f():
            pass


def test_function_forgot_parentheses():
    app = App()

    with pytest.raises(TypeError, match="positional argument"):

        @app.function  # type: ignore
        def f():
            pass


def test_cls_forgot_parentheses():
    app = App()

    with pytest.raises(TypeError, match="positional argument"):

        @app.cls  # type: ignore
        class XYZ:  # type: ignore
            pass


def test_method_forgot_parentheses():
    app = App()

    with pytest.raises(TypeError, match="positional argument"):

        @app.cls()
        class XYZ:
            @method  # type: ignore
            def f(self):
                pass


def test_sessioned_forgot_parentheses():
    app = App()

    with pytest.raises(TypeError, match="positional argument"):

        @app.server(port=8000)  # type: ignore
        @sessioned  # type: ignore
        class XYZ:  # type: ignore
            pass


def test_invalid_web_decorator_usage():
    app = App()

    with pytest.raises(TypeError, match="positional argument"):

        @app.function()  # type: ignore
        @fastapi_endpoint  # type: ignore
        def my_handle():
            pass

    with pytest.raises(TypeError, match="positional argument"):

        @app.function()  # type: ignore
        @asgi_app  # type: ignore
        def my_handle_asgi():
            pass

    with pytest.raises(TypeError, match="positional argument"):

        @app.function()  # type: ignore
        @wsgi_app  # type: ignore
        def my_handle_wsgi():
            pass


def test_fastapi_endpoint_method():
    app = App()

    with pytest.raises(InvalidError, match="cannot be combined"):

        @app.cls(serialized=True)
        class Container:
            @method()  # type: ignore
            @fastapi_endpoint()
            def generate(self):
                pass
