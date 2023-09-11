# Copyright Modal Labs 2023
import pytest

from modal import Stub, asgi_app, method, web_endpoint, wsgi_app
from modal.exception import InvalidError


def test_local_entrypoint_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError, match="local_entrypoint()"):

        @stub.local_entrypoint  # type: ignore
        def f():
            pass


def test_function_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError, match="function()"):

        @stub.function  # type: ignore
        def f():
            pass


def test_cls_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError, match="cls()"):

        @stub.cls  # type: ignore
        class XYZ:
            pass


def test_method_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError, match="method()"):

        @stub.cls()
        class XYZ:
            @method  # type: ignore
            def f(self):
                pass


def test_invalid_web_decorator_usage():
    stub = Stub()

    with pytest.raises(InvalidError, match="web_endpoint()"):

        @stub.function()  # type: ignore
        @web_endpoint  # type: ignore
        def my_handle():
            pass

    with pytest.raises(InvalidError, match="asgi_app()"):

        @stub.function()  # type: ignore
        @asgi_app  # type: ignore
        def my_handle_asgi():
            pass

    with pytest.raises(InvalidError, match="wsgi_app()"):

        @stub.function()  # type: ignore
        @wsgi_app  # type: ignore
        def my_handle_wsgi():
            pass
