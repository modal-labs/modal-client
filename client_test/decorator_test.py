# Copyright Modal Labs 2023
import pytest

from modal import Stub, method
from modal.exception import InvalidError


def test_local_entrypoint_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.local_entrypoint  # type: ignore
        def f():
            pass

    assert "local_entrypoint()" in str(excinfo.value)


def test_function_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.function  # type: ignore
        def f():
            pass

    assert "function()" in str(excinfo.value)


def test_cls_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.cls  # type: ignore
        class XYZ:
            pass

    assert "cls()" in str(excinfo.value)


def test_method_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.cls()
        class XYZ:
            @method  # type: ignore
            def f(self):
                pass

    assert "method()" in str(excinfo.value)
