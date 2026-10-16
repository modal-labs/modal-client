# Copyright Modal Labs 2026
"""Integration tests for decorator_factory_impl.py support file."""

from importlib.util import find_spec
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import decorator_factory

    registry = decorator_factory.Registry()

    @registry.function()
    def stringify(x: int) -> str:
        return str(x)

    assert stringify.remote(3) == "3"


def test_pyright_implementation():
    import decorator_factory_impl

    check_pyright([Path(decorator_factory_impl.__file__)])


def test_pyright_wrapper():
    import decorator_factory

    check_pyright([Path(decorator_factory.__file__)])


def test_pyright_usage():
    spec = find_spec("decorator_factory_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
