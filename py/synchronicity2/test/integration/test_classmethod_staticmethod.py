# Copyright Modal Labs 2026
"""Integration tests for classmethod and staticmethod wrapping."""

import asyncio
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import classmethod_staticmethod

    obj = classmethod_staticmethod.TestClass(42)
    assert obj.instance_method() == 42

    async def test_async():
        result = await obj.instance_method.aio()
        assert result == 42

    asyncio.run(test_async())

    assert classmethod_staticmethod.TestClass.async_classmethod(2) == 84

    async def test_async_cm():
        result = await classmethod_staticmethod.TestClass.async_classmethod.aio(3)
        assert result == 126

    asyncio.run(test_async_cm())

    assert classmethod_staticmethod.TestClass.sync_classmethod("test") == "sync_test"

    assert classmethod_staticmethod.TestClass.async_staticmethod(10, 20) == 30

    async def test_async_sm():
        result = await classmethod_staticmethod.TestClass.async_staticmethod.aio(15, 25)
        assert result == 40

    asyncio.run(test_async_sm())

    assert classmethod_staticmethod.TestClass.sync_staticmethod("hello") == "static_hello"


def test_generated_wrapper_preserves_docstrings():
    import classmethod_staticmethod

    wrapper_source = Path(classmethod_staticmethod.__file__).read_text()

    assert '"""Return the instance value asynchronously."""' in wrapper_source
    assert '"""Multiply the class-level sentinel asynchronously."""' in wrapper_source
    assert '"""Prefix a string through a synchronous classmethod."""' in wrapper_source
    assert '"""Add two numbers asynchronously from a staticmethod."""' in wrapper_source
    assert '"""Prefix a string through a synchronous staticmethod."""' in wrapper_source


def test_pyright_implementation():
    import classmethod_staticmethod_impl

    check_pyright([Path(classmethod_staticmethod_impl.__file__)])


def test_pyright_wrapper():
    import classmethod_staticmethod

    check_pyright([Path(classmethod_staticmethod.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("classmethod_staticmethod_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
