# Copyright Modal Labs 2026
"""Integration tests for renamed_exports_impl.py support file."""

import asyncio
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import renamed_exports

    assert hasattr(renamed_exports, "MyClass")
    assert hasattr(renamed_exports, "make_my_class")
    assert hasattr(renamed_exports, "unwrap_value")
    assert hasattr(renamed_exports, "AutoNamed")
    assert hasattr(renamed_exports, "make_auto_named")
    assert hasattr(renamed_exports, "_ExplicitlyPrivate")
    assert hasattr(renamed_exports, "_make_explicitly_private")
    assert not hasattr(renamed_exports, "_ImplMyClass")
    assert not hasattr(renamed_exports, "_make_my_class")
    assert not hasattr(renamed_exports, "_unwrap_value")
    assert not hasattr(renamed_exports, "_AutoNamed")
    assert not hasattr(renamed_exports, "_make_auto_named")
    assert not hasattr(renamed_exports, "ImplExplicitlyPrivate")
    assert not hasattr(renamed_exports, "make_explicitly_private")

    wrapped = renamed_exports.MyClass(10)
    assert wrapped.get() == 10
    assert renamed_exports.unwrap_value(wrapped) == 10

    made = renamed_exports.make_my_class(11)
    assert isinstance(made, renamed_exports.MyClass)
    assert made.get() == 11

    auto_named = renamed_exports.make_auto_named(13)
    assert isinstance(auto_named, renamed_exports.AutoNamed)
    assert auto_named.get() == 13

    explicit_private = renamed_exports._make_explicitly_private(14)
    assert isinstance(explicit_private, renamed_exports._ExplicitlyPrivate)
    assert explicit_private.get() == 14

    async def run_async():
        async_value = await wrapped.get.aio()
        made_async = await renamed_exports.make_my_class.aio(12)
        unwrapped_async = await renamed_exports.unwrap_value.aio(made_async)
        auto_async = await renamed_exports.make_auto_named.aio(15)
        explicit_async = await renamed_exports._make_explicitly_private.aio(16)
        return async_value, made_async.get(), unwrapped_async, auto_async.get(), explicit_async.get()

    assert asyncio.run(run_async()) == (10, 12, 12, 15, 16)


def test_generated_wrapper_uses_override_names():
    import renamed_exports

    wrapper_source = Path(renamed_exports.__file__).read_text()

    assert "class MyClass" in wrapper_source
    assert "def make_my_class" in wrapper_source
    assert "def unwrap_value" in wrapper_source
    assert "class AutoNamed" in wrapper_source
    assert "def make_auto_named" in wrapper_source
    assert "class _ExplicitlyPrivate" in wrapper_source
    assert "def _make_explicitly_private" in wrapper_source
    assert "class _ImplMyClass(" not in wrapper_source
    assert "def _make_my_class(" not in wrapper_source
    assert "def _unwrap_value(" not in wrapper_source
    assert "class _AutoNamed(" not in wrapper_source
    assert "def _make_auto_named(" not in wrapper_source
    assert "class ImplExplicitlyPrivate(" not in wrapper_source
    assert "def make_explicitly_private(" not in wrapper_source


def test_pyright_implementation():
    import renamed_exports_impl

    check_pyright([Path(renamed_exports_impl.__file__)])


def test_pyright_wrapper():
    import renamed_exports

    check_pyright([Path(renamed_exports.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("renamed_exports_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
