# Copyright Modal Labs 2026
"""Unit tests for source-based default expression resolution."""

from __future__ import annotations

import ast
import inspect
import pathlib
import pytest
import subprocess
import sys
import time

from synchronicity2.codegen.parsing.defaults import resolve_parameter_default_expressions

DEFAULT_GREETING = "hello"


def literal_default(value: str = "hello") -> None:
    return None


def multiline_default(
    value: tuple[int, int] = (
        1,
        2,
    ),
) -> None:
    return None


def positional_keyword_defaults(a, /, b: int = 10, *, c: str = "hello", d: bool = False) -> None:
    return None


def impl_module_default(value: str = DEFAULT_GREETING) -> None:
    return None


def subprocess_default(pipe: int = subprocess.PIPE) -> None:
    return None


def pathlib_default(path: pathlib.Path = pathlib.Path("demo")) -> None:
    return None


def unstable_default(value: float = time.time()) -> None:
    return None


class DefaultExpressionService:
    async def configure(self, greeting: str = "hello", *, retries: int = 3) -> None:
        return None


def _resolve(func):
    return resolve_parameter_default_expressions(
        func,
        inspect.signature(func),
        impl_module=sys.modules[func.__module__],
        source_label_prefix=f"{func.__module__}.{func.__qualname__}",
    )


def test_resolve_literal_default_verbatim():
    resolved = _resolve(literal_default)

    assert resolved["value"].expression == '"hello"'
    assert resolved["value"].import_modules == ()


def test_resolve_multiline_default_preserves_exact_source():
    resolved = _resolve(multiline_default)

    assert resolved["value"].expression == "(\n        1,\n        2,\n    )"


def test_resolve_positional_and_keyword_only_defaults():
    resolved = _resolve(positional_keyword_defaults)

    assert {name: item.expression for name, item in resolved.items()} == {
        "b": "10",
        "c": '"hello"',
        "d": "False",
    }


def test_resolve_method_defaults():
    resolved = _resolve(DefaultExpressionService.configure)

    assert {name: item.expression for name, item in resolved.items()} == {
        "greeting": '"hello"',
        "retries": "3",
    }


def test_resolve_impl_module_prefixed_default():
    resolved = _resolve(impl_module_default)

    assert resolved["value"].expression == f"{__name__}.DEFAULT_GREETING"
    assert resolved["value"].import_modules == ()


def test_resolve_qualified_module_default_with_plain_import_ref():
    resolved = _resolve(subprocess_default)

    assert resolved["pipe"].expression == "subprocess.PIPE"
    assert resolved["pipe"].import_modules == ("subprocess",)


def test_resolve_qualified_callable_default_with_plain_import_ref():
    resolved = _resolve(pathlib_default)

    assert resolved["path"].expression == 'pathlib.Path("demo")'
    assert resolved["path"].import_modules == ("pathlib",)


def test_resolve_rejects_missing_source(monkeypatch):
    monkeypatch.setattr(
        "synchronicity2.codegen.parsing.defaults.inspect.getsource",
        lambda _func: (_ for _ in ()).throw(OSError("missing")),
    )

    with pytest.raises(TypeError, match=r"Could not recover source"):
        _resolve(literal_default)


def test_resolve_rejects_unextractable_source_segment(monkeypatch):
    original_get_source_segment = ast.get_source_segment

    def patched_get_source_segment(source, node, *, padded=False):
        if isinstance(node, ast.Constant) and node.value == "hello":
            return None
        return original_get_source_segment(source, node, padded=padded)

    monkeypatch.setattr("synchronicity2.codegen.parsing.defaults.ast.get_source_segment", patched_get_source_segment)

    with pytest.raises(TypeError, match=r"default expression could not be extracted from source"):
        _resolve(literal_default)


def test_resolve_rejects_unstable_defaults():
    with pytest.raises(TypeError, match=r"unsupported default expression"):
        _resolve(unstable_default)


def test_resolve_rejects_closure_local_defaults():
    local_default = "hello"

    def impl(value: str = local_default) -> None:
        return None

    with pytest.raises(TypeError, match=r"unsupported default expression"):
        _resolve(impl)
