# Copyright Modal Labs 2026
"""Integration tests for opt-in generation of single-underscore methods."""

import asyncio
from importlib.util import find_spec
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime() -> None:
    import include_underscored_methods

    tool = include_underscored_methods.Tool("runtime")

    assert tool.public("x") == "runtime:public:x"
    assert tool._private("x") == "runtime:private:x"
    assert list(tool._stream(3)) == ["runtime:0", "runtime:1", "runtime:2"]
    assert include_underscored_methods.Tool._make("made")._private("x") == "made:private:x"
    assert not hasattr(include_underscored_methods.Tool, "_static")
    assert not hasattr(include_underscored_methods.DefaultTool(), "_hidden")

    async def check_async() -> tuple[str, list[str]]:
        private_result = await tool._private.aio("async")
        stream_result = [value async for value in tool._stream.aio(2)]
        return private_result, stream_result

    assert asyncio.run(check_async()) == ("runtime:private:async", ["runtime:0", "runtime:1"])


def test_generated_wrapper_source() -> None:
    import include_underscored_methods

    wrapper_source = Path(include_underscored_methods.__file__).read_text()

    assert "def _private(" in wrapper_source
    assert "def _stream(" in wrapper_source
    assert "def _make(" in wrapper_source
    assert "def _static(" not in wrapper_source
    assert "def _private_property(" not in wrapper_source
    assert "def _hidden(" not in wrapper_source


def test_pyright_implementation() -> None:
    import include_underscored_methods_impl

    check_pyright([Path(include_underscored_methods_impl.__file__)])


def test_pyright_wrapper() -> None:
    import include_underscored_methods

    check_pyright([Path(include_underscored_methods.__file__)])


def test_pyright_usage() -> None:
    spec = find_spec("include_underscored_methods_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
