# Copyright Modal Labs 2026
"""Integration tests for overload-support wrapper generation."""

from __future__ import annotations

import asyncio
from importlib.util import find_spec
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import overloads
    import overloads_impl

    assert overloads.duplicate(4) == 8
    assert overloads.duplicate("ab") == "abab"

    maybe_unwrapped = overloads.maybe_wrap(4, False)
    assert maybe_unwrapped == 4

    maybe_wrapped = overloads.maybe_wrap(4, True)
    assert isinstance(maybe_wrapped, overloads.Record)
    assert not isinstance(maybe_wrapped, overloads_impl.Record)
    assert maybe_wrapped.value == 4

    record = overloads.Record(6)
    assert record.value == 6

    resolver = overloads.Resolver(5)
    assert resolver.resolve(2) == 7
    assert resolver.resolve("name") == "name:5"

    async def check_async() -> None:
        assert await overloads.duplicate.aio(4) == 8
        assert await overloads.duplicate.aio("ab") == "abab"

        async_unwrapped = await overloads.maybe_wrap.aio(4, False)
        assert async_unwrapped == 4

        async_wrapped = await overloads.maybe_wrap.aio(4, True)
        assert isinstance(async_wrapped, overloads.Record)
        assert not isinstance(async_wrapped, overloads_impl.Record)
        assert async_wrapped.value == 4

        assert await resolver.resolve.aio(2) == 7
        assert await resolver.resolve.aio("name") == "name:5"

    asyncio.run(check_async())


def test_generated_wrapper_contains_overloads():
    import overloads

    source = Path(overloads.__file__).read_text()

    assert "class _duplicate_FunctionWithAio(FunctionWithAio):" in source
    assert "def __call__(self, value: int) -> int: ..." in source
    assert "def __call__(self, value: str) -> str: ..." in source
    assert "async def aio(self, value: int) -> int: ..." in source
    assert "async def aio(self, value: str) -> str: ..." in source
    assert "return self._sync_impl(value)" in source
    assert "def __call__(self, value: typing.Union[int, str]) -> typing.Union[int, str]:" in source
    assert "async def aio(self, value: typing.Union[int, str]) -> typing.Union[int, str]:" in source
    assert "@function_with_aio(_duplicate_FunctionWithAio)" in source
    assert "_run_function_async(overloads_impl.duplicate(value))" in source
    assert "class _maybe_wrap_FunctionWithAio(FunctionWithAio):" in source
    assert "def __call__(self, value: int, wrap: typing.Literal[False]) -> int: ..." in source
    assert 'def __call__(self, value: int, wrap: typing.Literal[True]) -> "Record": ...' in source
    assert "async def aio(self, value: int, wrap: typing.Literal[False]) -> int: ..." in source
    assert 'async def aio(self, value: int, wrap: typing.Literal[True]) -> "Record": ...' in source
    assert "@function_with_aio(_maybe_wrap_FunctionWithAio)" in source
    assert "class _Resolver_resolve_MethodWithAio(MethodWithAio):" in source
    assert "def __call__(self, value: int) -> int: ..." in source
    assert "def __call__(self, value: str) -> str: ..." in source
    assert "async def aio(self, value: int) -> int: ..." in source
    assert "async def aio(self, value: str) -> str: ..." in source
    assert "@method_with_aio(_Resolver_resolve_MethodWithAio)" in source
    assert (
        "_run_function_async(overloads_impl.Resolver.resolve(self._wrapper_instance._impl_instance, value))" in source
    )


def test_pyright_implementation():
    import overloads_impl

    check_pyright([Path(overloads_impl.__file__)])


def test_pyright_wrapper():
    import overloads

    check_pyright([Path(overloads.__file__)])


def test_pyright_usage():
    spec = find_spec("overloads_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
