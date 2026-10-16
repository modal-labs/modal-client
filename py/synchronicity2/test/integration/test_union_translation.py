# Copyright Modal Labs 2026
"""Integration tests for non-overloaded union translation."""

from __future__ import annotations

import asyncio
from importlib.util import find_spec
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import union_translation
    import union_translation_impl

    maybe_unwrapped = union_translation.maybe_box(4, False)
    assert maybe_unwrapped == 4

    maybe_wrapped = union_translation.maybe_box(4, True)
    assert isinstance(maybe_wrapped, union_translation.Box)
    assert not isinstance(maybe_wrapped, union_translation_impl.Box)
    assert maybe_wrapped.value == 4

    assert union_translation.extract_value(5) == 5
    assert union_translation.extract_value(union_translation.Box(6)) == 6

    async def check_async() -> None:
        async_unwrapped = await union_translation.maybe_box.aio(4, False)
        assert async_unwrapped == 4

        async_wrapped = await union_translation.maybe_box.aio(4, True)
        assert isinstance(async_wrapped, union_translation.Box)
        assert not isinstance(async_wrapped, union_translation_impl.Box)
        assert async_wrapped.value == 4

        assert await union_translation.extract_value.aio(5) == 5
        assert await union_translation.extract_value.aio(union_translation.Box(6)) == 6

    asyncio.run(check_async())


def test_pyright_implementation():
    import union_translation_impl

    check_pyright([Path(union_translation_impl.__file__)])


def test_pyright_wrapper():
    import union_translation

    check_pyright([Path(union_translation.__file__)])


def test_pyright_usage():
    spec = find_spec("union_translation_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
