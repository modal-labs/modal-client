# Copyright Modal Labs 2026
"""Integration tests for translated *args/**kwargs on class/static methods."""

import asyncio
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import variadic_translation

    first = variadic_translation.Node("first")
    second = variadic_translation.Node("second")

    assert variadic_translation.Collector.static_collect(first, second, primary=first) == (
        ["first", "second"],
        {"primary": "first"},
    )
    assert variadic_translation.Collector.class_collect(first, second, primary=second) == (
        ["first", "second"],
        {"primary": "second"},
    )

    async def test_async() -> None:
        assert await variadic_translation.Collector.static_collect.aio(first, second, primary=first) == (
            ["first", "second"],
            {"primary": "first"},
        )
        assert await variadic_translation.Collector.class_collect.aio(first, second, primary=second) == (
            ["first", "second"],
            {"primary": "second"},
        )

    asyncio.run(test_async())


def test_pyright_implementation():
    import variadic_translation_impl

    check_pyright([Path(variadic_translation_impl.__file__)])


def test_pyright_wrapper():
    import variadic_translation

    check_pyright([Path(variadic_translation.__file__)])


def test_pyright_usage():
    import variadic_translation_typecheck

    check_pyright([Path(variadic_translation_typecheck.__file__)])
