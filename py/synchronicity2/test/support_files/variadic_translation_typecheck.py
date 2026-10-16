# Copyright Modal Labs 2026
"""Consumer typing checks for translated *args/**kwargs wrappers."""

import variadic_translation
from typing_extensions import assert_type


def _sync_usage() -> None:
    first = variadic_translation.Node("first")
    second = variadic_translation.Node("second")

    static_result = variadic_translation.Collector.static_collect(first, second, primary=first)
    class_result = variadic_translation.Collector.class_collect(first, second, primary=second)

    assert_type(static_result, tuple[list[str], dict[str, str]])
    assert_type(class_result, tuple[list[str], dict[str, str]])


async def _async_usage() -> None:
    first = variadic_translation.Node("first")
    second = variadic_translation.Node("second")

    static_result = await variadic_translation.Collector.static_collect.aio(first, second, primary=first)
    class_result = await variadic_translation.Collector.class_collect.aio(first, second, primary=second)

    assert_type(static_result, tuple[list[str], dict[str, str]])
    assert_type(class_result, tuple[list[str], dict[str, str]])
