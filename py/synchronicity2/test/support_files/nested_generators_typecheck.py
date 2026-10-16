# Copyright Modal Labs 2026
"""Consumer typing checks for nested_generators wrappers."""

import nested_generators
from typing_extensions import assert_type


def _sync_usage() -> None:
    str_gen, int_gen = nested_generators.nested_async_generator(3)
    str_results = list(str_gen)
    int_results = list(int_gen)
    assert str_results == ["hello", "hello", "hello"]
    assert int_results == [0, 1, 2]
    assert_type(str_results, list[str])
    assert_type(int_results, list[int])


async def _async_usage() -> None:
    str_gen, int_gen = await nested_generators.nested_async_generator.aio(2)
    str_results: list[str] = []
    async for s in str_gen:
        str_results.append(s)
    int_results: list[int] = []
    async for i in int_gen:
        int_results.append(i)
    assert str_results == ["hello", "hello"]
    assert int_results == [0, 1]
    assert_type(str_results, list[str])
    assert_type(int_results, list[int])
