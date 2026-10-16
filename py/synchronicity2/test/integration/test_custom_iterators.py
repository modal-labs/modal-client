# Copyright Modal Labs 2026
"""Integration tests for custom_iterators_impl.py support file."""

import pytest
from pathlib import Path

from test.integration.test_utils import check_pyright


@pytest.mark.usefixtures("generated_wrappers")
def test_runtime():
    import custom_iterators

    for i in custom_iterators.get_iterator():
        print(i)

    for i in custom_iterators.get_iterable():
        print(i)

    iterable_1 = custom_iterators.CustomAsyncIterable()
    for i in iterable_1:
        print(i)

    iterable_2 = custom_iterators.IterableClassUsingGenerator()
    for i in iterable_2:
        print(i)

    iterable_3 = custom_iterators.IterableClassUsingGeneratorTyped()
    for i in iterable_3:
        print(i)

    iterator_1 = custom_iterators.CustomAsyncIterator([1, 2, 3])
    for i in iterator_1:
        break
    for i in iterator_1:
        pass
    assert iterator_1.num_iters == 2

    results = list(custom_iterators.get_custom_iterator())
    assert results == [10, 20, 30]


@pytest.mark.usefixtures("generated_wrappers")
@pytest.mark.asyncio
async def test_runtime_async():
    import custom_iterators

    async for i in custom_iterators.get_iterator():
        print(i)

    async for i in custom_iterators.get_iterable():
        print(i)

    iterable_1 = custom_iterators.CustomAsyncIterable()
    async for i in iterable_1:
        print(i)

    iterable_2 = custom_iterators.IterableClassUsingGenerator()
    async for i in iterable_2:
        print(i)

    iterable_3 = custom_iterators.IterableClassUsingGeneratorTyped()
    async for i in iterable_3:
        print(i)

    iterator_1 = custom_iterators.CustomAsyncIterator([1, 2, 3])
    async for i in iterator_1:
        break
    async for i in iterator_1:
        pass
    assert iterator_1.num_iters == 2

    results = []
    async for i in custom_iterators.get_custom_iterator():
        results.append(i)
    assert results == [10, 20, 30]


def test_pyright_implementation():
    import custom_iterators_impl

    check_pyright([Path(custom_iterators_impl.__file__)])


def test_pyright_wrapper():
    import custom_iterators

    check_pyright([Path(custom_iterators.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("custom_iterators_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
