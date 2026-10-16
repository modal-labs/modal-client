# Copyright Modal Labs 2026
"""Integration tests for nested_generators_impl.py support file."""

import asyncio
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import nested_generators

    str_gen_sync, int_gen_sync = nested_generators.nested_async_generator(3)
    assert list(str_gen_sync) == ["hello", "hello", "hello"]
    assert list(int_gen_sync) == [0, 1, 2]

    async def test_async():
        str_gen, int_gen = await nested_generators.nested_async_generator.aio(2)
        str_results = []
        async for s in str_gen:
            str_results.append(s)
        int_results = []
        async for i in int_gen:
            int_results.append(i)
        assert str_results == ["hello", "hello"]
        assert int_results == [0, 1]

    asyncio.run(test_async())

    async def test_independence():
        str_gen1, int_gen1 = await nested_generators.nested_async_generator.aio(1)
        async for _ in str_gen1:
            pass
        int_results = []
        async for i in int_gen1:
            int_results.append(i)
        return int_results

    assert asyncio.run(test_independence()) == [0]


def test_pyright_implementation():
    import nested_generators_impl

    check_pyright([Path(nested_generators_impl.__file__)])


def test_pyright_wrapper():
    import nested_generators

    check_pyright([Path(nested_generators.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("nested_generators_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
