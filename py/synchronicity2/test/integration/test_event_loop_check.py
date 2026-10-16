# Copyright Modal Labs 2026
"""Integration tests for event_loop_check_impl.py support file."""

import asyncio
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import event_loop_check

    async def test_function_aio():
        result = await event_loop_check.async_function.aio(5)
        return result

    assert asyncio.run(test_function_aio()) == 10

    async def test_generator_aio():
        results = []
        async for val in event_loop_check.async_generator.aio(3):
            results.append(val)
        return results

    assert asyncio.run(test_generator_aio()) == [0, 1, 2]

    checker = event_loop_check.EventLoopChecker(10)

    async def test_method_aio():
        result = await checker.async_method.aio()
        return result

    assert asyncio.run(test_method_aio()) == 20

    async def test_generator_method_aio():
        results = []
        async for val in checker.async_generator_method.aio(3):
            results.append(val)
        return results

    assert asyncio.run(test_generator_method_aio()) == [0, 10, 20]


def test_pyright_implementation():
    import event_loop_check_impl

    check_pyright([Path(event_loop_check_impl.__file__)])


def test_pyright_wrapper():
    import event_loop_check

    check_pyright([Path(event_loop_check.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("event_loop_check_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
