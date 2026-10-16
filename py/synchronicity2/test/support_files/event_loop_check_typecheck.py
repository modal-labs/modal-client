# Copyright Modal Labs 2026
"""Consumer typing checks for event_loop_check wrappers."""

import event_loop_check
from typing_extensions import assert_type


def _sync_usage() -> None:
    assert_type(event_loop_check.async_function(5), int)
    assert list(event_loop_check.async_generator(3)) == [0, 1, 2]
    checker = event_loop_check.EventLoopChecker(10)
    assert_type(checker.async_method(), int)
    assert list(checker.async_generator_method(3)) == [0, 10, 20]


async def _async_usage() -> None:
    assert_type(await event_loop_check.async_function.aio(5), int)
    results: list[int] = []
    async for val in event_loop_check.async_generator.aio(3):
        results.append(val)
    assert results == [0, 1, 2]
    checker = event_loop_check.EventLoopChecker(10)
    assert_type(await checker.async_method.aio(), int)
    results2: list[int] = []
    async for val in checker.async_generator_method.aio(3):
        results2.append(val)
    assert results2 == [0, 10, 20]
