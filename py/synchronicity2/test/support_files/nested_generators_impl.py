# Copyright Modal Labs 2026
"""Test module for nested async generators.

Contains functions that return tuples or other collections of async generators,
testing the code generation for complex nested generator types.
"""

import typing

from synchronicity2 import Module

# Create the wrapper module
wrapper_module = Module("nested_generators")


@wrapper_module.wrap_function()
async def nested_async_generator(
    i: int,
) -> tuple[typing.AsyncGenerator[str, None], typing.AsyncGenerator[int, None]]:
    """Return tuple of two async generators."""

    async def f():
        for _ in range(i):
            yield "hello"

    async def g():
        for j in range(i):
            yield j

    return (f(), g())
