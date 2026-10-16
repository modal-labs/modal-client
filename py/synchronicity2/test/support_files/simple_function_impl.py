# Copyright Modal Labs 2026
"""Simple async function without any class dependencies."""

import datetime
import subprocess
import typing

from synchronicity2 import Module

wrapper_module = Module("simple_function")
DEFAULT_GREETING = "hello"


@wrapper_module.wrap_function()
async def simple_add(a: int, b: int) -> int:
    """Add two numbers asynchronously."""
    return a + b


@wrapper_module.wrap_function()
async def greet(name: str = DEFAULT_GREETING) -> str:
    """Return a greeting value, exercising emitted default arguments."""
    return name


@wrapper_module.wrap_function()
async def default_pipe(pipe: int = subprocess.PIPE) -> int:
    """Return a module-qualified default that requires a plain import in the wrapper."""
    return pipe


@wrapper_module.wrap_function()
async def round_trip_timestamp(value: datetime.datetime) -> datetime.datetime:
    """Return a timestamp value, exercising annotation imports."""
    return value


@wrapper_module.wrap_function()
async def simple_generator() -> typing.AsyncGenerator[int, None]:
    """Simple async generator."""
    for i in range(3):
        yield i


@wrapper_module.wrap_function()
def returns_awaitable() -> typing.Awaitable[str]:
    """Return an awaitable result.

    This docstring should stay multiline.
    """

    async def return_str():
        return "hello"

    return return_str()
