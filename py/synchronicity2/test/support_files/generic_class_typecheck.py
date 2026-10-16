# Copyright Modal Labs 2026
"""Consumer typing checks for generic_class wrappers."""

from generic_class import FunctionWrapper, SomeContainer, WrappedType, returning_container
from typing_extensions import assert_type


def f(a: int) -> float: ...


wrapped_func = FunctionWrapper(f)

sync_res = wrapped_func.call(a=10)
assert_type(sync_res, float)


async def foo() -> None:
    async_res = await wrapped_func.call.aio(a=10)
    assert_type(async_res, float)


copy = wrapped_func.clone()

assert_type(copy.call(a=11), float)

assert_type(returning_container(), SomeContainer[WrappedType])
