# Copyright Modal Labs 2026
import decorator_factory
from typing_extensions import assert_type

registry = decorator_factory.Registry()


@registry.function()
def f(a: str) -> float:
    return 0.0


assert_type(f.remote("hello"), float)


@registry.function()
async def g(a: str) -> float:
    return 0.0


assert_type(g.remote("hello"), float)
