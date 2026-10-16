# Copyright Modal Labs 2026
from typing import Callable, Generic, ParamSpec, TypeVar

from typing_extensions import Self

import synchronicity2

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound="WrappedType")

mod = synchronicity2.Module("generic_class")


@mod.wrap_class()
class WrappedType:
    val: int

    def __init__(self, val: int):
        self.val = val


@mod.wrap_class()
class SomeContainer(Generic[T]):
    """A generic container class."""

    def __init__(self, value: T):
        assert isinstance(value, WrappedType)
        self.value = value

    async def get(self) -> T:
        return self.value

    async def set(self, value: T) -> None:
        assert isinstance(value, WrappedType)
        self.value = value


@mod.wrap_function()
def returning_container() -> SomeContainer[WrappedType]:
    return SomeContainer(WrappedType(1))


@mod.wrap_class()
class FunctionWrapper(Generic[P, R]):
    """A wrapper around a callable."""

    def __init__(self, f: Callable[P, R]):
        self.f = f

    async def call(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.f(*args, **kwargs)

    def clone(self) -> Self:
        return self
