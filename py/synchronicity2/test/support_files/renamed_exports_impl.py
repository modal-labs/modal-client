# Copyright Modal Labs 2026
from typing import Generic, TypeVar

import synchronicity2

T = TypeVar("T")

mod = synchronicity2.Module("renamed_exports")


@mod.wrap_class(name="MyClass")
class _ImplMyClass(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    async def get(self) -> T:
        return self.value


@mod.wrap_function(name="make_my_class")
async def _make_my_class(value: int) -> _ImplMyClass[int]:
    return _ImplMyClass(value)


@mod.wrap_function(name="unwrap_value")
async def _unwrap_value(instance: _ImplMyClass[int]) -> int:
    return instance.value


@mod.wrap_class()
class _AutoNamed:
    def __init__(self, value: int):
        self.value = value

    async def get(self) -> int:
        return self.value


@mod.wrap_function()
async def _make_auto_named(value: int) -> _AutoNamed:
    return _AutoNamed(value)


@mod.wrap_class(name="_ExplicitlyPrivate")
class _ImplExplicitlyPrivate:
    def __init__(self, value: int):
        self.value = value

    async def get(self) -> int:
        return self.value


@mod.wrap_function(name="_make_explicitly_private")
async def _make_explicitly_private_impl(value: int) -> _ImplExplicitlyPrivate:
    return _ImplExplicitlyPrivate(value)
