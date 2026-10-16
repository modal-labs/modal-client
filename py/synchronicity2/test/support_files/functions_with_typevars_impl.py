# Copyright Modal Labs 2026
import typing

import synchronicity2

T = typing.TypeVar("T", bound="SomeClass")

mod = synchronicity2.Module("functions_with_typevars")


@mod.wrap_class()
class SomeClass:
    pass


@mod.wrap_class()
class Container:
    """A class with methods that use TypeVars."""

    async def tuple_to_list(self, items: tuple[T, ...]) -> list[T]:
        """Wrap any item in a list."""
        return list(items)
