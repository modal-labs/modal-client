# Copyright Modal Labs 2026
import typing_extensions

from synchronicity2 import Module

lib = Module("class_with_self_references")


@lib.wrap_class()
class SomeClass:
    def accept_self(self, s: typing_extensions.Self) -> typing_extensions.Self:
        assert type(s) is type(self)
        return self

    def accept_self_by_name(self, s: "SomeClass") -> "SomeClass":
        assert isinstance(s, SomeClass)  # this can be a subclass too
        return self

    @classmethod
    async def create(cls) -> typing_extensions.Self:
        return cls()


@lib.wrap_class()
class SomeSubclass(SomeClass):
    """Empty subclass: inherited Self-typed methods should use the subclass as Self."""
