# Copyright Modal Labs 2026
"""Classes that opt into generating single-underscore methods."""

import typing

from synchronicity2 import Module

mod = Module("include_underscored_methods")


@mod.wrap_class(include_underscored_methods=True)
class Tool:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    async def public(self, value: str) -> str:
        return f"{self.prefix}:public:{value}"

    async def _private(self, value: str) -> str:
        return f"{self.prefix}:private:{value}"

    @property
    def _private_property(self) -> str:
        return f"{self.prefix}:property"

    async def _stream(self, count: int) -> typing.AsyncGenerator[str, None]:
        for i in range(count):
            yield f"{self.prefix}:{i}"

    @classmethod
    async def _make(cls, prefix: str) -> "Tool":
        return cls(prefix)

    @staticmethod
    async def _static(value: str) -> str:
        return f"static:{value}"


@mod.wrap_class()
class DefaultTool:
    async def _hidden(self) -> str:
        return "hidden"
