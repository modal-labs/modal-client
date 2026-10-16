# Copyright Modal Labs 2026
"""Implementation module for overload-related wrapper generation tests."""

from __future__ import annotations

import typing

from synchronicity2 import Module

wrapper_module = Module("overloads")


@wrapper_module.wrap_class()
class Record:
    value: int

    def __init__(self, value: int):
        self.value = value


@typing.overload
async def duplicate(value: int) -> int: ...


@typing.overload
async def duplicate(value: str) -> str: ...


@wrapper_module.wrap_function()
async def duplicate(value: int | str) -> int | str:
    if isinstance(value, str):
        return value * 2
    return value * 2


@typing.overload
async def maybe_wrap(value: int, wrap: typing.Literal[False]) -> int: ...


@typing.overload
async def maybe_wrap(value: int, wrap: typing.Literal[True]) -> Record: ...


@wrapper_module.wrap_function()
async def maybe_wrap(value: int, wrap: bool) -> int | Record:
    if wrap:
        return Record(value)
    return value


@wrapper_module.wrap_class()
class Resolver:
    def __init__(self, offset: int):
        self.offset = offset

    @typing.overload
    async def resolve(self, value: int) -> int: ...

    @typing.overload
    async def resolve(self, value: str) -> str: ...

    async def resolve(self, value: int | str) -> int | str:
        if isinstance(value, str):
            return value + f":{self.offset}"
        return value + self.offset
