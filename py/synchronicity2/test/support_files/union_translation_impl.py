# Copyright Modal Labs 2026
"""Implementation module for non-overloaded union translation integration tests."""

from __future__ import annotations

from synchronicity2 import Module

wrapper_module = Module("union_translation")


@wrapper_module.wrap_class()
class Box:
    value: int

    def __init__(self, value: int):
        self.value = value


@wrapper_module.wrap_function()
async def maybe_box(value: int, wrap: bool) -> int | Box:
    if wrap:
        return Box(value)
    return value


@wrapper_module.wrap_function()
async def extract_value(value: int | Box) -> int:
    if isinstance(value, Box):
        return value.value
    return value
