# Copyright Modal Labs 2026
"""Test implementation for async staticmethods with method-local type variables."""

import asyncio
from typing import TypeVar

from synchronicity2 import Module

T = TypeVar("T")

wrapper_module = Module("staticmethod_method_local_typevar")


@wrapper_module.wrap_class()
class EchoBox:
    @staticmethod
    async def echo(value: T) -> T:
        await asyncio.sleep(0.01)
        return value
