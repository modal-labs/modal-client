# Copyright Modal Labs 2026
"""Test implementation with classmethod and staticmethod."""

import asyncio

from synchronicity2 import Module

wrapper_module = Module("classmethod_staticmethod")


@wrapper_module.wrap_class()
class TestClass:
    def __init__(self, value: int):
        self.value = value

    async def instance_method(self) -> int:
        """Return the instance value asynchronously."""
        await asyncio.sleep(0.01)
        return self.value

    @classmethod
    async def async_classmethod(cls, multiplier: int) -> int:
        """Multiply the class-level sentinel asynchronously."""
        await asyncio.sleep(0.01)
        return 42 * multiplier

    @classmethod
    def sync_classmethod(cls, value: str) -> str:
        """Prefix a string through a synchronous classmethod."""
        return f"sync_{value}"

    @staticmethod
    async def async_staticmethod(x: int, y: int) -> int:
        """Add two numbers asynchronously from a staticmethod."""
        await asyncio.sleep(0.01)
        return x + y

    @staticmethod
    def sync_staticmethod(text: str) -> str:
        """Prefix a string through a synchronous staticmethod."""
        return f"static_{text}"
