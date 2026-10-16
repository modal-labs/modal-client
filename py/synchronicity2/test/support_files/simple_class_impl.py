# Copyright Modal Labs 2026
"""Simple class with async methods, no dependencies on other classes."""

import typing

from synchronicity2 import Module

wrapper_module = Module("simple_class")


@wrapper_module.wrap_class()
class Counter:
    """A simple counter class."""

    count: int

    def __init__(self, start: int = 0):
        self.count = start

    async def increment(self) -> int:
        """Increment and return the new count."""
        self.count += 1
        return self.count

    async def get_multiples(self, n: int) -> typing.AsyncGenerator[int, None]:
        """Generate multiples of the count."""
        for i in range(n):
            yield self.count * i

    def sync_method(self) -> int:
        """Return the current count without changing it."""
        return self.count
