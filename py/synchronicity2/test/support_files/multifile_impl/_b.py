# Copyright Modal Labs 2026
"""Module B for multifile integration testing."""

import typing

from synchronicity2 import Module

if typing.TYPE_CHECKING:
    import multifile_impl._a

wrapper_module = Module("multifile.b")


@wrapper_module.wrap_class()
class B:
    """B test class."""

    def __init__(self, name: str = "test"):
        self.name = name

    async def get_name(self) -> str:
        return self.name


@wrapper_module.wrap_function()
async def get_a() -> "multifile_impl._a.A":
    """Create and return an A instance."""
    from ._a import A

    return A()
