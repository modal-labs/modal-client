# Copyright Modal Labs 2026
"""Module A for multifile integration testing."""

import typing

from synchronicity2 import Module

if typing.TYPE_CHECKING:
    import multifile_impl._b

wrapper_module = Module("multifile.a")


@wrapper_module.wrap_class()
class A:
    """A test class."""

    def __init__(self, value: int = 42):
        self.value = value

    async def get_value(self) -> int:
        return self.value


@wrapper_module.wrap_function()
async def get_b() -> "multifile_impl._b.B":
    """Create and return a B instance."""
    from ._b import B

    return B()
