# Copyright Modal Labs 2026
"""Base implementation module for cross-module wrapper resolution."""

import typing

from synchronicity2 import Module

if typing.TYPE_CHECKING:
    pass

wrapper_module = Module("cross_wrapper.base")


@wrapper_module.wrap_class()
class Foo: ...


@wrapper_module.wrap_function()
def foo_getter() -> Foo:
    from ._sub import bar

    return bar
