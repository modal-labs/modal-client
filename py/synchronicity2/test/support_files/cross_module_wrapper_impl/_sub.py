# Copyright Modal Labs 2026
"""Subclass implementation module for cross-module wrapper resolution."""

from synchronicity2 import Module

from ._base import Foo

wrapper_module = Module("cross_wrapper.sub")


@wrapper_module.wrap_class()
class Bar(Foo): ...


bar = Bar()


@wrapper_module.wrap_function()
def bar_getter() -> Bar:
    return bar
