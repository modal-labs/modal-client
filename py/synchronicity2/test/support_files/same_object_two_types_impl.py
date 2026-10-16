# Copyright Modal Labs 2026
from synchronicity2 import Module

wrapper_module = Module("same_object_two_types")


@wrapper_module.wrap_class()
class Foo: ...


@wrapper_module.wrap_class()
class Bar(Foo): ...


bar = Bar()


@wrapper_module.wrap_function()
def foo_getter() -> Foo:
    return bar  # this is ok since bar is a Foo


@wrapper_module.wrap_function()
def bar_getter() -> Bar:
    return bar
