# Copyright Modal Labs 2026
"""Consumer typing checks for same_object_two_types wrappers."""

from same_object_two_types import Bar, Foo, bar_getter, foo_getter
from typing_extensions import assert_type

foo = foo_getter()
bar = bar_getter()

assert_type(foo, Foo)
assert_type(bar, Bar)
