# Copyright Modal Labs 2026
"""Consumer typing checks for custom_iterators wrappers."""

import custom_iterators
from typing_extensions import assert_type

custom_iterator = custom_iterators.CustomAsyncIterator([1, 2, 3])
assert_type(next(iter(custom_iterator)), int)

it = custom_iterators.IterableClassUsingGeneratorTyped()
assert_type(next(iter(it)), int)
