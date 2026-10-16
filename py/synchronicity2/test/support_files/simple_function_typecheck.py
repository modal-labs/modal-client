# Copyright Modal Labs 2026
"""Consumer typing checks for generated simple_function wrappers."""

import datetime
import subprocess

import simple_function
from typing_extensions import assert_type

x = simple_function.simple_add(1, 2)
assert_type(x, int)

y = simple_function.greet()
assert_type(y, str)

y = simple_function.default_pipe()
assert_type(y, int)
assert y == subprocess.PIPE

y = simple_function.round_trip_timestamp(datetime.datetime.now())
assert_type(y, datetime.datetime)

y = simple_function.returns_awaitable()
assert_type(y, str)
