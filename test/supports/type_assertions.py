# Copyright Modal Labs 2024
from typing_extensions import assert_type

from .functions import typed_func

ret = typed_func.remote(a="hello")
assert_type(ret, float)
