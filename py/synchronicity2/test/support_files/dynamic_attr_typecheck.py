# Copyright Modal Labs 2026
import typing

import dynamic_attr
from typing_extensions import assert_type

owner = dynamic_attr.DynamicOwner(dynamic_attr.Payload(7))

assert_type(owner.payload, dynamic_attr.Payload | typing.Any)
assert_type(owner.count, dynamic_attr.Payload | typing.Any)
