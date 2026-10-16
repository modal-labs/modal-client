# Copyright Modal Labs 2026
import typing

import descriptor_dunder
from typing_extensions import assert_type

payload = descriptor_dunder.Payload(3)
descriptor = descriptor_dunder.Descriptor(payload)


class Holder:
    attr = descriptor


assert_type(Holder.attr, descriptor_dunder.Descriptor | descriptor_dunder.Payload | typing.Any)
assert_type(Holder().attr, descriptor_dunder.Descriptor | descriptor_dunder.Payload | typing.Any)
