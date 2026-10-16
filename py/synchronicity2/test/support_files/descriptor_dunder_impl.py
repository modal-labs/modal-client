# Copyright Modal Labs 2026
import typing

import synchronicity2

module = synchronicity2.Module("descriptor_dunder")


@module.wrap_class()
class Payload:
    value: int

    def __init__(self, value: int):
        self.value = value


@module.wrap_class()
class Descriptor:
    def __init__(self, payload: Payload):
        self._payload = payload

    def __get__(self, obj, objtype=None) -> "Descriptor | Payload | typing.Any":
        if obj is None:
            return self
        return self._payload
