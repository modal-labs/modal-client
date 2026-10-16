# Copyright Modal Labs 2026
import typing

import synchronicity2

module = synchronicity2.Module("dynamic_attr")


@module.wrap_class()
class Payload:
    value: int

    def __init__(self, value: int):
        self.value = value


@module.wrap_class()
class DynamicOwner:
    def __init__(self, payload: Payload):
        self._payload = payload

    def __getattr__(self, name: str) -> Payload | typing.Any:
        if name == "payload":
            return self._payload
        if name == "count":
            return 3
        raise AttributeError(name)


@module.wrap_class()
class UnmarkedDynamicOwner:
    def __init__(self, payload: Payload):
        self._payload = payload

    def __getattr__(self, name: str) -> typing.Any:
        if name == "payload":
            return self._payload
        if name == "count":
            return 3
        raise AttributeError(name)
