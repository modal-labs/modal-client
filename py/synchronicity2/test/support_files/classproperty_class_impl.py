# Copyright Modal Labs 2026
"""Class with @classproperty decorators for translated class-level access."""

from synchronicity2 import Module, classproperty

wrapper_module = Module("classproperty_class")


@wrapper_module.wrap_class()
class Manager:
    """A wrapped type returned by a classproperty."""

    label: str

    def __init__(self, label: str):
        self.label = label

    async def describe(self) -> str:
        return f"manager:{self.label}"


@wrapper_module.wrap_class()
class Service:
    """A class exposing translated class-level properties."""

    _manager = Manager("primary")

    @classproperty
    def manager(cls) -> Manager:
        return cls._manager

    @classproperty
    def default_name(cls) -> str:
        return "service"
