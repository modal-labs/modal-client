# Copyright Modal Labs 2026
"""Class with @property decorators for sync property delegation."""

from synchronicity2 import Module

wrapper_module = Module("property_class")


@wrapper_module.wrap_class()
class Tag:
    """A simple wrapped type used as a property value."""

    label: str

    def __init__(self, label: str):
        self.label = label


@wrapper_module.wrap_class()
class Settings:
    """A class with sync properties."""

    def __init__(self, name: str, max_retries: int = 3):
        self._name = name
        self._max_retries = max_retries
        self._call_count = 0
        self._tag = Tag("default")

    @property
    def name(self) -> str:
        """Read-only property."""
        return self._name

    @property
    def max_retries(self) -> int:
        """Read-write property."""
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self._max_retries = value

    @property
    def call_count(self) -> int:
        """Computed read-only property."""
        return self._call_count

    @property
    def tag(self) -> Tag:
        """Property that returns a wrapped type."""
        return self._tag

    @tag.setter
    def tag(self, value: Tag) -> None:
        self._tag = value

    async def do_work(self) -> str:
        """Async method that uses internal state."""
        self._call_count += 1
        return f"{self._name}: done (attempt {self._call_count}/{self._max_retries})"
