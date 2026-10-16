# Copyright Modal Labs 2026
class LegacyValueImpl:
    def __init__(self, value: int):
        self._value = value

    @property
    def value(self) -> int:
        return self._value
