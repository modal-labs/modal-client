# Copyright Modal Labs 2026
from synchronicity1_legacy_impl import LegacyValueImpl

from synchronicity2 import Module

wrapper_module = Module("synchronicity1_interop")


@wrapper_module.wrap_function()
async def echo_legacy(value: LegacyValueImpl) -> LegacyValueImpl:
    assert isinstance(value, LegacyValueImpl)
    return value


@wrapper_module.wrap_function()
async def echo_legacy_list(values: list[LegacyValueImpl]) -> list[LegacyValueImpl]:
    assert all(isinstance(value, LegacyValueImpl) for value in values)
    return values
