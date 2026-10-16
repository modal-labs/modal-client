# Copyright Modal Labs 2026
from synchronicity1_interop import echo_legacy, echo_legacy_list
from synchronicity1_legacy import LegacyValue

value = LegacyValue(1)
result: LegacyValue = echo_legacy(value)
results: list[LegacyValue] = echo_legacy_list([value])
