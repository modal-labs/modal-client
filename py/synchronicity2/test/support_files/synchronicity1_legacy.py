# Copyright Modal Labs 2026
from synchronicity1_interop_runtime import synchronicity1_synchronizer
from synchronicity1_legacy_impl import LegacyValueImpl

LegacyValue = synchronicity1_synchronizer.wrap(
    LegacyValueImpl,
    name="LegacyValue",
    target_module=__name__,
)
