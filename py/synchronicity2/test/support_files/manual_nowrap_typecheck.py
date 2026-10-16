# Copyright Modal Labs 2026
"""Consumer typing checks for manual-wrapper re-exports."""

import manual_nowrap
from typing_extensions import assert_type

manual_value = manual_nowrap.manual_function(1)
assert_type(manual_value, str)

box = manual_nowrap.ManualBox("box")
manual_method_value = box.manual_method(2)
assert_type(manual_method_value, str)

direct_box = manual_nowrap.DirectBox("value")
assert_type(direct_box, manual_nowrap.DirectBox)
assert_type(direct_box.reveal(), str)


async def _async_usage() -> None:
    manual_async_value = await manual_nowrap.manual_function.aio(3)
    assert_type(manual_async_value, str)

    manual_method_async_value = await box.manual_method.aio(4)
    assert_type(manual_method_async_value, str)

    auto_method_async_value = await box.auto_method.aio(5)
    assert_type(auto_method_async_value, str)
