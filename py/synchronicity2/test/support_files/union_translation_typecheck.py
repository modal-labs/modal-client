# Copyright Modal Labs 2026
"""Consumer typing checks for non-overloaded union translation wrappers."""

from __future__ import annotations

import union_translation
from typing_extensions import assert_type

wrapped = union_translation.Box(3)

sync_maybe_box = union_translation.maybe_box(3, True)
assert_type(sync_maybe_box, int | union_translation.Box)

sync_extract_wrapped = union_translation.extract_value(wrapped)
assert_type(sync_extract_wrapped, int)

sync_extract_int = union_translation.extract_value(7)
assert_type(sync_extract_int, int)


async def _async_usage() -> None:
    async_maybe_box = await union_translation.maybe_box.aio(3, True)
    assert_type(async_maybe_box, int | union_translation.Box)

    async_extract_wrapped = await union_translation.extract_value.aio(wrapped)
    assert_type(async_extract_wrapped, int)

    async_extract_int = await union_translation.extract_value.aio(7)
    assert_type(async_extract_int, int)
