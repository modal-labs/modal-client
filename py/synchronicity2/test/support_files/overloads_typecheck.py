# Copyright Modal Labs 2026
"""Consumer typing checks for generated overload wrappers."""

from __future__ import annotations

import overloads
from typing_extensions import assert_type

sync_number = overloads.duplicate(3)
assert_type(sync_number, int)

sync_text = overloads.duplicate("ab")
assert_type(sync_text, str)

sync_unwrapped = overloads.maybe_wrap(3, False)
assert_type(sync_unwrapped, int)

sync_wrapped = overloads.maybe_wrap(3, True)
assert_type(sync_wrapped, overloads.Record)

resolver = overloads.Resolver(7)
resolved_number = resolver.resolve(1)
assert_type(resolved_number, int)

resolved_text = resolver.resolve("x")
assert_type(resolved_text, str)


async def _async_usage() -> None:
    async_number = await overloads.duplicate.aio(3)
    assert_type(async_number, int)

    async_text = await overloads.duplicate.aio("ab")
    assert_type(async_text, str)

    async_unwrapped = await overloads.maybe_wrap.aio(3, False)
    assert_type(async_unwrapped, int)

    async_wrapped = await overloads.maybe_wrap.aio(3, True)
    assert_type(async_wrapped, overloads.Record)

    async_resolved_number = await resolver.resolve.aio(1)
    assert_type(async_resolved_number, int)

    async_resolved_text = await resolver.resolve.aio("x")
    assert_type(async_resolved_text, str)
