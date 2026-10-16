# Copyright Modal Labs 2026
"""Consumer typing checks for multifile generated wrappers (sync + async)."""

from multifile.a import A, get_b
from multifile.b import B, get_a
from typing_extensions import assert_type


def _sync_usage() -> None:
    a = A(value=42)
    assert_type(a, A)
    assert_type(a.get_value(), int)

    b = get_b()
    assert_type(b, B)

    a2 = get_a()
    assert_type(a2, A)


async def _async_usage() -> None:
    a = A(value=42)
    av = await a.get_value.aio()
    assert_type(av, int)

    b = await get_b.aio()
    assert_type(b, B)

    a2 = await get_a.aio()
    assert_type(a2, A)
