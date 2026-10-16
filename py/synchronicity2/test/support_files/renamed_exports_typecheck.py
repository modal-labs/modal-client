# Copyright Modal Labs 2026
"""Consumer typing checks for renamed wrapper exports."""

from renamed_exports import (
    AutoNamed,
    MyClass,
    _ExplicitlyPrivate,
    _make_explicitly_private,
    make_auto_named,
    make_my_class,
    unwrap_value,
)
from typing_extensions import assert_type

wrapped = MyClass(1)

assert_type(wrapped.get(), int)
assert_type(unwrap_value(wrapped), int)
assert_type(make_my_class(2), MyClass[int])
assert_type(make_auto_named(3), AutoNamed)
assert_type(_make_explicitly_private(4), _ExplicitlyPrivate)


async def check_async() -> None:
    assert_type(await wrapped.get.aio(), int)
    made = await make_my_class.aio(3)
    assert_type(made, MyClass[int])
    assert_type(await unwrap_value.aio(made), int)
    auto_named = await make_auto_named.aio(4)
    assert_type(auto_named, AutoNamed)
    explicit_private = await _make_explicitly_private.aio(5)
    assert_type(explicit_private, _ExplicitlyPrivate)
