# Copyright Modal Labs 2026
"""Integration tests for class_with_inheritance_impl.py support file."""

import pytest
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import class_with_inheritance
    import class_with_inheritance_impl

    assert hasattr(class_with_inheritance, "WrappedBase")
    assert hasattr(class_with_inheritance, "WrappedSub")

    base = class_with_inheritance.WrappedBase(b="foo")
    sub = class_with_inheritance.WrappedSub(b="bar")

    assert isinstance(sub, class_with_inheritance.WrappedSub)
    assert not isinstance(sub, class_with_inheritance_impl.WrappedSub)
    assert not isinstance(sub, class_with_inheritance_impl.WrappedBase)
    assert isinstance(sub, class_with_inheritance.WrappedBase)
    assert not isinstance(base, class_with_inheritance.WrappedSub)

    sub2 = class_with_inheritance.WrappedSub(b="foo")
    result = sub2.wrapped_method(t=class_with_inheritance.WrappedType())
    assert result == []

    with pytest.raises(AttributeError, match="unwrapped_method"):
        sub2.unwrapped_method()  # type: ignore

    assert sub2._impl_instance.unwrapped_method() is True
    assert sub2.wrapped_in_sub() == {}

    base2 = class_with_inheritance.WrappedBase(b="foo")
    sub3 = class_with_inheritance.WrappedSub(b="bar")
    assert base2._impl_instance.a == 1
    assert base2._impl_instance.b == "hello"
    assert sub3._impl_instance.a == 1
    assert sub3._impl_instance.b == "hello"
    assert sub3._impl_instance.c == 1.5

    impl_sub = class_with_inheritance_impl.WrappedSub("hello")
    wrapper1 = class_with_inheritance.WrappedSub._from_impl(impl_sub)
    wrapper2 = class_with_inheritance.WrappedSub._from_impl(impl_sub)
    assert wrapper1 is wrapper2

    assert "wrapped_in_sub" in dir(class_with_inheritance.WrappedSub)
    assert "wrapped_method" in dir(class_with_inheritance.WrappedSub)
    sub4 = class_with_inheritance.WrappedSub(b="hello")
    assert sub4.wrapped_method(class_with_inheritance.WrappedType()) == []
    assert sub4.wrapped_in_sub() == {}

    instance_without_own_methods = class_with_inheritance.ClassWithoutOwnMethods("hello")
    instance_without_own_methods.wrapped_method(class_with_inheritance.WrappedType())


def test_pyright_implementation():
    import class_with_inheritance_impl

    check_pyright([Path(class_with_inheritance_impl.__file__)])


def test_pyright_wrapper():
    import class_with_inheritance

    check_pyright([Path(class_with_inheritance.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("class_with_inheritance_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
