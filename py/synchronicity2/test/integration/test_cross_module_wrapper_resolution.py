# Copyright Modal Labs 2026
"""Integration test for lazy cross-module wrapper resolution."""

import sys


def test_runtime_lazy_import():
    import cross_wrapper.base

    assert "cross_wrapper.sub" not in sys.modules

    foo = cross_wrapper.base.foo_getter()

    assert "cross_wrapper.sub" in sys.modules

    import cross_wrapper.sub

    bar = cross_wrapper.sub.bar_getter()

    assert isinstance(foo, cross_wrapper.base.Foo)
    assert isinstance(foo, cross_wrapper.sub.Bar)
    assert foo is bar
