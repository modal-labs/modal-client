# Copyright Modal Labs 2022
import pytest

from modal.exception import DeprecationError
from modal_utils.decorator_utils import decorator_with_options


@decorator_with_options
def dec(f, add=0):
    def wrapped(x):
        return f(x) + add

    return wrapped


def test_no_args_warns():
    with pytest.warns(DeprecationError, match="[^.]dec"):

        @dec
        def f(x):
            return x + 7

    assert f(42) == 49  # should still be usable


def test_empty_args():
    @dec()
    def f(x):
        return x + 7

    assert f(42) == 49  # should still be usable


def test_args():
    @dec(add=5)
    def g(x):
        return x + 1

    assert g(42) == 48


def double(x):
    return 2 * x


def test_direct_no_args_warns():
    with pytest.warns(DeprecationError, match="[^.]dec"):
        p = dec(double)
    assert p(42) == 84


def test_direct_empty_args():
    p = dec()(double)
    assert p(42) == 84


def test_direct_args_warns():
    with pytest.warns(DeprecationError, match="[^.]dec"):
        q = dec(double, add=9)
    assert q(42) == 93


def test_indirect_args_nowarning():
    q = dec(add=9)(double)
    assert q(42) == 93


# Test it as a method too


class Cls:
    def __init__(self, add_more):
        self.add_more = add_more

    @decorator_with_options
    def dec(self, f, add=0):
        def wrapped(x):
            return f(x) + add + self.add_more

        return wrapped


def test_method_simple_warns():
    c = Cls(3)

    with pytest.warns(DeprecationError, match="cls.dec"):

        @c.dec
        def f(x):
            return x + 7

    assert f(42) == 52


def test_method_args():
    c = Cls(4)

    @c.dec(add=5)
    def g(x):
        return x + 1

    assert g(42) == 52


def test_method_direct_warns():
    c = Cls(5)
    with pytest.warns(DeprecationError, match="cls.dec"):
        p = c.dec(double)
    assert p(42) == 89


def test_method_indirect_ok():
    c = Cls(5)
    p = c.dec()(double)
    assert p(42) == 89


def test_method_direct_args_warns():
    c = Cls(6)
    with pytest.warns(DeprecationError, match="cls.dec"):
        q = c.dec(double, add=9)
    assert q(42) == 99


def test_method_indirect_args_ok():
    c = Cls(6)
    q = c.dec(add=9)(double)
    assert q(42) == 99
