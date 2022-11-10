# Copyright Modal Labs 2022
from modal_utils.decorator_utils import decorator_with_options


@decorator_with_options
def dec(f, add=0):
    def wrapped(x):
        return f(x) + add

    return wrapped


def test_simple():
    @dec
    def f(x):
        return x + 7

    assert f(42) == 49


def test_args():
    @dec(add=5)
    def g(x):
        return x + 1

    assert g(42) == 48


def double(x):
    return 2 * x


def test_direct():
    p = dec(double)
    assert p(42) == 84


def test_direct_args():
    q = dec(double, add=9)
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


def test_method_simple():
    c = Cls(3)

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


def test_method_direct():
    c = Cls(5)
    p = c.dec(double)
    assert p(42) == 89


def test_method_direct_args():
    c = Cls(6)
    q = c.dec(double, add=9)
    assert q(42) == 99
