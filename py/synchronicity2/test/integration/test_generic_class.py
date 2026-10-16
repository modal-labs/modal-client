# Copyright Modal Labs 2026
"""Integration tests for generic_class_impl.py support file."""

from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import generic_class

    container_42 = generic_class.SomeContainer(generic_class.WrappedType(42))
    container_43 = generic_class.SomeContainer(generic_class.WrappedType(43))
    assert container_42.get().val == 42
    assert container_43.get().val == 43

    container_100 = generic_class.SomeContainer(generic_class.WrappedType(100))
    assert container_100.get().val == 100
    container_100.set(generic_class.WrappedType(200))
    assert container_100.get().val == 200

    def add(x: int, y: int) -> int:
        return x + y

    wrapper = generic_class.FunctionWrapper(add)
    assert wrapper.call(5, 10) == 15

    def add2(x: int, y: int) -> int:
        return x + y

    wrapper2 = generic_class.FunctionWrapper(add2)
    assert wrapper2._impl_instance.f == add2

    container3 = generic_class.SomeContainer(generic_class.WrappedType(3))
    assert hasattr(container3, "get")
    assert hasattr(container3, "set")
    wrapper3 = generic_class.FunctionWrapper(lambda x: x)
    assert hasattr(wrapper3, "call")


def test_pyright_implementation():
    import generic_class_impl

    check_pyright([Path(generic_class_impl.__file__)])


def test_pyright_wrapper():
    import generic_class

    check_pyright([Path(generic_class.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("generic_class_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
