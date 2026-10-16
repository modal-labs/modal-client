# Copyright Modal Labs 2026
"""Integration tests for class_with_translation_impl.py support file.

Tests execution and type checking of generated code for classes requiring type translation.
"""

from pathlib import Path

from test.integration.test_utils import check_pyright, check_pyright_with_xfail


def test_runtime():
    # check that actual invokations of the wrappers work correctly
    import class_with_self_references

    a = class_with_self_references.SomeClass()
    assert a.accept_self(a) is a
    assert a.accept_self_by_name(a) is a

    sub = class_with_self_references.SomeSubclass()
    assert sub.accept_self(sub) is sub
    assert sub.accept_self_by_name(sub) is sub

    assert a.accept_self_by_name(sub)
    assert isinstance(class_with_self_references.SomeClass.create(), class_with_self_references.SomeClass)


def test_pyright_implementation():
    # check that the implementation type checks correctly
    import class_with_self_references_impl

    check_pyright([Path(class_with_self_references_impl.__file__)])


def test_pyright_wrapper():
    import class_with_self_references

    # check that the generated wrapper itself has correct types
    check_pyright([Path(class_with_self_references.__file__)])


def test_pyright_usage():
    # check that usage of the generated wrapper type checks correctly
    check_pyright_with_xfail("class_with_self_references_typecheck")
