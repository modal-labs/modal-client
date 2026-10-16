# Copyright Modal Labs 2026
"""Integration tests for functions_with_typevars_impl.py support file."""

from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import functions_with_typevars

    assert hasattr(functions_with_typevars, "SomeClass")

    container = functions_with_typevars.Container()
    some_obj = functions_with_typevars.SomeClass()
    result = container.tuple_to_list((some_obj, some_obj))
    assert len(result) == 2
    for entry in result:
        assert isinstance(entry, functions_with_typevars.SomeClass)


def test_pyright_implementation():
    import functions_with_typevars_impl

    check_pyright([Path(functions_with_typevars_impl.__file__)])


def test_pyright_wrapper():
    import functions_with_typevars

    check_pyright([Path(functions_with_typevars.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("functions_with_typevars_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
