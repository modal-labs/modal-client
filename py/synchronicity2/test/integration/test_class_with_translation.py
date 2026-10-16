# Copyright Modal Labs 2026
"""Integration tests for class_with_translation_impl.py support file."""

from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import class_with_translation

    node = class_with_translation.create_node(42)
    assert node.value == 42

    child = node.create_child(99)
    assert child.value == 99

    result_parent, result_child = class_with_translation.connect_nodes(node, child)
    assert result_parent.value == 42
    assert result_child.value == 99
    assert result_parent is node
    assert result_child is child

    node2 = class_with_translation.create_node(1)
    returned_node, _ = class_with_translation.connect_nodes(node2, node2)
    assert returned_node is node2

    _ = class_with_translation.WrappedTypeInConstructor(class_with_translation.create_node(10))


def test_pyright_implementation():
    import class_with_translation_impl

    check_pyright([Path(class_with_translation_impl.__file__)])


def test_pyright_wrapper():
    import class_with_translation

    check_pyright([Path(class_with_translation.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("class_with_translation_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
