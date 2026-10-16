# Copyright Modal Labs 2026
"""Integration tests for property_class_impl.py support file."""

from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import property_class

    settings = property_class.Settings("myapp", 5)

    # Read-only property
    assert settings.name == "myapp"

    # Read-write property
    assert settings.max_retries == 5
    settings.max_retries = 10
    assert settings.max_retries == 10

    # Computed read-only property
    assert settings.call_count == 0

    # Async method interacts with properties
    result = settings.do_work()
    assert result == "myapp: done (attempt 1/10)"
    assert settings.call_count == 1


def test_readonly_property_has_no_setter():
    import property_class

    settings = property_class.Settings("test")

    # Read-only properties should not have setters
    try:
        settings.name = "other"  # type: ignore
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_wrapped_type_property():
    """Properties returning/accepting wrapped types are translated correctly."""
    import property_class

    settings = property_class.Settings("test")

    # Getter returns wrapper type, not impl type
    tag = settings.tag
    assert isinstance(tag, property_class.Tag)
    assert tag.label == "default"

    # Setter accepts wrapper type, translates to impl type
    new_tag = property_class.Tag("important")
    settings.tag = new_tag
    assert settings.tag.label == "important"
    assert isinstance(settings.tag, property_class.Tag)


def test_pyright_implementation():
    import property_class_impl

    check_pyright([Path(property_class_impl.__file__)])


def test_pyright_wrapper():
    import property_class

    check_pyright([Path(property_class.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("property_class_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
