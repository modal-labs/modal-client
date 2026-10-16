# Copyright Modal Labs 2026
"""Integration tests for classproperty_class_impl.py support file."""

from importlib.util import find_spec
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import classproperty_class

    assert isinstance(classproperty_class.Service.manager, classproperty_class.Manager)
    assert classproperty_class.Service.manager is classproperty_class.Service.manager
    assert classproperty_class.Service.manager.label == "primary"
    assert classproperty_class.Service.manager.describe() == "manager:primary"
    assert classproperty_class.Service.default_name == "service"

    service = classproperty_class.Service()
    assert service.manager is classproperty_class.Service.manager
    assert service.default_name == "service"


def test_pyright_implementation():
    import classproperty_class_impl

    check_pyright([Path(classproperty_class_impl.__file__)])


def test_pyright_wrapper():
    import classproperty_class

    check_pyright([Path(classproperty_class.__file__)])


def test_pyright_usage():
    spec = find_spec("classproperty_class_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
