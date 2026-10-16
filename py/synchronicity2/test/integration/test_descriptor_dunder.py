# Copyright Modal Labs 2026
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import descriptor_dunder

    payload = descriptor_dunder.Payload(3)
    descriptor = descriptor_dunder.Descriptor(payload)

    class Holder:
        attr = descriptor

    class_attr = Holder.attr
    assert isinstance(class_attr, descriptor_dunder.Descriptor)
    assert class_attr is descriptor

    instance_attr = Holder().attr
    assert isinstance(instance_attr, descriptor_dunder.Payload)
    assert instance_attr.value == 3


def test_pyright_wrapper():
    import descriptor_dunder

    check_pyright([Path(descriptor_dunder.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("descriptor_dunder_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
