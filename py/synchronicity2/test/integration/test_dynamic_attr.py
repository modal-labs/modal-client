# Copyright Modal Labs 2026
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import dynamic_attr

    owner = dynamic_attr.DynamicOwner(dynamic_attr.Payload(7))
    unmarked_owner = dynamic_attr.UnmarkedDynamicOwner(dynamic_attr.Payload(11))

    payload = owner.payload
    assert isinstance(payload, dynamic_attr.Payload)
    assert payload.value == 7
    assert owner.count == 3

    unmarked_payload = unmarked_owner.payload
    assert type(unmarked_payload).__module__ == "dynamic_attr_impl"
    assert unmarked_payload.value == 11
    assert unmarked_owner.count == 3


def test_pyright_wrapper():
    import dynamic_attr

    check_pyright([Path(dynamic_attr.__file__)])


def test_pyright_usage():
    from importlib.util import find_spec

    spec = find_spec("dynamic_attr_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
