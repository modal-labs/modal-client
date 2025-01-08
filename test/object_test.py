# Copyright Modal Labs 2022
import pytest

from modal import Secret
from modal.dict import _Dict
from modal.exception import InvalidError
from modal.object import _Object
from modal.queue import _Queue


def test_new_hydrated(client):
    assert isinstance(_Dict._new_hydrated("di-123", client, None), _Dict)
    assert isinstance(_Queue._new_hydrated("qu-123", client, None), _Queue)

    with pytest.raises(InvalidError):
        _Queue._new_hydrated("di-123", client, None)  # Wrong prefix for type

    assert isinstance(_Object._new_hydrated("qu-123", client, None), _Queue)
    assert isinstance(_Object._new_hydrated("di-123", client, None), _Dict)

    with pytest.raises(InvalidError):
        _Object._new_hydrated("xy-123", client, None)


def test_constructor():
    with pytest.raises(InvalidError) as excinfo:
        Secret({"foo": 123})

    assert "Secret" in str(excinfo.value)
    assert "constructor" in str(excinfo.value)


def test_types():
    assert _Object._get_type_from_id("di-123") == _Dict
    assert _Dict._is_id_type("di-123")
    assert not _Dict._is_id_type("qu-123")
    assert _Queue._is_id_type("qu-123")
    assert not _Queue._is_id_type("di-123")
