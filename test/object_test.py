# Copyright Modal Labs 2022
import pytest

from modal import Image, Queue, Secret, Volume
from modal._object import _Object
from modal.dict import Dict, _Dict
from modal.exception import InvalidError
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


def test_on_demand_hydration(client):
    obj = Dict.from_name("test-dict", create_if_missing=True).hydrate(client)
    assert obj.object_id is not None


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


@pytest.mark.parametrize(
    "Kls, msg",
    [
        (Image, r"Please use `Image\.from_id` instead"),
        (Dict, r"Please use `Dict\.from_name`, `Dict\.ephemeral` instead"),
        (Volume, r"Please use `Volume\.from_name`, `Volume\.ephemeral` instead"),
        (Queue, r"Please use `Queue\.from_name`, `Queue\.ephemeral` instead"),
        (Secret, r"Please use `Secret\.from_name` instead"),
    ],
)
def test_improve_error_messaage(Kls, msg):
    with pytest.raises(InvalidError, match=msg):
        _ = Kls()
