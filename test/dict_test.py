# Copyright Modal Labs 2022
import pytest
import time

from modal import Dict
from modal.exception import InvalidError, NotFoundError


def test_dict_app(servicer, client):
    d = Dict.lookup("my-amazing-dict", {"xyz": 123}, create_if_missing=True, client=client)
    d["foo"] = 42
    d["foo"] += 5
    assert d["foo"] == 47
    assert d.len() == 2

    assert sorted(d.keys()) == ["foo", "xyz"]
    assert sorted(d.values()) == [47, 123]
    assert sorted(d.items()) == [("foo", 47), ("xyz", 123)]

    d.clear()
    assert d.len() == 0
    with pytest.raises(KeyError):
        _ = d["foo"]

    assert d.get("foo", default=True)
    d["foo"] = None
    assert d["foo"] is None

    Dict.delete("my-amazing-dict", client=client)
    with pytest.raises(NotFoundError):
        Dict.lookup("my-amazing-dict", client=client)


def test_dict_ephemeral(servicer, client):
    assert servicer.n_dict_heartbeats == 0
    with Dict.ephemeral({"bar": 123}, client=client, _heartbeat_sleep=1) as d:
        d["foo"] = 42
        assert d.len() == 2
        assert d["foo"] == 42
        assert d["bar"] == 123
        time.sleep(1.5)  # Make time for 2 heartbeats
    assert servicer.n_dict_heartbeats == 2


def test_dict_lazy_hydrate_named(set_env_client, servicer):
    with servicer.intercept() as ctx:
        d = Dict.from_name("foo", create_if_missing=True)
        assert len(ctx.get_requests("DictGetOrCreate")) == 0  # sanity check that the get request is lazy
        d["foo"] = 42
        assert d["foo"] == 42
        assert len(ctx.get_requests("DictGetOrCreate")) == 1  # just sanity check that object is only hydrated once...


@pytest.mark.parametrize("name", ["has space", "has/slash", "a" * 65])
def test_invalid_name(servicer, client, name):
    with pytest.raises(InvalidError, match="Invalid Dict name"):
        Dict.lookup(name)
