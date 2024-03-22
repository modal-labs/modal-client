# Copyright Modal Labs 2022
import pytest
import time

from modal import Dict, Stub
from modal.exception import DeprecationError


def test_dict_app(servicer, client):
    stub = Stub()
    with pytest.warns(DeprecationError):
        stub.d = Dict.new()
    with stub.run(client=client):
        stub.d["foo"] = 42
        stub.d["foo"] += 5
        assert stub.d["foo"] == 47
        assert stub.d.len() == 1

        stub.d.clear()
        assert stub.d.len() == 0
        with pytest.raises(KeyError):
            _ = stub.d["foo"]

        assert stub.d.get("foo", default=True)
        stub.d["foo"] = None
        assert stub.d["foo"] is None


def test_dict_lookup(servicer, client):
    d = Dict.lookup("xyz", {"foo": "bar"}, create_if_missing=True, client=client)
    d["xyz"] = 123
    assert d.len() == 2
    assert d["foo"] == "bar"


def test_dict_ephemeral(servicer, client):
    assert servicer.n_dict_heartbeats == 0
    with Dict.ephemeral({"bar": 123}, client=client, _heartbeat_sleep=1) as d:
        d["foo"] = 42
        assert d.len() == 2
        assert d["foo"] == 42
        assert d["bar"] == 123
        time.sleep(1.5)  # Make time for 2 heartbeats
    assert servicer.n_dict_heartbeats == 2
