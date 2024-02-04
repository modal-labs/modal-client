# Copyright Modal Labs 2022
import pytest

from modal import Dict, Stub


def test_dict_app(servicer, client):
    stub = Stub()
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


def test_dict_deploy(servicer, client):
    d = Dict.lookup("xyz", create_if_missing=True, client=client)
    d["xyz"] = 123
