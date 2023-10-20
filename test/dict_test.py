# Copyright Modal Labs 2022
import pytest

from modal import Dict, Stub


def test_dict(servicer, client):
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
