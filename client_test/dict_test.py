# Copyright Modal Labs 2022

from modal import Dict, Stub


def test_dict(servicer, client):
    stub = Stub()
    stub.d = Dict.new()
    with stub.run(client=client):
        stub.d["foo"] = 42
        stub.d["foo"] += 5
        assert stub.d["foo"] == 47
