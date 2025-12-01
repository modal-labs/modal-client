# Copyright Modal Labs 2022
import pytest
import sys
import time

from modal import Dict
from modal._serialization import serialize
from modal.exception import AlreadyExistsError, DeprecationError, DeserializationError, InvalidError, NotFoundError
from modal_proto import api_pb2


def test_dict_named(servicer, client):
    name = "my-amazing-dict"
    d = Dict.from_name(name, create_if_missing=True)
    assert d.name == name

    d.hydrate(client)
    info = d.info()
    assert info.name == name
    assert info.created_by == servicer.default_username

    d["xyz"] = 123
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

    Dict.objects.delete("my-amazing-dict", client=client)
    with pytest.raises(NotFoundError):
        Dict.from_name("my-amazing-dict").hydrate(client)
    Dict.objects.delete("my-amazing-dict", client=client, allow_missing=True)


def test_dict_ephemeral(servicer, client):
    assert servicer.n_dict_heartbeats == 0
    with Dict.ephemeral(client=client, _heartbeat_sleep=1) as d:
        d["foo"] = 42
        assert d.len() == 1
        assert d["foo"] == 42
        time.sleep(1.5)  # Make time for 2 heartbeats
    assert servicer.n_dict_heartbeats == 2


def test_dict_lazy_hydrate_named(client, servicer):
    with servicer.intercept() as ctx:
        d = Dict.from_name("foo", create_if_missing=True, client=client)
        assert len(ctx.get_requests("DictGetOrCreate")) == 0  # sanity check that the get request is lazy
        d["foo"] = 42
        assert d["foo"] == 42
        assert len(ctx.get_requests("DictGetOrCreate")) == 1  # just sanity check that object is only hydrated once...


@pytest.mark.parametrize("name", ["has space", "has/slash", "a" * 65])
def test_invalid_name(servicer, client, name):
    with pytest.raises(InvalidError, match="Invalid Dict name"):
        Dict.from_name(name).hydrate(client)


def test_dict_update(servicer, client):
    with Dict.ephemeral(client=client, _heartbeat_sleep=1) as d:
        d.update({"foo": 1, "bar": 2}, foo=3, baz=4)
        items = list(d.items())
        assert sorted(items) == [("bar", 2), ("baz", 4), ("foo", 3)]


def test_dict_put_skip_if_exists(client):
    with Dict.ephemeral(client=client, _heartbeat_sleep=1) as d:
        assert d.put("foo", 1, skip_if_exists=True)
        assert not d.put("foo", 2, skip_if_exists=True)
        items = list(d.items())
        assert items == [("foo", 1)]


def test_dict_namespace_deprecated(servicer, client):
    # Test from_name with namespace parameter warns
    with pytest.warns(
        DeprecationError,
        match="The `namespace` parameter for `modal.Dict.from_name` is deprecated",
    ):
        Dict.from_name("test-dict", namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE)

    # Test that from_name without namespace parameter doesn't warn about namespace
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Dict.from_name("test-dict")
    # Filter out any unrelated warnings
    namespace_warnings = [w for w in record if "namespace" in str(w.message).lower()]
    assert len(namespace_warnings) == 0


def test_dict_list(servicer, client):
    for i in range(5):
        Dict.from_name(f"test-dict-{i}", create_if_missing=True).hydrate(client)
    if sys.platform == "win32":
        time.sleep(1 / 32)

    print(servicer.deployed_dicts)

    dict_list = Dict.objects.list(client=client)
    assert len(dict_list) == 5
    assert all(d.name.startswith("test-dict-") for d in dict_list)
    assert all(d.info().created_by == servicer.default_username for d in dict_list)

    dict_list = Dict.objects.list(max_objects=2, client=client)
    assert len(dict_list) == 2


def test_dict_create(servicer, client):
    Dict.objects.create(name="test-dict-create", client=client)
    Dict.from_name("test-dict-create").hydrate(client)
    with pytest.raises(AlreadyExistsError):
        Dict.objects.create(name="test-dict-create", client=client)
    Dict.objects.create(name="test-dict-create", allow_existing=True, client=client)
    with pytest.raises(InvalidError, match="Invalid Dict name"):
        Dict.objects.create(name="has space", client=client)


def test_dict_pop(servicer, client):
    with Dict.ephemeral(client=client, _heartbeat_sleep=1) as d:
        d["existing_key"] = "existing_value"

        assert d.pop("existing_key") == "existing_value"
        assert "existing_key" not in d

        assert d.pop("non_existent_key", "default_value") == "default_value"
        assert d.pop("non_existent_key", None) is None

        with pytest.raises(KeyError):
            d.pop("non_existent_key")


def test_dict_deserialization_errors(servicer, client):
    name = "test-dict-deserialization-errors"
    d = Dict.from_name(name, create_if_missing=True, client=client)

    key = "foo"
    d[key] = "good"
    value_error = rf"Failed to deserialize value for key {key!r} from Dict '{name}'"

    with servicer.intercept() as ctx:
        ctx.add_response("DictGet", api_pb2.DictGetResponse(found=True, value=b"bad"))
        with pytest.raises(DeserializationError, match=value_error):
            d.get(key)

    with servicer.intercept() as ctx:
        ctx.add_response("DictPop", api_pb2.DictPopResponse(found=True, value=b"bad"))
        with pytest.raises(DeserializationError, match=value_error):
            d.pop(key)

    async def bad_values(servicer, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.DictEntry(key=serialize("foo"), value=b"bad"))

    with servicer.intercept() as ctx:
        ctx.set_responder("DictContents", bad_values)
        with pytest.raises(DeserializationError, match=value_error):
            list(d.values())

    with servicer.intercept() as ctx:
        ctx.set_responder("DictContents", bad_values)
        with pytest.raises(DeserializationError, match=value_error):
            list(d.items())

    async def bad_keys(servicer, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.DictEntry(key=b"bad"))

    key_error = rf"Failed to deserialize a key from Dict '{name}'"

    with servicer.intercept() as ctx:
        ctx.set_responder("DictContents", bad_keys)
        error = rf"Failed to deserialize key from Dict '{name}'"
        with pytest.raises(DeserializationError, match=key_error):
            list(d.keys())

    with servicer.intercept() as ctx:
        ctx.set_responder("DictContents", bad_keys)
        with pytest.raises(DeserializationError, match=key_error):
            list(d.items())
