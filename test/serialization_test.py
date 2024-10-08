# Copyright Modal Labs 2022
import pytest
import random

from modal import Queue
from modal._serialization import (
    deserialize,
    deserialize_data_format,
    deserialize_proto_params,
    payload_handler,
    serialize,
    serialize_data_format,
    serialize_proto_params,
)
from modal._utils.rand_pb_testing import rand_pb
from modal.exception import DeserializationError
from modal_proto import api_pb2

from .supports.skip import skip_old_py


@pytest.mark.asyncio
async def test_roundtrip(servicer, client):
    async with Queue.ephemeral(client=client) as q:
        data = serialize(q)
        # TODO: strip synchronizer reference from synchronicity entities!
        assert len(data) < 350  # Used to be 93...
        # Note: if this blows up significantly, it's most likely because
        # cloudpickle can't find a class in the global scope. When this
        # happens, it tries to serialize the entire class along with the
        # object. The reason it doesn't find the class in the global scope
        # is most likely because the name doesn't match. To fix this, make
        # sure that cls.__name__ (which is something synchronicity sets)
        # is the same as the symbol defined in the global scope.
        q_roundtrip = deserialize(data, client)
        assert isinstance(q_roundtrip, Queue)
        assert q.object_id == q_roundtrip.object_id


@skip_old_py("random.randbytes() was introduced in python 3.9", (3, 9))
@pytest.mark.asyncio
async def test_asgi_roundtrip():
    rand = random.Random(42)
    for _ in range(10000):
        msg = rand_pb(api_pb2.Asgi, rand)
        buf = msg.SerializeToString()
        asgi_obj = deserialize_data_format(buf, api_pb2.DATA_FORMAT_ASGI, None)
        assert asgi_obj is None or (isinstance(asgi_obj, dict) and asgi_obj["type"])
        buf = serialize_data_format(asgi_obj, api_pb2.DATA_FORMAT_ASGI)
        asgi_obj_roundtrip = deserialize_data_format(buf, api_pb2.DATA_FORMAT_ASGI, None)
        assert asgi_obj == asgi_obj_roundtrip


def test_deserialization_error(client):
    # Curated object that we should not be able to deserialize
    obj = (
        b"\x80\x04\x95(\x00\x00\x00\x00\x00\x00\x00\x8c\x17"
        b"undeserializable_module\x94\x8c\x05Dummy\x94\x93\x94)\x81\x94."
    )
    with pytest.raises(DeserializationError, match="'undeserializable_module' .+ local environment"):
        deserialize(obj, client)


@pytest.mark.parametrize(
    ["pydict", "params"],
    [
        (
            {"foo": "bar", "i": 5},
            [
                api_pb2.ClassParameterSpec(name="foo", type=api_pb2.PARAM_TYPE_STRING),
                api_pb2.ClassParameterSpec(name="i", type=api_pb2.PARAM_TYPE_INT),
            ],
        )
    ],
)
def test_proto_serde_params_success(pydict, params):
    serialized_params = serialize_proto_params(pydict, params)
    reconstructed = deserialize_proto_params(serialized_params, params)
    assert reconstructed == pydict


def test_proto_serde_failure_incomplete_params():
    # construct an incorrect serialization:
    incomplete_proto_params = api_pb2.ClassParameterSet(
        parameters=[api_pb2.ClassParameterValue(name="a", type=api_pb2.PARAM_TYPE_STRING, string_value="b")]
    )
    encoded_params = incomplete_proto_params.SerializeToString(deterministic=True)
    with pytest.raises(AttributeError):
        deserialize_proto_params(encoded_params, [api_pb2.ClassParameterSpec(name="x", type=api_pb2.PARAM_TYPE_STRING)])

    # TODO: add test for incorrect types


def test_payload_value_str(client):
    pv = payload_handler.serialize("hello")
    assert isinstance(pv, api_pb2.PayloadValue)
    s = payload_handler.deserialize(pv, client)
    assert s == "hello"


def test_payload_value_int(client):
    pv = payload_handler.serialize(4242)
    assert isinstance(pv, api_pb2.PayloadValue)
    i = payload_handler.deserialize(pv, client)
    assert i == 4242


def test_payload_value_bool(client):
    pv = payload_handler.serialize(True)
    assert isinstance(pv, api_pb2.PayloadValue)
    b = payload_handler.deserialize(pv, client)
    assert b == True


def test_payload_value_float(client):
    pv = payload_handler.serialize(4242.0)
    assert isinstance(pv, api_pb2.PayloadValue)
    f = payload_handler.deserialize(pv, client)
    assert f == 4242.0


def test_payload_value_bytes(client):
    pv = payload_handler.serialize(b"foo")
    assert isinstance(pv, api_pb2.PayloadValue)
    b = payload_handler.deserialize(pv, client)
    assert b == b"foo"


def test_payload_value_none(client):
    pv = payload_handler.serialize(None)
    assert isinstance(pv, api_pb2.PayloadValue)
    n = payload_handler.deserialize(pv, client)
    assert n is None


def test_payload_value_list(client):
    pv = payload_handler.serialize(["foo", 123])
    assert isinstance(pv, api_pb2.PayloadValue)
    l = payload_handler.deserialize(pv, client)
    assert l == ["foo", 123]


def test_payload_value_dict(client):
    pv = payload_handler.serialize({"foo": "bar", "baz": 123})
    assert isinstance(pv, api_pb2.PayloadValue)
    d = payload_handler.deserialize(pv, client)
    assert d == {"foo": "bar", "baz": 123}


class C:
    ...


def test_payload_value_pickle(client):
    c = C()
    pv = payload_handler.serialize(c)
    assert isinstance(pv, api_pb2.PayloadValue)
    o = payload_handler.deserialize(pv, client)
    assert isinstance(o, C)
