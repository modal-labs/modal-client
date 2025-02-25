# Copyright Modal Labs 2022
import pytest
import random

import modal
from modal import Queue
from modal._serialization import (
    deserialize,
    deserialize_data_format,
    deserialize_proto_params,
    proto_to_python_payload,
    python_to_proto_payload,
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
    ["pydict", "params", "expected_bytes"],
    [
        (
            {"foo": "bar", "i": 5},
            [
                api_pb2.ClassParameterSpec(name="foo", type=api_pb2.PARAM_TYPE_STRING),
                api_pb2.ClassParameterSpec(name="i", type=api_pb2.PARAM_TYPE_INT),
            ],
            # only update this byte sequence if you are aware of the consequences of changing
            # serialization byte output - it could invalidate existing container pools for users
            # on redeployment, and possibly cause startup crashes if new containers can't
            # deserialize old proto parameters.
            b"\n\x0c\n\x03foo\x10\x01\x1a\x03bar\n\x07\n\x01i\x10\x02 \x05",
        )
    ],
)
def test_proto_serde_params_success(pydict, params, expected_bytes):
    serialized_params = serialize_proto_params(pydict, params)
    # it's important that the serialization doesn't change, since the serialized params bytes
    # are used as a key for the container pooling of parameterized services (classes)
    assert serialized_params == expected_bytes
    reconstructed = deserialize_proto_params(serialized_params, params)
    assert reconstructed == pydict


def test_proto_serde_failure_incomplete_params():
    # construct an incorrect serialization:
    incomplete_proto_params = api_pb2.ClassParameterSet(
        parameters=[api_pb2.ClassParameterValue(name="a", type=api_pb2.PARAM_TYPE_STRING, string_value="b")]
    )
    encoded_params = incomplete_proto_params.SerializeToString(deterministic=True)
    with pytest.raises(AttributeError, match="Constructor arguments don't match"):
        deserialize_proto_params(encoded_params, [api_pb2.ClassParameterSpec(name="x", type=api_pb2.PARAM_TYPE_STRING)])

    # TODO: add test for incorrect types


def _call(*args, **kwargs):
    return args, kwargs


@pytest.fixture()
def disable_pickle_payloads(monkeypatch):
    def bork():
        raise Exception("This test is expected to not use pickling")

    monkeypatch.setattr("modal._serialization.serialize", lambda _: bork())


@pytest.mark.parametrize(
    ["python_arg_kwargs", "expected_proto_bytes"],
    [
        (_call("foo"), b"\n\x0b\n\t\n\x00\x10\x01\x1a\x03foo\x12\x00"),  # positional args
        (_call(bar=3), b"\n\x00\x12\x0f\n\r\n\x03bar\x12\x06\n\x00\x10\x02 \x03"),
        (
            _call("foo", bar=2),
            b"\n\x0b\n\t\n\x00\x10\x01\x1a\x03foo\x12\x0f\n\r\n\x03bar\x12\x06\n\x00\x10\x02 \x02",
        ),  # mix
        (
            _call([1, 2]),
            b"\n\x18\n\x16\n\x00\x10\x042\x10\n\x06\n\x00\x10\x02 \x01\n\x06\n\x00\x10\x02 \x02\x12\x00",
        ),  # list
        (
            _call([1, "bar"]),
            b"\n\x1b\n\x19\n\x00\x10\x042\x13\n\x06\n\x00\x10\x02 \x01\n\t\n\x00\x10\x01\x1a\x03bar\x12\x00",
        ),  # mixed list
        (
            _call({"some_key": 123}),
            b"\n\x1c\n\x1a\n\x00\x10\x05:\x14\n\x12\n\x08some_key\x12\x06\n\x00\x10\x02 {\x12\x00",
        ),  # dict
    ],
)
@pytest.mark.usefixtures("disable_pickle_payloads")
def test_proto_serde_stability(python_arg_kwargs, expected_proto_bytes, client):
    # simulates a call from an older client (typically fewer supported types) to a newer
    proto_payload = python_to_proto_payload(*python_arg_kwargs)
    proto_bytes = proto_payload.SerializeToString(deterministic=True)
    assert proto_bytes == expected_proto_bytes  # possibly relax this to only enforce being able to decode?
    recovered_payload = api_pb2.Payload()
    recovered_payload.ParseFromString(proto_bytes)
    assert recovered_payload == proto_payload
    recovered_python_arg_kwargs = proto_to_python_payload(recovered_payload, client)
    assert recovered_python_arg_kwargs == python_arg_kwargs


def test_payload_modal_object(client):
    with modal.Dict.ephemeral(client=client) as dct:
        dct["foo"] = "bar"
        proto_payload = python_to_proto_payload(*_call(dct))
        proto_bytes = proto_payload.SerializeToString()
        recovered_payload = api_pb2.Payload()
        recovered_payload.ParseFromString(proto_bytes)
        assert recovered_payload == proto_payload
        recovered_python_arg_kwargs = proto_to_python_payload(recovered_payload, client)
        recovered_dct = recovered_python_arg_kwargs[0][0]
        assert recovered_dct.is_hydrated
        assert recovered_dct.object_id == dct.object_id
        assert dct["foo"] == "bar"
