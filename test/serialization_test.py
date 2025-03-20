# Copyright Modal Labs 2022
import pytest
import random

from modal import Queue
from modal._serialization import (
    apply_defaults,
    deserialize,
    deserialize_data_format,
    deserialize_proto_params,
    proto_type_enum_to_payload_handler,
    serialize,
    serialize_data_format,
    serialize_proto_params,
    validate_params,
)
from modal._utils.rand_pb_testing import rand_pb
from modal.exception import DeserializationError, InvalidError
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
    ["pydict", "expected_bytes"],
    [
        (
            {"foo": "bar", "i": 5},
            # only update this byte sequence if you are aware of the consequences of changing
            # serialization byte output - it could invalidate existing container pools for users
            # on redeployment, and possibly cause startup crashes if new containers can't
            # deserialize old proto parameters.
            b"\n\x0c\n\x03foo\x10\x01\x1a\x03bar\n\x07\n\x01i\x10\x02 \x05",
        ),
        ({"x": b"\x00"}, b"\n\x08\n\x01x\x10\x042\x01\x00"),
    ],
)
def test_proto_serde_params_success(pydict, expected_bytes):
    serialized_params = serialize_proto_params(pydict)
    # it's important that the serialization doesn't change, since the serialized params bytes
    # are used as a key for the container pooling of parameterized services (classes)
    assert serialized_params == expected_bytes
    reconstructed = deserialize_proto_params(serialized_params)
    assert reconstructed == pydict


def test_proto_serde_failure_incomplete_params():
    # construct an incorrect serialization:
    schema = [api_pb2.ClassParameterSpec(name="x", type=api_pb2.PARAM_TYPE_STRING)]
    with pytest.raises(InvalidError, match="Missing required parameter: x"):
        validate_params({"a": "b"}, schema)

    with pytest.raises(TypeError, match="Expected str, got bytes"):
        validate_params({"x": b"b"}, schema)

    with pytest.raises(InvalidError, match="provided but are not present in the schema"):
        validate_params({"x": "y", "a": "b"}, schema)

    # this should pass:
    validate_params({"x": "y"}, schema)


def test_apply_defaults():
    schema = [
        api_pb2.ClassParameterSpec(name="x", type=api_pb2.PARAM_TYPE_STRING, has_default=True, string_default="hello")
    ]
    assert apply_defaults({}, schema) == {"x": "hello"}
    assert apply_defaults({"x": "goodbye"}, schema) == {"x": "goodbye"}
    assert apply_defaults({"y": "goodbye"}, schema) == {"x": "hello", "y": "goodbye"}


def test_non_implemented_proto_type():
    proto_type_enum_to_payload_handler(api_pb2.PARAM_TYPE_UNKNOWN)
