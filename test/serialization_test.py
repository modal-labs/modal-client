# Copyright Modal Labs 2022
import inspect
import pytest
import random
import typing

from modal import Queue
from modal._serialization import (
    apply_defaults,
    deserialize,
    deserialize_data_format,
    deserialize_proto_params,
    serialize,
    serialize_data_format,
    serialize_proto_params,
    signature_to_protobuf_schema,
    type_register,
    validate_parameter_values,
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
        validate_parameter_values({"a": "b"}, schema)

    with pytest.raises(TypeError, match="Expected str, got bytes"):
        validate_parameter_values({"x": b"b"}, schema)

    with pytest.raises(InvalidError, match="provided but are not present in the schema"):
        validate_parameter_values({"x": "y", "a": "b"}, schema)

    # this should pass:
    validate_parameter_values({"x": "y"}, schema)


def test_apply_defaults():
    schema = [
        api_pb2.ClassParameterSpec(name="x", type=api_pb2.PARAM_TYPE_STRING, has_default=True, string_default="hello")
    ]
    assert apply_defaults({}, schema) == {"x": "hello"}
    assert apply_defaults({"x": "goodbye"}, schema) == {"x": "goodbye"}
    assert apply_defaults({"y": "goodbye"}, schema) == {"x": "hello", "y": "goodbye"}


def test_non_implemented_proto_type():
    with pytest.raises(InvalidError, match="No payload handler implemented for payload type PARAM_TYPE_UNKNOWN"):
        type_register.get_decoder(api_pb2.PARAM_TYPE_UNKNOWN)

    with pytest.raises(InvalidError, match="recognize payload type 1000"):
        type_register.get_decoder(1000)  # type: ignore


def test_schema_extraction_unknown():
    def with_empty(a): ...

    def with_any(a: typing.Any): ...

    class Custom:
        pass

    def with_custom(a: Custom): ...

    for func in [with_empty, with_any, with_custom]:
        fields = signature_to_protobuf_schema(inspect.signature(func))
        assert fields == [
            api_pb2.ClassParameterSpec(
                name="a", has_default=False, full_type=api_pb2.GenericPayloadType(type=api_pb2.PARAM_TYPE_UNKNOWN)
            )
        ]

    def with_default(a=5): ...

    fields = signature_to_protobuf_schema(inspect.signature(with_default))
    assert fields == [
        api_pb2.ClassParameterSpec(
            name="a",
            full_type=api_pb2.GenericPayloadType(type=api_pb2.PARAM_TYPE_UNKNOWN),
            has_default=True,
            default_value=api_pb2.ClassParameterValue(
                type=api_pb2.PARAM_TYPE_INT,
                int_value=5,
            ),
        )
    ]


def test_schema_extraction_int():
    def f(int_value: int = 1337): ...

    sig = inspect.signature(f)
    (int_spec,) = signature_to_protobuf_schema(sig)
    assert int_spec == api_pb2.ClassParameterSpec(
        name="int_value",
        type=api_pb2.PARAM_TYPE_INT,
        full_type=api_pb2.GenericPayloadType(type=api_pb2.PARAM_TYPE_INT),
        has_default=True,
        default_value=api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_INT, int_value=1337),
        int_default=1337,
    )


def test_schema_extraction_str():
    def foo(str_value: str = "foo"):
        pass

    (str_spec,) = signature_to_protobuf_schema(inspect.signature(foo))
    assert str_spec == api_pb2.ClassParameterSpec(
        name="str_value",
        type=api_pb2.PARAM_TYPE_STRING,
        full_type=api_pb2.GenericPayloadType(type=api_pb2.PARAM_TYPE_STRING),
        has_default=True,
        default_value=api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_STRING, string_value="foo"),
        string_default="foo",
    )


def test_schema_extraction_bytes():
    def foo(a: bytes = b"foo"):
        pass

    (bytes_spec,) = signature_to_protobuf_schema(inspect.signature(foo))
    assert bytes_spec == api_pb2.ClassParameterSpec(
        name="a",
        type=api_pb2.PARAM_TYPE_BYTES,  # for backward compatibility
        has_default=True,
        bytes_default=b"foo",  # for backward compatibility
        full_type=api_pb2.GenericPayloadType(type=api_pb2.PARAM_TYPE_BYTES),
        default_value=api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_BYTES, bytes_value=b"foo"),
    )


def test_schema_extraction_list():
    def new_f(simple_list: list[int]): ...
    def old_f(simple_list: typing.List[int]): ...

    for f in [new_f, old_f]:
        (list_spec,) = signature_to_protobuf_schema(inspect.signature(f))
        assert list_spec == api_pb2.ClassParameterSpec(
            name="simple_list",
            full_type=api_pb2.GenericPayloadType(
                type=api_pb2.PARAM_TYPE_LIST, sub_types=[api_pb2.GenericPayloadType(type=api_pb2.PARAM_TYPE_INT)]
            ),
            has_default=False,
        )


def test_schema_extraction_nested_list():
    def f(nested_list: list[list[bytes]]): ...

    (list_spec,) = signature_to_protobuf_schema(inspect.signature(f))
    assert list_spec == api_pb2.ClassParameterSpec(
        name="nested_list",
        full_type=api_pb2.GenericPayloadType(
            type=api_pb2.PARAM_TYPE_LIST,
            sub_types=[
                api_pb2.GenericPayloadType(
                    type=api_pb2.PARAM_TYPE_LIST, sub_types=[api_pb2.GenericPayloadType(type=api_pb2.PARAM_TYPE_BYTES)]
                )
            ],
        ),
        has_default=False,
    )
