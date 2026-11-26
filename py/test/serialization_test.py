# Copyright Modal Labs 2022
import inspect
import io
import pickle
import pytest
import random
import typing

from grpclib import GRPCError, Status

from modal import Queue, Sandbox
from modal._serialization import (
    apply_defaults,
    deserialize,
    deserialize_data_format,
    deserialize_proto_params,
    get_callable_schema,
    serialize,
    serialize_data_format,
    serialize_proto_params,
    signature_to_parameter_specs,
    validate_parameter_values,
)
from modal._type_manager import parameter_serde_registry
from modal._utils.async_utils import synchronize_api
from modal._utils.rand_pb_testing import rand_pb
from modal.exception import DeserializationError, InvalidError, NotFoundError
from modal_proto import api_pb2


async def _deserialize_on_synchronizer(data, client):
    return deserialize(data, client)


deserialize_on_synchronizer = synchronize_api(_deserialize_on_synchronizer)


def _serialize_sandbox_with_metadata(sandbox, metadata):
    class SandboxPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if obj is sandbox:
                return (sandbox.object_id, "o", metadata)
            return None

    buf = io.BytesIO()
    SandboxPickler(buf).dump(sandbox)
    return buf.getvalue()


class ObjectMetadataUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return pid


@pytest.mark.parametrize(
    "sandbox_id,wait_rpc,task_rpc",
    [
        ("sb-nGEijt9WbBMlGrsPH9FOaC", "SandboxWait", "SandboxGetTaskId"),
        ("sb-01ARZ3NDEKTSV4RRFFQ69G5FAV", "SandboxWaitV2", "SandboxGetTaskIdV2"),
    ],
)
def test_fresh_sandbox_serialization_and_logs(client, servicer, sandbox_id, wait_rpc, task_rpc):
    sb = Sandbox.from_id(sandbox_id)
    assert not sb._is_hydrated
    with servicer.intercept() as ctx:
        data = serialize(sb)
        assert ObjectMetadataUnpickler(io.BytesIO(data)).load() == (sandbox_id, "o", api_pb2.SandboxHandleMetadata())
        restored = deserialize_on_synchronizer(data, client)
        assert restored.object_id == sandbox_id
        assert restored.client is client
        assert restored._is_hydrated
        assert ctx.calls == []
        ctx.add_response(
            wait_rpc,
            api_pb2.SandboxWaitResponse(metadata=api_pb2.SandboxHandleMetadata(app_id="ap-lazy")),
        )
        ctx.add_response(task_rpc, api_pb2.SandboxGetTaskIdResponse(task_id="ta-lazy"))
        ctx.add_response(
            "AppFetchLogs",
            api_pb2.AppFetchLogsResponse(
                batches=[api_pb2.TaskLogsBatch(task_id="ta-lazy", items=[api_pb2.TaskLogs(data="lazy log\n")])]
            ),
        )
        assert [str(entry) for entry in restored.logs.tail(1)] == ["lazy log\n"]
        (req,) = ctx.get_requests("AppFetchLogs")
        assert req.app_id == "ap-lazy"
        assert len(ctx.get_requests(wait_rpc)) == 1


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


@pytest.mark.parametrize(
    ("sandbox_id", "is_v2"),
    [
        ("sb-nGEijt9WbBMlGrsPH9FOaC", False),
        ("sb-01ARZ3NDEKTSV4RRFFQ69G5FAV", True),
    ],
    ids=["v1", "v2"],
)
@pytest.mark.parametrize("has_result", [False, True])
def test_sandbox_serialization_payload(servicer, client, sandbox_id, is_v2, has_result):
    result = (
        api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE, exitcode=42) if has_result else None
    )

    async def wait_handler(self, stream):
        await stream.recv_message()
        await stream.send_message(
            api_pb2.SandboxWaitResponse(
                result=result,
                metadata=api_pb2.SandboxHandleMetadata(app_id="ap-serialized-sandbox", result=result),
            )
        )

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxWaitV2" if is_v2 else "SandboxWait", wait_handler)
        sandbox = Sandbox.from_id(sandbox_id, client=client)
        sandbox.poll()

    assert ObjectMetadataUnpickler(io.BytesIO(serialize(sandbox))).load() == (
        sandbox_id,
        "o",
        api_pb2.SandboxHandleMetadata(app_id="ap-serialized-sandbox", result=result),
    )


@pytest.mark.parametrize(
    ("sandbox_id", "is_v2"),
    [
        ("sb-nGEijt9WbBMlGrsPH9FOaC", False),
        ("sb-01ARZ3NDEKTSV4RRFFQ69G5FAV", True),
    ],
    ids=["v1", "v2"],
)
@pytest.mark.parametrize("has_result", [False, True])
@pytest.mark.parametrize("operation", ["poll", "logs", "detached_logs"])
@pytest.mark.parametrize("metadata_kind", ["complete", "id_only", "missing_app_id"])
def test_sandbox_metadata_roundtrip(servicer, client, sandbox_id, is_v2, has_result, operation, metadata_kind):
    servicer.sandbox_app_id = "ap-from-main"
    result = (
        api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE, exitcode=42) if has_result else None
    )

    async def wait_handler(self, stream):
        await stream.recv_message()
        await stream.send_message(
            api_pb2.SandboxWaitResponse(
                result=result,
                metadata=api_pb2.SandboxHandleMetadata(app_id=servicer.sandbox_app_id, result=result),
            )
        )

    sandbox = Sandbox.from_id(sandbox_id, client=client)
    metadata: api_pb2.SandboxHandleMetadata | None = api_pb2.SandboxHandleMetadata(app_id="ap-from-main", result=result)
    if metadata_kind == "id_only":
        metadata = None
    elif metadata_kind == "missing_app_id":
        metadata = api_pb2.SandboxHandleMetadata(result=result)
    data = _serialize_sandbox_with_metadata(sandbox, metadata)

    async def get_task_id(self, stream):
        await stream.recv_message()
        await stream.send_message(api_pb2.SandboxGetTaskIdResponse(task_id="ta-serialized"))

    async def fetch_logs(self, stream):
        req = await stream.recv_message()
        assert req.app_id == "ap-from-main"
        await stream.send_message(
            api_pb2.AppFetchLogsResponse(
                batches=[
                    api_pb2.TaskLogsBatch(task_id="ta-serialized", items=[api_pb2.TaskLogs(data="serialized log\n")])
                ]
            )
        )

    wait_rpc = "SandboxWaitV2" if is_v2 else "SandboxWait"
    with servicer.intercept() as ctx:
        ctx.set_responder(wait_rpc, wait_handler)
        ctx.set_responder("SandboxGetTaskIdV2" if is_v2 else "SandboxGetTaskId", get_task_id)
        ctx.set_responder("AppFetchLogs", fetch_logs)
        sandbox_roundtrip = deserialize_on_synchronizer(data, client)
        assert isinstance(sandbox_roundtrip, Sandbox)
        assert sandbox_roundtrip.object_id == sandbox.object_id
        assert not ctx.get_requests(wait_rpc)
        cached_result = has_result and metadata is not None
        assert sandbox_roundtrip.returncode == (42 if cached_result else None)
        if operation == "detached_logs":
            sandbox_roundtrip.detach()
        if operation == "poll":
            assert sandbox_roundtrip.poll() == (42 if has_result else None)
            needs_lookup = not cached_result
        else:
            assert [str(entry) for entry in sandbox_roundtrip.logs.tail(1)] == ["serialized log\n"]
            assert sandbox_roundtrip.returncode == (42 if has_result else None)
            needs_lookup = metadata is None or not metadata.app_id
        requests = ctx.get_requests(wait_rpc)
        assert len(requests) == int(needs_lookup)
        if needs_lookup:
            [req] = requests
            assert req.sandbox_id == sandbox_id
            assert req.timeout == 0


@pytest.mark.parametrize("sandbox_id", ["sb-nGEijt9WbBMlGrsPH9FOaC", "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"])
def test_sandbox_metadata_preserves_empty_result(client, sandbox_id):
    metadata = api_pb2.SandboxHandleMetadata(app_id="ap-main", result=api_pb2.GenericResult())
    sandbox = Sandbox.from_id(sandbox_id, client=client)
    restored = deserialize_on_synchronizer(_serialize_sandbox_with_metadata(sandbox, metadata), client)
    _, _, restored_metadata = ObjectMetadataUnpickler(io.BytesIO(serialize(restored))).load()
    assert restored_metadata.HasField("result")
    assert restored_metadata.SerializeToString() == metadata.SerializeToString()


@pytest.mark.parametrize(
    ("sandbox_id", "wait_rpc"),
    [("sb-nGEijt9WbBMlGrsPH9FOaC", "SandboxWait"), ("sb-01ARZ3NDEKTSV4RRFFQ69G5FAV", "SandboxWaitV2")],
)
@pytest.mark.parametrize("operation", ["poll", "wait"])
def test_sandbox_empty_result_observes_exit(servicer, client, sandbox_id, wait_rpc, operation):
    sandbox = Sandbox.from_id(sandbox_id, client=client)
    metadata = api_pb2.SandboxHandleMetadata(app_id="ap-main", result=api_pb2.GenericResult())
    restored = deserialize_on_synchronizer(_serialize_sandbox_with_metadata(sandbox, metadata), client)
    assert restored.returncode is None

    async def exited(servicer, stream):
        await stream.recv_message()
        await stream.send_message(
            api_pb2.SandboxWaitResponse(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE, exitcode=42)
            )
        )

    with servicer.intercept() as ctx:
        ctx.set_responder(wait_rpc, exited)
        if operation == "poll":
            assert restored.poll() == 42
        else:
            restored.wait()
        assert restored.returncode == 42
        assert len(ctx.get_requests(wait_rpc)) == 1


@pytest.mark.parametrize("sandbox_id", ["sb-nGEijt9WbBMlGrsPH9FOaC", "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"])
@pytest.mark.parametrize("operation", ["poll", "detached_logs"])
def test_deserialized_sandbox_lookup_error(servicer, client, sandbox_id, operation):
    sandbox = Sandbox.from_id(sandbox_id, client=client)
    data = _serialize_sandbox_with_metadata(sandbox, None)

    async def missing(self, stream):
        raise GRPCError(Status.NOT_FOUND, "unknown sandbox")

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxWait", missing)
        ctx.set_responder("SandboxWaitV2", missing)
        restored = deserialize_on_synchronizer(data, client)
        with pytest.raises(NotFoundError, match="unknown sandbox"):
            if operation == "poll":
                restored.poll()
            else:
                restored.detach()
                list(restored.logs.tail(1))


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

    with pytest.raises(InvalidError, match="provided but are not defined"):
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
    with pytest.raises(InvalidError, match="No class parameter decoder implemented for type PARAM_TYPE_LIST"):
        # This tests if attempt to get the manager for a type we don't support, like list
        parameter_serde_registry.decode(api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_LIST))

    with pytest.raises(InvalidError, match="No class parameter decoder implemented for type 1000"):
        # Test for an enum value that isn't even defined in this version
        parameter_serde_registry.decode(api_pb2.ClassParameterValue(type=1000))  # type: ignore


def test_schema_extraction_unknown():
    def with_empty(a): ...

    def with_any(a: typing.Any): ...

    class Custom:
        pass

    def with_custom(a: Custom): ...

    for func in [with_empty, with_any, with_custom]:
        print(func.__name__)
        fields = list(get_callable_schema(func, is_web_endpoint=False).arguments)
        assert fields == [
            api_pb2.ClassParameterSpec(
                name="a",
                has_default=False,
                full_type=api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_UNKNOWN),
            )
        ]

    def with_default(a=5): ...

    fields = list(get_callable_schema(with_default, is_web_endpoint=False).arguments)
    assert fields == [
        api_pb2.ClassParameterSpec(
            name="a", full_type=api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_UNKNOWN), has_default=True
        )
    ]


def test_schema_extraction_int():
    def f(int_value: int = 1337): ...

    sig = inspect.signature(f)
    (int_spec,) = signature_to_parameter_specs(sig)
    assert int_spec == api_pb2.ClassParameterSpec(
        name="int_value",
        type=api_pb2.PARAM_TYPE_INT,
        full_type=api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_INT),
        has_default=True,
        int_default=1337,
    )


def test_schema_extraction_str():
    def foo(str_value: str = "foo"):
        pass

    (str_spec,) = signature_to_parameter_specs(inspect.signature(foo))
    assert str_spec == api_pb2.ClassParameterSpec(
        name="str_value",
        type=api_pb2.PARAM_TYPE_STRING,
        full_type=api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_STRING),
        has_default=True,
        string_default="foo",
    )


def test_schema_extraction_bytes():
    def foo(a: bytes = b"foo"):
        pass

    (bytes_spec,) = signature_to_parameter_specs(inspect.signature(foo))
    assert bytes_spec == api_pb2.ClassParameterSpec(
        name="a",
        type=api_pb2.PARAM_TYPE_BYTES,  # for backward compatibility
        has_default=True,
        bytes_default=b"foo",  # for backward compatibility
        full_type=api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_BYTES),
    )


def test_schema_extraction_list():
    def new_f(simple_list: list[int]): ...
    def old_f(simple_list: list[int]): ...

    for f in [new_f, old_f]:
        (list_spec,) = get_callable_schema(f, is_web_endpoint=False).arguments
        assert list_spec == api_pb2.ClassParameterSpec(
            name="simple_list",
            full_type=api_pb2.GenericPayloadType(
                base_type=api_pb2.PARAM_TYPE_LIST,
                sub_types=[api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_INT)],
            ),
            has_default=False,
        )


def optional_str_union(a: str | None = None): ...


def optional_str_typing(a: typing.Optional[str] = None): ...


def optional_str_typing_union(a: typing.Union[None, str] = None): ...  # noqa: UP007


def optional_list(a: list[bytes] | None = None): ...


@pytest.mark.parametrize(
    "f, expected_type",
    [
        (
            optional_str_union,
            api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_STRING),
        ),
        (
            optional_str_typing,
            api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_STRING),
        ),
        (
            optional_str_typing_union,
            api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_STRING),
        ),
        (
            optional_list,
            api_pb2.GenericPayloadType(
                base_type=api_pb2.PARAM_TYPE_LIST,
                sub_types=[api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_BYTES)],
            ),
        ),
    ],
)
def test_schema_extraction_optional(f, expected_type):
    (spec,) = get_callable_schema(f, is_web_endpoint=False).arguments
    assert spec == api_pb2.ClassParameterSpec(name="a", full_type=expected_type, has_default=True)


def test_schema_extraction_optional_without_default():
    # The schema has no nullability flag, so a required `X | None` is recorded as a required `X`.
    def f(a: int | None) -> str | None: ...

    schema = get_callable_schema(f, is_web_endpoint=False)
    (spec,) = schema.arguments
    assert spec == api_pb2.ClassParameterSpec(
        name="a",
        full_type=api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_INT),
        has_default=False,
    )
    assert schema.return_type == api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_STRING)


def multi_type_union(a: int | str): ...


def multi_type_optional_union(a: int | str | None): ...


@pytest.mark.parametrize("f", [multi_type_union, multi_type_optional_union])
def test_schema_extraction_multi_type_union_is_unknown(f):
    (spec,) = get_callable_schema(f, is_web_endpoint=False).arguments
    assert spec == api_pb2.ClassParameterSpec(
        name="a",
        full_type=api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_UNKNOWN),
        has_default=False,
    )


def test_schema_extraction_nested_list():
    def f(nested_list: list[list[bytes]]): ...

    (list_spec,) = get_callable_schema(f, is_web_endpoint=False).arguments
    assert list_spec == api_pb2.ClassParameterSpec(
        name="nested_list",
        full_type=api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_LIST,
            sub_types=[
                api_pb2.GenericPayloadType(
                    base_type=api_pb2.PARAM_TYPE_LIST,
                    sub_types=[api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_BYTES)],
                )
            ],
        ),
        has_default=False,
    )


def test_schema_extraction_nested_dict():
    def f(nested_dict: dict[str, dict[str, bytes]] = {}): ...

    (dict_spec,) = get_callable_schema(f, is_web_endpoint=False).arguments
    assert dict_spec == api_pb2.ClassParameterSpec(
        name="nested_dict",
        full_type=api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_DICT,
            sub_types=[
                api_pb2.GenericPayloadType(
                    base_type=api_pb2.PARAM_TYPE_STRING,
                ),
                api_pb2.GenericPayloadType(
                    base_type=api_pb2.PARAM_TYPE_DICT,
                    sub_types=[
                        api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_STRING),
                        api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_BYTES),
                    ],
                ),
            ],
        ),
        has_default=True,
    )


def test_schema_extraction_dict_with_non_str_key_is_unknown():
    def f(dct: dict): ...

    (dict_spec,) = get_callable_schema(f, is_web_endpoint=False).arguments
    print(dict_spec)
    assert dict_spec == api_pb2.ClassParameterSpec(
        name="dct",
        full_type=api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_DICT,
        ),
    )


def test_schema_extraction_bool():
    def f(bool_val: bool = True): ...

    (bool_spec,) = signature_to_parameter_specs(inspect.signature(f))
    print(bool_spec)
    assert bool_spec == api_pb2.ClassParameterSpec(
        name="bool_val",
        type=api_pb2.PARAM_TYPE_BOOL,
        full_type=api_pb2.GenericPayloadType(
            base_type=api_pb2.PARAM_TYPE_BOOL,
        ),
        has_default=True,
        bool_default=True,
    )


@pytest.mark.parametrize("v", [True, False])
def test_parameter_value_serde_bool(v):
    encoded = parameter_serde_registry.encode(v)
    assert encoded == api_pb2.ClassParameterValue(type=api_pb2.PARAM_TYPE_BOOL, bool_value=v)
    decoded = parameter_serde_registry.decode(encoded)
    assert decoded is v


def test_parameter_validate_bool():
    with pytest.raises(TypeError):
        parameter_serde_registry.validate_value_for_enum_type(api_pb2.PARAM_TYPE_BOOL, 1)
