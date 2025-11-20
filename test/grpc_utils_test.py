# Copyright Modal Labs 2022
import pytest
import time

from google.protobuf.any_pb2 import Any
from grpclib import GRPCError, Status

import modal
from modal import __version__
from modal._utils.async_utils import synchronize_api
from modal._utils.grpc_utils import (
    CustomProtoStatusDetailsCodec,
    Retry,
    connect_channel,
    create_channel,
    get_server_retry_policy,
)
from modal.exception import InvalidError
from modal_proto import api_grpc, api_pb2, sandbox_router_pb2

from .supports.skip import skip_windows_unix_socket


@pytest.mark.asyncio
async def test_http_channel(servicer, credentials):
    token_id, token_secret = credentials
    metadata = {
        "x-modal-client-type": str(api_pb2.CLIENT_TYPE_CLIENT),
        "x-modal-python-version": "3.12.1",
        "x-modal-client-version": __version__,
        "x-modal-token-id": token_id,
        "x-modal-token-secret": token_secret,
    }
    assert servicer.client_addr.startswith("http://")
    channel = create_channel(servicer.client_addr)
    client_stub = api_grpc.ModalClientStub(channel)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req, metadata=metadata)
    assert len(resp.blob_ids) > 0

    channel.close()


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_unix_channel(servicer):
    metadata = {
        "x-modal-client-type": str(api_pb2.CLIENT_TYPE_CONTAINER),
        "x-modal-python-version": "3.12.1",
        "x-modal-client-version": __version__,
    }
    assert servicer.container_addr.startswith("unix://")
    channel = create_channel(servicer.container_addr)
    client_stub = api_grpc.ModalClientStub(channel)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req, metadata=metadata)
    assert len(resp.blob_ids) > 0

    channel.close()


@pytest.mark.asyncio
async def test_http_broken_channel(monkeypatch):
    monkeypatch.setattr(modal._utils.async_utils, "RETRY_N_ATTEMPTS_OVERRIDE", 1)
    ch = create_channel("https://xyz.invalid")
    with pytest.raises(OSError):
        await connect_channel(ch)


@pytest.mark.asyncio
async def test_retry_transient_errors(servicer, client):
    client_stub = client.stub

    @synchronize_api
    async def wrapped_blob_create(req, **kwargs):
        return await client_stub.BlobCreate(req, **kwargs)

    # Use the BlobCreate request for retries
    req = api_pb2.BlobCreateRequest()

    def wrap_grpc_error(status):
        return [GRPCError(s, "foobar") for s in status]

    # Fail 3 times -> should still succeed
    servicer.fail_blob_create = wrap_grpc_error([Status.UNAVAILABLE] * 3)
    await wrapped_blob_create.aio(req)
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "3"

    # Fail 4 times -> should fail
    servicer.fail_blob_create = wrap_grpc_error([Status.UNAVAILABLE] * 4)
    with pytest.raises(GRPCError):
        await wrapped_blob_create.aio(req)
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "3"

    # Fail 5 times, but set max_retries to infinity
    servicer.fail_blob_create = wrap_grpc_error([Status.UNAVAILABLE] * 5)
    assert await wrapped_blob_create.aio(req, retry=Retry(max_retries=None, base_delay=0))
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "5"

    # Not a transient error.
    servicer.fail_blob_create = wrap_grpc_error([Status.PERMISSION_DENIED])
    with pytest.raises(GRPCError):
        assert await wrapped_blob_create.aio(req, retry=Retry(max_retries=None, base_delay=0))
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "0"

    # Make sure to respect total_timeout
    t0 = time.time()
    servicer.fail_blob_create = wrap_grpc_error([Status.UNAVAILABLE] * 99)
    with pytest.raises(GRPCError):
        assert await wrapped_blob_create.aio(req, retry=Retry(max_retries=None, total_timeout=3))
    total_time = time.time() - t0
    assert total_time <= 3.1

    # Check input_plane_region included
    servicer.fail_blob_create = []  # Reset to no failures
    await wrapped_blob_create.aio(req, metadata=[("x-modal-input-plane-region", "us-east")])
    assert servicer.blob_create_metadata.get("x-modal-input-plane-region") == "us-east"

    # Check input_plane_region not included
    servicer.fail_blob_create = wrap_grpc_error([Status.UNAVAILABLE] * 3)
    await wrapped_blob_create.aio(req)
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "3"
    assert servicer.blob_create_metadata.get("x-modal-input-plane-region") is None

    # Check all metadata is included
    servicer.fail_blob_create = wrap_grpc_error([Status.UNAVAILABLE] * 3)
    await wrapped_blob_create.aio(req, metadata=[("x-modal-input-plane-region", "us-east")])
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "3"
    assert servicer.blob_create_metadata.get("x-modal-input-plane-region") == "us-east"


@pytest.mark.asyncio
async def test_retry_timeout_error(servicer, client):
    client_stub = client.stub

    @synchronize_api
    async def wrapped_blob_create(req, **kwargs):
        return await client_stub.BlobCreate(req, **kwargs)

    # Use the BlobCreate request for retries
    req = api_pb2.BlobCreateRequest()
    with pytest.raises(InvalidError, match="Retry must be None when timeout is set"):
        await wrapped_blob_create.aio(req, timeout=4.0)


def test_CustomProtoStatusDetailsCodec_round_trip():
    blob_msg = api_pb2.BlobCreateResponse(blob_id="abc")
    sandbox_msg = sandbox_router_pb2.SandboxExecPollResponse(code=31)
    msgs = [blob_msg, sandbox_msg]

    codec = CustomProtoStatusDetailsCodec()
    encoded_msg = codec.encode(Status.CANCELLED, "this-is-a-message", msgs)
    assert isinstance(encoded_msg, bytes)

    decoded_status = api_pb2.RPCStatus.FromString(encoded_msg)
    assert decoded_status.message == "this-is-a-message"
    assert decoded_status.code == Status.CANCELLED.value

    decoded_msg = codec.decode(Status.CANCELLED, None, encoded_msg)
    assert len(decoded_msg) == 2
    assert decoded_msg == msgs


def test_CustomProtoStatusDetailsCodec_unknown():
    encoded_details = [
        Any(type_url="abc", value=b"bad"),
    ]
    encoded_msg = api_pb2.RPCStatus(details=encoded_details).SerializeToString()
    codec = CustomProtoStatusDetailsCodec()

    decoded_msg = codec.decode(Status.INTERNAL, None, encoded_msg)
    assert not decoded_msg


def test_CustomProtoStatusDetailsCodec_google_common_proto_compat():
    """Check that rpc's encoded with the default GRPC codec works with the
    CustomProtoStatusDetailsCodec decoder."""

    # ProtoStatusDetailsCodec requires `googleapis-common-protos` to be installed,
    # which installs `google.rpc`.
    pytest.importorskip("google.rpc")

    from grpclib.encoding.proto import ProtoStatusDetailsCodec

    blob_msg = api_pb2.BlobCreateResponse(blob_id="abc")
    sandbox_msg = sandbox_router_pb2.SandboxExecPollResponse(code=31)
    msgs = [blob_msg, sandbox_msg]

    grpclib_codec = ProtoStatusDetailsCodec()
    message = grpclib_codec.encode(Status.INTERNAL, "my-message", details=msgs)
    codec = CustomProtoStatusDetailsCodec()

    decoded_msg = codec.decode(Status.INTERNAL, None, message)
    assert len(decoded_msg) == 2
    assert decoded_msg == msgs


@pytest.mark.asyncio
async def test_codec_with_channel(servicer, client):
    """Check codec works with channel."""

    details = [api_pb2.BlobCreateResponse(blob_id="abc")]

    async def raise_error(servicer, stream):
        raise GRPCError(Status.INTERNAL, "Blob create failed", details=details)

    req = api_pb2.BlobCreateRequest()

    with servicer.intercept() as ctx:
        ctx.set_responder("BlobCreate", raise_error)
        with pytest.raises(GRPCError) as excinfo:
            await client.stub.BlobCreate(req, retry=None, timeout=0.1)
    assert excinfo.value.details == details


@pytest.mark.asyncio
async def test_flash_container_register_deregister(servicer, client):
    client_stub = client.stub

    @synchronize_api
    async def wrapped_flash_container_register(req, **kwargs):
        return await client_stub.FlashContainerRegister(req, **kwargs)

    @synchronize_api
    async def wrapped_flash_container_deregister(req, **kwargs):
        return await client_stub.FlashContainerDeregister(req, **kwargs)

    # Test invalid request
    register_req = api_pb2.FlashContainerRegisterRequest(service_name="test", host="localhost", port=8000)
    with pytest.raises(InvalidError, match="Retry must be None when timeout is set"):
        await wrapped_flash_container_register.aio(register_req, timeout=4.0)

    # Test working request
    await wrapped_flash_container_register.aio(register_req)
    assert servicer.flash_container_registrations == {"test": "http://localhost:8000"}

    # Test working request
    deregister_req = api_pb2.FlashContainerDeregisterRequest(service_name="test")
    await wrapped_flash_container_deregister.aio(deregister_req, timeout=4.0, retry=None)
    assert servicer.flash_container_registrations == {}


@pytest.mark.parametrize(
    "exception, expected_instruction",
    [
        (ValueError(), None),
        (GRPCError(Status.UNAVAILABLE, "my-message"), None),
        (GRPCError(Status.UNAVAILABLE, "my-message", details=[api_pb2.FlashContainerListResponse()]), None),
        (
            GRPCError(
                Status.UNAVAILABLE,
                "my-message",
                details=[
                    api_pb2.RPCRetryPolicy(
                        retry_after_secs=2,
                    )
                ],
            ),
            api_pb2.RPCRetryPolicy(retry_after_secs=2),
        ),
    ],
)
def test_get_server_retry_policy(exception, expected_instruction):
    assert get_server_retry_policy(exception) == expected_instruction


@synchronize_api
async def test_retry_transient_errors_grpc_retry(servicer, client, caplog, monkeypatch):
    monkeypatch.setattr(modal._utils.grpc_utils, "SERVER_RETRY_WARNING_TIME_INTERVAL", 0.2)
    req = api_pb2.BlobCreateRequest()
    servicer.fail_blob_create = [GRPCError(Status.RESOURCE_EXHAUSTED, "foobar")] + [
        GRPCError(Status.RESOURCE_EXHAUSTED, "foobar-message", details=[api_pb2.RPCRetryPolicy(retry_after_secs=0.1)])
    ] * 10

    with pytest.raises(GRPCError):
        await client.stub.BlobCreate(req)

    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "10"

    assert caplog.text.count("foobar-message. Will retry in 0.10 seconds") == 5


@synchronize_api
async def test_retry_transient_errors_grpc_retry_total_timeout(servicer, client, monkeypatch):
    """No retries when MODAL_MAX_THROTTLE_WAIT is lower than retry_after_secs."""

    monkeypatch.setenv("MODAL_MAX_THROTTLE_WAIT", "1")
    req = api_pb2.BlobCreateRequest()
    servicer.fail_blob_create = [
        GRPCError(
            Status.RESOURCE_EXHAUSTED,
            "foobar-message",
            details=[api_pb2.RPCRetryPolicy(retry_after_secs=3)],
        )
    ]

    with pytest.raises(GRPCError):
        await client.stub.BlobCreate(req)

    assert servicer.blob_create_metadata.get("x-retry-attempt") == "0"
