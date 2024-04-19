# Copyright Modal Labs 2022
import pytest
import time

from grpclib import GRPCError, Status

from modal._utils.grpc_utils import create_channel, retry_transient_errors
from modal_proto import api_grpc, api_pb2

from .supports.skip import skip_windows_unix_socket


@pytest.mark.asyncio
async def test_http_channel(servicer):
    assert servicer.remote_addr.startswith("http://")
    channel = create_channel(servicer.remote_addr)
    client_stub = api_grpc.ModalClientStub(channel)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)
    assert resp.blob_id

    channel.close()


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_unix_channel(unix_servicer):
    assert unix_servicer.remote_addr.startswith("unix://")
    channel = create_channel(unix_servicer.remote_addr)
    client_stub = api_grpc.ModalClientStub(channel)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)
    assert resp.blob_id

    channel.close()


@pytest.mark.asyncio
async def test_retry_transient_errors(servicer):
    channel = create_channel(servicer.remote_addr)
    client_stub = api_grpc.ModalClientStub(channel)

    # Use the BlobCreate request for retries
    req = api_pb2.BlobCreateRequest()

    # Fail 3 times -> should still succeed
    servicer.fail_blob_create = [Status.UNAVAILABLE] * 3
    assert await retry_transient_errors(client_stub.BlobCreate, req)
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "3"

    # Fail 4 times -> should fail
    servicer.fail_blob_create = [Status.UNAVAILABLE] * 4
    with pytest.raises(GRPCError):
        await retry_transient_errors(client_stub.BlobCreate, req)
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "3"

    # Fail 5 times, but set max_retries to infinity
    servicer.fail_blob_create = [Status.UNAVAILABLE] * 5
    assert await retry_transient_errors(client_stub.BlobCreate, req, max_retries=None, base_delay=0)
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "5"

    # Not a transient error.
    servicer.fail_blob_create = [Status.PERMISSION_DENIED]
    with pytest.raises(GRPCError):
        assert await retry_transient_errors(client_stub.BlobCreate, req, max_retries=None, base_delay=0)
    assert servicer.blob_create_metadata.get("x-idempotency-key")
    assert servicer.blob_create_metadata.get("x-retry-attempt") == "0"

    # Make sure to respect total_timeout
    t0 = time.time()
    servicer.fail_blob_create = [Status.UNAVAILABLE] * 99
    with pytest.raises(GRPCError):
        assert await retry_transient_errors(client_stub.BlobCreate, req, max_retries=None, total_timeout=3)
    total_time = time.time() - t0
    assert total_time <= 3.1

    channel.close()
