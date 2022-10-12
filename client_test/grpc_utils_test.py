import platform
import pytest
import time

import pytest_asyncio
from grpclib import GRPCError, Status

from modal_proto import api_grpc, api_pb2
from modal_utils.async_utils import TaskContext
from modal_utils.grpc_utils import (
    ChannelPool,
    _create_channel,
    create_channel,
    retry_transient_errors,
)


@pytest.mark.asyncio
async def test_http_channel(servicer):
    assert servicer.remote_addr.startswith("http://")
    channel = _create_channel(servicer.remote_addr)
    client_stub = api_grpc.ModalClientStub(channel)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)
    assert resp.blob_id


@pytest.mark.skipif(platform.system() == "Windows", reason="Windows doesn't have UNIX sockets")
@pytest.mark.asyncio
async def test_unix_channel(unix_servicer):
    assert unix_servicer.remote_addr.startswith("unix://")
    channel = _create_channel(unix_servicer.remote_addr)
    client_stub = api_grpc.ModalClientStub(channel)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)
    assert resp.blob_id


@pytest_asyncio.fixture(scope="function")
async def task_context():
    async with TaskContext(grace=1) as tc:
        yield tc


@pytest.mark.asyncio
async def test_channel_pool(task_context, servicer):
    channel_pool = ChannelPool(task_context, servicer.remote_addr, None, None, None)
    client_stub = api_grpc.ModalClientStub(channel_pool)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)
    assert resp.blob_id


@pytest.mark.asyncio
async def test_retry_transient_errors(task_context, servicer):
    channel = create_channel(task_context, servicer.remote_addr)
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
