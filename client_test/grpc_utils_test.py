# Copyright Modal Labs 2022
import asyncio
import math
import pytest
import time

from grpclib import GRPCError, Status

from modal_proto import api_grpc, api_pb2
from modal_utils.grpc_utils import ChannelPool, create_channel, retry_transient_errors

from .supports.skip import skip_windows_unix_socket


@pytest.mark.asyncio
async def test_http_channel(servicer):
    assert servicer.remote_addr.startswith("http://")
    channel = create_channel(servicer.remote_addr, use_pool=False)
    client_stub = api_grpc.ModalClientStub(channel)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)
    assert resp.blob_id

    channel.close()


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_unix_channel(unix_servicer):
    assert unix_servicer.remote_addr.startswith("unix://")
    channel = create_channel(unix_servicer.remote_addr, use_pool=False)
    client_stub = api_grpc.ModalClientStub(channel)

    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)
    assert resp.blob_id

    channel.close()


@pytest.mark.asyncio
async def test_channel_pool(servicer, n=1000):
    channel_pool = create_channel(servicer.remote_addr, use_pool=True)
    assert isinstance(channel_pool, ChannelPool)
    client_stub = api_grpc.ModalClientStub(channel_pool)

    # Trigger a lot of requests
    for i in range(n):
        req = api_pb2.BlobCreateRequest()
        resp = await client_stub.BlobCreate(req)
        assert resp.blob_id

    # Make sure we created the right number of subchannels
    assert len(channel_pool._subchannels) == math.ceil(n / channel_pool._max_requests)

    channel_pool.close()


@pytest.mark.asyncio
async def test_channel_pool_closed_transport(servicer):
    channel_pool = create_channel(servicer.remote_addr, use_pool=True)
    assert isinstance(channel_pool, ChannelPool)

    connection = await channel_pool.__connect__()
    connection.connection_lost(None)  # simulates a h2 connection being closed

    client_stub = api_grpc.ModalClientStub(channel_pool)
    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)  # this should close the terminated connection and start a new one
    assert resp.blob_id
    channel_pool.close()


@pytest.mark.asyncio
async def test_channel_pool_max_active(servicer):
    channel_pool = create_channel(servicer.remote_addr, use_pool=True)
    assert isinstance(channel_pool, ChannelPool)
    channel_pool._max_active = 1.0
    client_stub = api_grpc.ModalClientStub(channel_pool)

    # Do a few requests and assert there's just one subchannel
    for i in range(3):
        req = api_pb2.BlobCreateRequest()
        resp = await client_stub.BlobCreate(req)
        assert resp.blob_id
    assert len(channel_pool._subchannels) == 1

    # Sleep a couple of seconds and do a new request: it should create a new subchannel
    await asyncio.sleep(2.0)
    req = api_pb2.BlobCreateRequest()
    resp = await client_stub.BlobCreate(req)
    assert resp.blob_id
    assert len(channel_pool._subchannels) == 2

    channel_pool.close()


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
