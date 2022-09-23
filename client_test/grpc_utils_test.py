import pytest

import pytest_asyncio
from grpclib import GRPCError, Status

from modal_proto import api_grpc, api_pb2
from modal_utils.async_utils import TaskContext
from modal_utils.grpc_utils import ChannelPool, retry_transient_errors
from modal_utils.server_connection import GRPCConnectionFactory


@pytest_asyncio.fixture
async def client_stub(servicer):
    # Most of this is duplicated from the Client class
    async with TaskContext(grace=1) as task_context:
        credentials = ("foo-token", "foo-secret")
        connection_factory = GRPCConnectionFactory(
            servicer.remote_addr,
            api_pb2.CLIENT_TYPE_CLIENT,
            credentials,
        )
        channel_pool = ChannelPool(task_context, connection_factory)
        await channel_pool.start()
        yield api_grpc.ModalClientStub(channel_pool)
        channel_pool.close()


@pytest.mark.asyncio
async def test_retry_transient_errors(servicer, client_stub):
    # Use the BlobCreate request for retries
    try:
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
    finally:
        servicer.fail_blob_create = []
