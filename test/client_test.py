import asyncio
import pytest

from polyester.client import Client
from polyester.proto import api_pb2


@pytest.mark.asyncio
async def test_client(servicer):
    client = Client(servicer.remote_addr, api_pb2.ClientType.CLIENT, ('foo-id', 'foo-secret'))

    # TODO: let's rethink how we're doing it, should we bring the context mgr back maybe?
    await client._start()
    await client._start_client()
    await asyncio.sleep(0.1)
    await client._start_session()
    await asyncio.sleep(0.1)
    await client._close()

    assert len(servicer.requests) == 3
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CLIENT
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)
    assert isinstance(servicer.requests[2], api_pb2.SessionCreateRequest)
    # assert isinstance(servicer.requests[2], api_pb2.ByeRequest)


@pytest.mark.asyncio
async def test_container_client(servicer):
    client = Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ('ta-123', 'task-secret'))

    # TODO: let's rethink how we're doing it, should we bring the context mgr back maybe?
    await client._start()
    await client._start_client()
    await asyncio.sleep(0.1)  # enough for a handshake to go through
    await client._close()

    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CONTAINER
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)
    # assert isinstance(servicer.requests[2], api_pb2.ByeRequest)
