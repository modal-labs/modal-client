import asyncio
import pytest

from polyester.client import Client
from polyester.proto import api_pb2


@pytest.mark.asyncio
async def test_client(servicer):
    async with Client(servicer.remote_addr, api_pb2.ClientType.CLIENT, ("foo-id", "foo-secret")) as client:
        await asyncio.sleep(0.1)

    # TODO: Test sessions too!!

    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CLIENT
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)
    # assert isinstance(servicer.requests[2], api_pb2.SessionCreateRequest)
    # assert isinstance(servicer.requests[2], api_pb2.ByeRequest)


@pytest.mark.asyncio
async def test_container_client(servicer):
    async with Client(servicer.remote_addr, api_pb2.ClientType.CONTAINER, ("ta-123", "task-secret")) as client:
        await asyncio.sleep(0.1)  # enough for a handshake to go through

    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CONTAINER
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)
    # assert isinstance(servicer.requests[2], api_pb2.ByeRequest)
