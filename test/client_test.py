import asyncio

import pytest

from modal._client import Client
from modal.exception import ConnectionError, VersionError
from modal.proto import api_pb2


@pytest.mark.asyncio
async def test_client(servicer, client):
    await asyncio.sleep(0.1)  # wait for heartbeat
    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CT_CLIENT
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)


@pytest.mark.asyncio
async def test_container_client(servicer, container_client):
    await asyncio.sleep(0.1)  # wait for heartbeat
    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.ClientType.CT_CONTAINER
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)


@pytest.mark.asyncio
async def test_client_connection_failure():
    with pytest.raises(ConnectionError):
        async with Client("https://xyz.invalid", api_pb2.ClientType.CT_CLIENT, None):
            pass


@pytest.mark.asyncio
async def test_client_old_version(servicer):
    with pytest.raises(VersionError):
        async with Client(
            servicer.remote_addr, api_pb2.ClientType.CT_CLIENT, ("foo-id", "foo-secret"), version="0.0.0"
        ):
            pass
