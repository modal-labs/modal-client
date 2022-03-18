import asyncio
import pytest

import modal.exception
from modal.client import AioClient
from modal.exception import ConnectionError, VersionError
from modal_proto import api_pb2


@pytest.mark.asyncio
async def test_client(servicer, client):
    await asyncio.sleep(0.1)  # wait for heartbeat
    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.CLIENT_TYPE_CLIENT
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)


@pytest.mark.asyncio
async def test_container_client(servicer, container_client):
    await asyncio.sleep(0.1)  # wait for heartbeat
    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.CLIENT_TYPE_CONTAINER
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)


@pytest.mark.asyncio
async def test_client_connection_failure():
    with pytest.raises(ConnectionError):
        async with AioClient("https://xyz.invalid", api_pb2.CLIENT_TYPE_CLIENT, None):
            pass


@pytest.mark.asyncio
async def test_client_old_version(servicer):
    with pytest.raises(VersionError):
        async with AioClient(
            servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret"), version="0.0.0"
        ):
            pass


@pytest.mark.asyncio
async def test_server_client_gone_disconnects_client(servicer):
    async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        servicer.heartbeat_return_client_gone = True
        await client._heartbeat()
        await asyncio.sleep(0)  # let event loop take care of cleanup

        with pytest.raises(modal.exception.ConnectionError):
            client.stub
