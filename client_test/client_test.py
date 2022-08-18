import asyncio
import pytest

from grpc import StatusCode

import modal.exception
from modal.client import AioClient, Client
from modal.exception import AuthError, ConnectionError, VersionError
from modal_proto import api_pb2


@pytest.mark.asyncio
async def test_client(servicer, client):
    await asyncio.sleep(0.1)  # wait for heartbeat
    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.CLIENT_TYPE_CLIENT
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)


@pytest.mark.asyncio
async def test_container_client(servicer, aio_container_client):
    await asyncio.sleep(0.1)  # wait for heartbeat
    assert len(servicer.requests) == 2
    assert isinstance(servicer.requests[0], api_pb2.ClientCreateRequest)
    assert servicer.requests[0].client_type == api_pb2.CLIENT_TYPE_CONTAINER
    assert isinstance(servicer.requests[1], api_pb2.ClientHeartbeatRequest)


@pytest.mark.asyncio
async def test_client_dns_failure():
    with pytest.raises(ConnectionError) as excinfo:
        async with AioClient("https://xyz.invalid", api_pb2.CLIENT_TYPE_CLIENT, None):
            pass
    assert "DNS resolution failed for xyz.invalid" in str(excinfo.value)
    assert "HTTP exception" in str(excinfo.value)


@pytest.mark.asyncio
async def test_client_connection_failure():
    with pytest.raises(ConnectionError) as excinfo:
        async with AioClient("https://localhost:443", api_pb2.CLIENT_TYPE_CLIENT, None):
            pass
    assert "failed to connect" in str(excinfo.value).lower()
    assert "HTTP exception: ConnectionRefusedError" in str(excinfo.value)


@pytest.mark.asyncio
async def test_client_connection_timeout(servicer):
    with pytest.raises(ConnectionError) as excinfo:
        async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, None, version="timeout"):
            pass
    # The HTTP lookup will return 400 because the GRPC server rejects the http request
    assert "Deadline Exceeded" in str(excinfo.value)
    assert "HTTP status: 400" in str(excinfo.value)


@pytest.mark.asyncio
async def test_client_server_error(servicer):
    with pytest.raises(ConnectionError) as excinfo:
        async with AioClient("https://github.com", api_pb2.CLIENT_TYPE_CLIENT, None):
            pass
    # Can't connect over GRPC, but the HTTP lookup should succeed
    assert "HTTP status: 200" in str(excinfo.value)


@pytest.mark.asyncio
async def test_client_old_version(servicer):
    with pytest.raises(VersionError):
        async with AioClient(
            servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret"), version="0.0.0"
        ):
            pass


@pytest.mark.asyncio
async def test_client_deprecated(servicer):
    with pytest.deprecated_call():
        async with AioClient(
            servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret"), version="deprecated"
        ):
            pass


@pytest.mark.asyncio
async def test_client_unauthenticated(servicer):
    with pytest.raises(AuthError):
        async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, None, version="unauthenticated"):
            pass


@pytest.mark.skip("TODO: flakes in Github Actions")
@pytest.mark.asyncio
async def test_server_client_gone_disconnects_client(servicer):
    async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        servicer.heartbeat_status_code = StatusCode.NOT_FOUND
        await client._heartbeat()
        await asyncio.sleep(0)  # let event loop take care of cleanup

        with pytest.raises(modal.exception.ConnectionError):
            client.stub


@pytest.mark.asyncio
async def test_client_heartbeat_retry(servicer):
    async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        servicer.heartbeat_status_code = StatusCode.UNAVAILABLE
        # No error.
        await client._heartbeat()
        servicer.heartbeat_status_code = StatusCode.DEADLINE_EXCEEDED
        # No error.
        await client._heartbeat()
        servicer.heartbeat_status_code = StatusCode.UNAUTHENTICATED
        # Raises.
        with pytest.raises(ConnectionError) as excinfo:
            await client._heartbeat()
        assert "UNAUTHENTICATED" in str(excinfo.value)
        assert "HTTP status" in str(excinfo.value)


def test_client_from_env(servicer):
    _override_config = {
        "server_url": "https://foo.invalid",
        "token_id": "foo-id",
        "token_secret": "foo-secret",
        "task_id": None,
        "task_secret": None,
    }

    try:
        # First, a failing one
        with pytest.raises(ConnectionError):
            Client.from_env(_override_config=_override_config)

        # Make sure later clients can still succeed
        _override_config["server_url"] = servicer.remote_addr
        client_1 = Client.from_env(_override_config=_override_config)
        client_2 = Client.from_env(_override_config=_override_config)
        assert isinstance(client_1, Client)
        assert isinstance(client_2, Client)
        assert client_1 == client_2

    finally:
        Client.stop_env_client()

    try:
        # After stopping, creating a new client should return a new one
        client_3 = Client.from_env(_override_config=_override_config)
        assert client_3 != client_1
    finally:
        Client.stop_env_client()
