import asyncio
import pytest
import time

from grpclib import Status

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
    assert excinfo.value


@pytest.mark.asyncio
async def test_client_connection_failure():
    with pytest.raises(ConnectionError) as excinfo:
        async with AioClient("https://localhost:443", api_pb2.CLIENT_TYPE_CLIENT, None):
            pass
    assert excinfo.value


@pytest.mark.asyncio
async def test_client_connection_timeout(servicer):
    with pytest.raises(ConnectionError) as excinfo:
        async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, None, version="timeout"):
            pass
    # The HTTP lookup will return 400 because the GRPC server rejects the http request
    assert "deadline" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_client_server_error(servicer):
    with pytest.raises(ConnectionError) as excinfo:
        async with AioClient("https://github.com", api_pb2.CLIENT_TYPE_CLIENT, None):
            pass
    # Can't connect over gRPC, but the HTTP lookup should succeed
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
    with pytest.warns(modal.exception.DeprecationError):
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
        servicer.heartbeat_status_code = Status.NOT_FOUND
        await client._heartbeat()
        await asyncio.sleep(0)  # let event loop take care of cleanup

        with pytest.raises(modal.exception.ConnectionError):
            client.stub


@pytest.mark.asyncio
async def test_client_heartbeat_retry(servicer):
    async with AioClient(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret")) as client:
        servicer.heartbeat_status_code = Status.UNAVAILABLE
        # No error.
        await client._heartbeat()
        servicer.heartbeat_status_code = Status.DEADLINE_EXCEEDED
        # No error.
        await client._heartbeat()
        servicer.heartbeat_status_code = Status.UNAUTHENTICATED
        # Raises.
        with pytest.raises(ConnectionError) as excinfo:
            await client._heartbeat()
        assert "UNAUTHENTICATED" in str(excinfo.value)
        assert "HTTP status" in str(excinfo.value)


def client_from_env(remote_addr, _override_time=None):
    _override_config = {
        "server_url": remote_addr,
        "token_id": "foo-id",
        "token_secret": "foo-secret",
        "task_id": None,
        "task_secret": None,
    }
    return Client.from_env(_override_config=_override_config, _override_time=_override_time)


def test_client_from_env(servicer):
    try:
        # First, a failing one
        with pytest.raises(ConnectionError):
            client_from_env("https://foo.invalid")

        # Make sure later clients can still succeed
        client_1 = client_from_env(servicer.remote_addr)
        client_2 = client_from_env(servicer.remote_addr)
        assert isinstance(client_1, Client)
        assert isinstance(client_2, Client)
        assert client_1 == client_2

    finally:
        Client.stop_env_client()

    try:
        # After stopping, creating a new client should return a new one
        client_3 = client_from_env(servicer.remote_addr)
        client_4 = client_from_env(servicer.remote_addr)
        assert client_3 != client_1
        assert client_4 == client_3

        # Inject a heartbeat failure in the client
        servicer.heartbeat_status_code = Status.NOT_FOUND
        client_3._heartbeat()

        # Make sure the new env client is different
        client_5 = client_from_env(servicer.remote_addr)
        assert client_5 != client_4

        # Fetch another one seconds later
        client_6 = client_from_env(servicer.remote_addr, time.time() + 1)
        assert client_6 == client_5

        # Fetch another much later, should be different
        client_7 = client_from_env(servicer.remote_addr, time.time() + 999)
        assert client_7 != client_6
    finally:
        Client.stop_env_client()
