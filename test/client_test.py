# Copyright Modal Labs 2022
import platform
import pytest

from google.protobuf.empty_pb2 import Empty

import modal.exception
from modal import Client
from modal.exception import AuthError, ConnectionError, VersionError
from modal_proto import api_pb2

from .supports.skip import skip_windows_unix_socket

TEST_TIMEOUT = 4.0  # align this with the container client timeout in client.py


def test_client_type(servicer, client):
    assert len(servicer.requests) == 1
    assert isinstance(servicer.requests[0], Empty)
    assert servicer.client_create_metadata["x-modal-client-type"] == str(api_pb2.CLIENT_TYPE_CLIENT)


def test_client_platform_string(servicer, client):
    platform_str = servicer.client_create_metadata["x-modal-platform"]
    system, release, machine = platform_str.split("-")
    if platform.system() == "Darwin":
        assert system == "macOS"
        assert release == platform.mac_ver()[0].replace("-", "_")
    else:
        assert system == platform.system().replace("-", "_")
        assert release == platform.release().replace("-", "_")
    assert machine == platform.machine().replace("-", "_")


@pytest.mark.asyncio
@skip_windows_unix_socket
async def test_container_client_type(unix_servicer, container_client):
    assert len(unix_servicer.requests) == 1  # no heartbeat, just ClientHello
    assert isinstance(unix_servicer.requests[0], Empty)
    assert unix_servicer.client_create_metadata["x-modal-client-type"] == str(api_pb2.CLIENT_TYPE_CONTAINER)


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_client_dns_failure():
    with pytest.raises(ConnectionError) as excinfo:
        async with Client("https://xyz.invalid", api_pb2.CLIENT_TYPE_CONTAINER, None):
            pass
    assert excinfo.value


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
@skip_windows_unix_socket
async def test_client_connection_failure():
    with pytest.raises(ConnectionError) as excinfo:
        async with Client("https://localhost:443", api_pb2.CLIENT_TYPE_CONTAINER, None):
            pass
    assert excinfo.value


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
@skip_windows_unix_socket
async def test_client_connection_failure_unix_socket():
    with pytest.raises(ConnectionError) as excinfo:
        async with Client("unix:/tmp/xyz.txt", api_pb2.CLIENT_TYPE_CONTAINER, None):
            pass
    assert excinfo.value


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
@skip_windows_unix_socket
async def test_client_connection_timeout(unix_servicer, monkeypatch):
    monkeypatch.setattr("modal.client.CLIENT_CREATE_ATTEMPT_TIMEOUT", 1.0)
    monkeypatch.setattr("modal.client.CLIENT_CREATE_TOTAL_TIMEOUT", 3.0)
    with pytest.raises(ConnectionError) as excinfo:
        async with Client(unix_servicer.remote_addr, api_pb2.CLIENT_TYPE_CONTAINER, None, version="timeout"):
            pass

    # The HTTP lookup will return 400 because the GRPC server rejects the http request
    assert "deadline" in str(excinfo.value).lower()


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_client_server_error(servicer):
    with pytest.raises(ConnectionError) as excinfo:
        async with Client("https://github.com", api_pb2.CLIENT_TYPE_CLIENT, None):
            pass
    # Can't connect over gRPC, but the HTTP lookup should succeed
    assert "HTTP status: 200" in str(excinfo.value)


@pytest.mark.asyncio
async def test_client_old_version(servicer):
    with pytest.raises(VersionError):
        async with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret"), version="0.0.0"):
            pass


@pytest.mark.asyncio
async def test_client_deprecated(servicer):
    with pytest.warns(modal.exception.DeprecationError):
        async with Client(
            servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret"), version="deprecated"
        ):
            pass


@pytest.mark.asyncio
async def test_client_unauthenticated(servicer):
    with pytest.raises(AuthError):
        async with Client(servicer.remote_addr, api_pb2.CLIENT_TYPE_CLIENT, None, version="unauthenticated"):
            pass


def client_from_env(remote_addr):
    _override_config = {
        "server_url": remote_addr,
        "token_id": "foo-id",
        "token_secret": "foo-secret",
        "task_id": None,
        "task_secret": None,
    }
    return Client.from_env(_override_config=_override_config)


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
        Client.set_env_client(None)

    try:
        # After stopping, creating a new client should return a new one
        client_3 = client_from_env(servicer.remote_addr)
        client_4 = client_from_env(servicer.remote_addr)
        assert client_3 != client_1
        assert client_4 == client_3
    finally:
        Client.set_env_client(None)
