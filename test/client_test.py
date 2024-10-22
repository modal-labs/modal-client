# Copyright Modal Labs 2022
import platform
import pytest
import subprocess
import sys

from google.protobuf.empty_pb2 import Empty

import modal.exception
from modal import Client
from modal.exception import AuthError, ConnectionError, DeprecationError, InvalidError, VersionError
from modal_proto import api_pb2

from .supports.skip import skip_windows, skip_windows_unix_socket

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
async def test_container_client_type(servicer, container_client):
    assert len(servicer.requests) == 1  # no heartbeat, just ClientHello
    assert isinstance(servicer.requests[0], Empty)
    assert servicer.client_create_metadata["x-modal-client-type"] == str(api_pb2.CLIENT_TYPE_CONTAINER)


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_client_dns_failure():
    with pytest.raises(ConnectionError) as excinfo:
        async with Client("https://xyz.invalid", api_pb2.CLIENT_TYPE_CONTAINER, None):
            pass
    assert excinfo.value


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
@skip_windows("Windows test crashes on connection failure")
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
async def test_client_connection_timeout(servicer, monkeypatch):
    monkeypatch.setattr("modal.client.CLIENT_CREATE_ATTEMPT_TIMEOUT", 1.0)
    monkeypatch.setattr("modal.client.CLIENT_CREATE_TOTAL_TIMEOUT", 3.0)
    with pytest.raises(ConnectionError) as excinfo:
        async with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CONTAINER, None, version="timeout"):
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
async def test_client_old_version(servicer, credentials):
    with pytest.raises(VersionError):
        async with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials, version="0.0.0"):
            pass


@pytest.mark.asyncio
async def test_client_deprecated(servicer, credentials):
    with pytest.warns(modal.exception.DeprecationError):
        async with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials, version="deprecated"):
            pass


@pytest.mark.asyncio
async def test_client_unauthenticated(servicer):
    with pytest.raises(AuthError):
        async with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, None, version="unauthenticated"):
            pass


def client_from_env(client_addr, credentials):
    token_id, token_secret = credentials
    _override_config = {
        "server_url": client_addr,
        "token_id": token_id,
        "token_secret": token_secret,
        "task_id": None,
        "task_secret": None,
    }
    return Client.from_env(_override_config=_override_config)


def test_client_from_env_client(servicer, credentials):
    try:
        # First, a failing one
        with pytest.raises(ConnectionError):
            client_from_env("https://foo.invalid", credentials)

        # Make sure later clients can still succeed
        client_1 = client_from_env(servicer.client_addr, credentials)
        client_2 = client_from_env(servicer.client_addr, credentials)
        assert isinstance(client_1, Client)
        assert isinstance(client_2, Client)
        assert client_1 == client_2

    finally:
        Client.set_env_client(None)

    try:
        # After stopping, creating a new client should return a new one
        client_3 = client_from_env(servicer.client_addr, credentials)
        client_4 = client_from_env(servicer.client_addr, credentials)
        assert client_3 != client_1
        assert client_4 == client_3
    finally:
        Client.set_env_client(None)


def test_multiple_profile_error(servicer, modal_config):
    config = """
    [prof-1]
    token_id = 'ak-abc'
    token_secret = 'as_xyz'
    active = true

    [prof-2]
    token_id = 'ak-abc'
    token_secret = 'as_xyz'
    active = true
    """
    with modal_config(config):
        with pytest.raises(InvalidError, match="More than one Modal profile is active"):
            Client.verify(servicer.client_addr, None)


def test_implicit_default_profile_warning(servicer, modal_config, credentials):
    config = """
    [default]
    token_id = 'ak-abc'
    token_secret = 'as_xyz'

    [other]
    token_id = 'ak-abc'
    token_secret = 'as_xyz'
    """
    with modal_config(config):
        with pytest.raises(DeprecationError, match="Support for using an implicit 'default' profile is deprecated."):
            Client.verify(servicer.client_addr, None)

    config = """
    [default]
    token_id = 'ak-abc'
    token_secret = 'as_xyz'
    """
    with modal_config(config):
        # A single profile should be fine, even if not explicitly active and named 'default'
        Client.verify(servicer.client_addr, credentials)


def test_import_modal_from_thread(supports_dir):
    # this mainly ensures that we don't make any assumptions about which thread *imports* modal
    # For example, in Python <3.10, creating loop-bound asyncio primitives in global scope would
    # trigger an exception if there is no event loop in the thread (and it's not the main thread)
    subprocess.check_call([sys.executable, supports_dir / "import_modal_from_thread.py"])


@pytest.mark.asyncio
async def test_from_env_container(servicer, container_env):
    servicer.required_creds = {}  # Disallow default client creds
    await Client.from_env.aio()
    # TODO(erikbern): once we no longer run ClientHello by default, add a ping here


@pytest.mark.asyncio
async def test_from_credentials_client(servicer, set_env_client, server_url_env, token_env):
    # Note: this explicitly uses a lot of fixtures to make sure those are ignored
    creds = ("ak-foo-1", "as-bar")
    servicer.required_creds = {creds}
    await Client.from_credentials.aio(*creds)
    # TODO(erikbern): once we no longer run ClientHello by default, add a ping here


@pytest.mark.asyncio
async def test_from_credentials_container(servicer, container_env):
    creds = ("ak-foo-2", "as-bar")
    servicer.required_creds = {creds}
    await Client.from_credentials.aio(*creds)
    # TODO(erikbern): once we no longer run ClientHello by default, add a ping here
