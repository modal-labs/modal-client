# Copyright Modal Labs 2024
import asyncio
import pytest
import time

import jwt

from modal._utils.async_utils import synchronize_api
from modal._utils.auth_token_manager import AuthTokenManager
from modal.exception import RemoteError


@pytest.fixture
def auth_token_manager(client):
    """Create an AuthTokenManager instance for testing."""
    return AuthTokenManager(client.stub)


@pytest.fixture
def valid_jwt_token():
    """Create a valid JWT token with expiry."""
    # Create a JWT with exp claim set to 1 hour from now
    exp_time = int(time.time()) + 3600
    payload = {"exp": exp_time, "type": "valid"}
    return jwt.encode(payload, "my-secret-key", algorithm="HS256")

@pytest.fixture
def another_valid_jwt_token():
    """Create a valid JWT token with expiry."""
    # Create a JWT with exp claim set to 1 hour from now
    exp_time = int(time.time()) + 3600
    payload = {"exp": exp_time, "type": "another_valid"}
    return jwt.encode(payload, "my-secret-key", algorithm="HS256")

@pytest.fixture
def expired_jwt_token():
    """Create an expired JWT token."""
    # Create a JWT with exp claim set to 1 hour ago
    exp_time = int(time.time()) - 3600
    payload = {"exp": exp_time, "type": "expired"}
    return jwt.encode(payload, "my-secret-key", algorithm="HS256")


@pytest.fixture
def token_without_exp():
    """Create a JWT token without exp claim."""
    payload = {"type": "without_exp"}
    return jwt.encode(payload, "my-secret-key", algorithm="HS256")


@pytest.fixture
def token_near_expiry():
    """Create a JWT token that expires in 4 minutes (within refresh window)."""
    exp_time = int(time.time()) + 240  # 4 minutes from now
    payload = {"exp": exp_time, "type": "near_expiry"}
    return jwt.encode(payload, "my-secret-key", algorithm="HS256")


@pytest.mark.asyncio
async def test_get_token_initial_fetch(auth_token_manager, valid_jwt_token, client, servicer):
    """Test getting token when no token exists."""

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    servicer.auth_token = valid_jwt_token
    assert await wrapped_get_token.aio() == valid_jwt_token


@pytest.mark.asyncio
async def test_get_token_cached(auth_token_manager, valid_jwt_token, servicer):
    """Test that cached token is returned without making new request."""

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    # Set up initial token
    servicer.auth_token = valid_jwt_token
    assert await wrapped_get_token.aio() == valid_jwt_token

    # Set a bogus token in the servicer, and verify we get the cached valid token
    servicer.auth_token = "bogus"
    token = await wrapped_get_token.aio()
    assert token == valid_jwt_token


@pytest.mark.asyncio
async def test_get_token_expired(auth_token_manager, expired_jwt_token, valid_jwt_token, servicer):
    """Test that expired token triggers refresh."""
    # Set up expired token
    auth_token_manager._token = expired_jwt_token
    auth_token_manager._expiry = time.time() - 100  # Expired

    # Set up new token in servicer
    servicer.auth_token = valid_jwt_token

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    token = await wrapped_get_token.aio()
    assert token == valid_jwt_token
    assert auth_token_manager._token == valid_jwt_token


@pytest.mark.asyncio
async def test_get_token_needs_refresh(auth_token_manager, token_near_expiry, valid_jwt_token, servicer):
    """Test that token is refreshed when it's close to expiry."""
    # Set up token that expires within refresh window
    auth_token_manager._token = token_near_expiry
    exp_time = time.time() + auth_token_manager.REFRESH_WINDOW - 60
    auth_token_manager._expiry = exp_time

    # Set up new token in servicer
    servicer.auth_token = valid_jwt_token

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    token = await wrapped_get_token.aio()
    assert token == valid_jwt_token
    assert auth_token_manager._token == valid_jwt_token


@pytest.mark.asyncio
async def test_get_token_no_exp_claim(auth_token_manager, token_without_exp, servicer):
    """Test handling of token without exp claim."""
    servicer.auth_token = token_without_exp

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    token = await wrapped_get_token.aio()
    assert token == token_without_exp
    assert auth_token_manager._token == token_without_exp
    # Should use default expiry
    assert auth_token_manager._expiry > time.time()
    assert auth_token_manager._expiry <= time.time() + auth_token_manager.DEFAULT_EXPIRY


@pytest.mark.asyncio
async def test_get_token_empty_response(auth_token_manager, servicer):
    """Test handling of empty token response."""
    servicer.auth_token = ""

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    with pytest.raises(RemoteError, match="Did not receive auth token from server"):
        await wrapped_get_token.aio()


@pytest.mark.asyncio
async def test_get_token_none_response(auth_token_manager, servicer):
    """Test handling of None token response."""
    servicer.auth_token = None

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    with pytest.raises(RemoteError, match="Did not receive auth token from server"):
        await wrapped_get_token.aio()


@pytest.mark.asyncio
async def test_concurrent_token_fetch(auth_token_manager, valid_jwt_token, servicer):
    """Test that concurrent calls don't make multiple requests."""
    servicer.auth_token = valid_jwt_token

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    # Make concurrent calls
    tasks = [wrapped_get_token.aio() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # All should return the same token
    assert all(token == valid_jwt_token for token in results)
    # The server should have been called only once.
    assert servicer.auth_tokens_generated == 1


@pytest.mark.asyncio
async def test_concurrent_refresh(auth_token_manager, token_near_expiry, valid_jwt_token, servicer):
    """Test that when get_token is called concurrently, test that old but valid token is returned."""
    # Set up token that needs refresh
    auth_token_manager._token = "old.but.valid.token"
    auth_token_manager._expiry = time.time() + 240  # 4 minutes from now

    # Set up new token in servicer
    servicer.auth_token = valid_jwt_token

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    # Make concurrent calls
    tasks = [wrapped_get_token.aio() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # At least one call should have returned the new token
    assert valid_jwt_token in results
    # When called concurrently, only one coroutine should fetch a new token, and the others should use the older but
    # still valid token to improve throughput. Note, this isn't guaranteed, just very likely. May need to fix if flakey.
    assert "old.but.valid.token" in results
    # The new token should be cached
    assert auth_token_manager._token == valid_jwt_token


def test_decode_jwt_valid(valid_jwt_token):
    """Test JWT decoding with valid token."""
    decoded = AuthTokenManager._decode_jwt(valid_jwt_token)
    assert "exp" in decoded
    assert "type" in decoded
    assert decoded["type"] == "valid"


def test_decode_jwt_without_exp(token_without_exp):
    """Test JWT decoding with token that has no exp claim."""
    decoded = AuthTokenManager._decode_jwt(token_without_exp)
    assert "exp" not in decoded
    assert "type" in decoded
    assert decoded["type"] == "without_exp"


def test_decode_jwt_invalid_format():
    """Test JWT decoding with invalid token format."""
    with pytest.raises(ValueError):
        AuthTokenManager._decode_jwt("invalid.token")


def test_needs_refresh_true(auth_token_manager):
    """Test _needs_refresh returns True when token expires soon."""
    # Set expiry to 4 minutes from now (within refresh window)
    auth_token_manager._expiry = time.time() + 240
    assert auth_token_manager._needs_refresh() is True


def test_needs_refresh_false(auth_token_manager):
    """Test _needs_refresh returns False when token is not close to expiry."""
    # Set expiry to 10 minutes from now (outside refresh window)
    auth_token_manager._expiry = time.time() + 600
    assert auth_token_manager._needs_refresh() is False


def test_is_expired_true(auth_token_manager):
    """Test _is_expired returns True for expired token."""
    # Set expiry to 1 minute ago
    auth_token_manager._expiry = time.time() - 60
    assert auth_token_manager._is_expired() is True


def test_is_expired_false(auth_token_manager):
    """Test _is_expired returns False for valid token."""
    # Set expiry to 1 minute from now
    auth_token_manager._expiry = time.time() + 60
    assert auth_token_manager._is_expired() is False


@pytest.mark.asyncio
async def test_multiple_refresh_cycles(auth_token_manager, servicer):
    """Test multiple refresh cycles work correctly."""
    exp_time = int(time.time()) + 3600
    tokens = [jwt.encode({"exp": exp_time, "name": "t0"}, "my-secret-key", algorithm="HS256"),
              jwt.encode({"exp": exp_time, "name": "t1"}, "my-secret-key", algorithm="HS256"),
              jwt.encode({"exp": exp_time, "name": "t2"}, "my-secret-key", algorithm="HS256")]

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    # First call
    servicer.auth_token = tokens[0]
    token0 = await wrapped_get_token.aio()
    assert token0 == tokens[0]

    # Expire the token
    auth_token_manager._expiry = time.time() - 100

    # Second call
    servicer.auth_token = tokens[1]
    token1 = await wrapped_get_token.aio()
    assert token1 == tokens[1]

    # Expire again
    auth_token_manager._expiry = time.time() - 100

    # Third call
    servicer.auth_token = tokens[2]
    token2 = await wrapped_get_token.aio()
    assert token2 == tokens[2]

