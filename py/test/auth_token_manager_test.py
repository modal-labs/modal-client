# Copyright Modal Labs 2024
import asyncio
import pytest
import time

import jwt

from modal._utils.async_utils import synchronize_api
from modal._utils.auth_token_manager import _AuthTokenManager
from modal._utils.grpc_utils import DEFAULT_MAX_RETRIES
from modal.exception import AuthError, ExecutionError, PermissionDeniedError
from modal_proto import api_pb2

SECRET_KEY = "your-super-flexible-and-long-shared-secret-key"


@pytest.fixture
def auth_token_manager(client):
    """Create an AuthTokenManager instance for testing."""
    return _AuthTokenManager(client.stub)


@pytest.fixture
def valid_jwt_token():
    """Create a valid JWT token with expiry."""
    # Create a JWT with exp claim set to 1 hour from now
    exp = int(time.time()) + 3600
    payload = {"exp": exp, "type": "valid"}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


@pytest.fixture
def another_valid_jwt_token():
    """Create a valid JWT token with expiry."""
    # Create a JWT with exp claim set to 1 hour from now
    exp = int(time.time()) + 3600
    payload = {"exp": exp, "type": "another_valid"}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


@pytest.fixture
def expired_jwt_token():
    """Create an expired JWT token."""
    # Create a JWT with exp claim set to 1 hour ago
    exp = int(time.time()) - 3600
    payload = {"exp": exp, "type": "expired"}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


@pytest.fixture
def token_without_exp():
    """Create a JWT token without exp claim."""
    payload = {"type": "without_exp"}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


@pytest.fixture
def token_due_for_refresh():
    """Create a JWT token that is still valid but has used up its refresh fraction."""
    exp = int(time.time()) + 240  # 4 minutes from now
    payload = {"exp": exp, "type": "due_for_refresh"}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


@pytest.mark.asyncio
async def test_get_token_initial_fetch(auth_token_manager, valid_jwt_token, client, servicer):
    """Test getting token when no token exists."""

    # All these tests wrap get_token with @synchronize_api because they hang forever without it.
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
    auth_token_manager._set_expiry(exp_time(expired_jwt_token))

    # Set up new token in servicer
    servicer.auth_token = valid_jwt_token

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    token = await wrapped_get_token.aio()
    assert token == valid_jwt_token
    assert auth_token_manager._token == valid_jwt_token


@pytest.mark.asyncio
async def test_get_token_needs_refresh(auth_token_manager, token_due_for_refresh, valid_jwt_token, servicer):
    """Test that a still-valid token is refreshed once it is past its refresh point."""
    auth_token_manager._token = token_due_for_refresh
    auth_token_manager._expiry = exp_time(token_due_for_refresh)
    auth_token_manager._refresh_at = time.time() - 1

    # Set up new token in servicer
    servicer.auth_token = valid_jwt_token

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    token = await wrapped_get_token.aio()
    assert token == valid_jwt_token
    assert auth_token_manager._token == valid_jwt_token


@pytest.mark.asyncio
async def test_get_token_expired_refresh_failure_falls_back(auth_token_manager, expired_jwt_token, monkeypatch):
    """Test that a failed refresh falls back to the expired cached token."""
    auth_token_manager._token = expired_jwt_token
    auth_token_manager._expiry = exp_time(expired_jwt_token)

    async def fail_auth_token_get(*args, **kwargs):
        raise RuntimeError("auth server unavailable")

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", fail_auth_token_get)

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    assert await wrapped_get_token.aio() == expired_jwt_token


@pytest.mark.asyncio
async def test_get_token_without_cached_token_refresh_failure_raises(auth_token_manager, monkeypatch):
    """Test that a failed initial fetch still raises without a cached token."""

    async def fail_auth_token_get(*args, **kwargs):
        raise RuntimeError("auth server unavailable")

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", fail_auth_token_get)

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    with pytest.raises(RuntimeError, match="auth server unavailable"):
        await wrapped_get_token.aio()


@pytest.mark.asyncio
async def test_get_token_fetch_only_retries_without_cached_token(auth_token_manager, valid_jwt_token, monkeypatch):
    """A failed fetch is only user-visible without a cached token, so that's the only case we retry."""
    captured: dict = {}

    async def recording_auth_token_get(request, **kwargs):
        captured.update(kwargs)
        return api_pb2.AuthTokenGetResponse(token=valid_jwt_token)

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", recording_auth_token_get)

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    assert await wrapped_get_token.aio() == valid_jwt_token
    assert captured["retry"].max_retries == DEFAULT_MAX_RETRIES

    # The token is now cached, so a subsequent refresh falls back to it instead of retrying.
    auth_token_manager._set_expiry(0.0)
    assert await wrapped_get_token.aio() == valid_jwt_token
    assert captured["retry"].max_retries == 0


@pytest.mark.asyncio
async def test_get_token_expired_refresh_failure_backs_off(auth_token_manager, expired_jwt_token, monkeypatch):
    """After a failed refresh, the cached token is reused without re-hitting the server during the backoff window."""
    auth_token_manager._token = expired_jwt_token
    auth_token_manager._expiry = exp_time(expired_jwt_token)

    calls = 0

    async def fail_auth_token_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("auth server unavailable")

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", fail_auth_token_get)

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    # First call attempts a refresh, fails, and falls back to the cached expired token.
    assert await wrapped_get_token.aio() == expired_jwt_token
    assert calls == 1
    assert auth_token_manager._in_backoff()
    expected_retry_after = time.time() + auth_token_manager.FAILURE_BACKOFF_BASE
    assert auth_token_manager._retry_after == pytest.approx(expected_retry_after, abs=1)

    # Second call is within the backoff window, so it returns the cached token without another RPC.
    assert await wrapped_get_token.aio() == expired_jwt_token
    assert calls == 1


@pytest.mark.asyncio
async def test_get_token_empty_response_backs_off(auth_token_manager, expired_jwt_token, monkeypatch):
    """An empty refresh response falls back to the cached token and enters backoff."""
    auth_token_manager._token = expired_jwt_token
    auth_token_manager._expiry = exp_time(expired_jwt_token)
    calls = 0

    async def empty_auth_token_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return api_pb2.AuthTokenGetResponse()

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", empty_auth_token_get)

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    assert await wrapped_get_token.aio() == expired_jwt_token
    assert calls == 1
    assert auth_token_manager._in_backoff()
    assert await wrapped_get_token.aio() == expired_jwt_token
    assert calls == 1


@pytest.mark.asyncio
async def test_get_token_undecodable_response_backs_off(auth_token_manager, expired_jwt_token, monkeypatch):
    """An undecodable refresh response falls back to the cached token and enters backoff."""
    auth_token_manager._token = expired_jwt_token
    auth_token_manager._expiry = exp_time(expired_jwt_token)
    calls = 0

    async def undecodable_auth_token_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return api_pb2.AuthTokenGetResponse(token="not-a-jwt")

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", undecodable_auth_token_get)

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    assert await wrapped_get_token.aio() == expired_jwt_token
    assert calls == 1
    assert auth_token_manager._in_backoff()
    assert await wrapped_get_token.aio() == expired_jwt_token
    assert calls == 1


@pytest.mark.asyncio
async def test_get_token_refresh_retried_after_backoff(
    auth_token_manager, expired_jwt_token, valid_jwt_token, servicer
):
    """Once the backoff window elapses, the next call refreshes again."""
    auth_token_manager._token = expired_jwt_token
    auth_token_manager._expiry = exp_time(expired_jwt_token)
    # Simulate an active backoff from a recent failure.
    auth_token_manager._retry_after = time.time() + auth_token_manager.FAILURE_BACKOFF_BASE
    servicer.auth_token = valid_jwt_token

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    # Within the backoff window: cached expired token is returned, no refresh.
    assert await wrapped_get_token.aio() == expired_jwt_token

    # Backoff has elapsed: the refresh happens and the new token is cached.
    auth_token_manager._retry_after = 0.0
    assert await wrapped_get_token.aio() == valid_jwt_token
    assert auth_token_manager._token == valid_jwt_token


@pytest.mark.parametrize("error_type", [AuthError, PermissionDeniedError])
@pytest.mark.asyncio
async def test_get_token_auth_denied_does_not_fall_back(auth_token_manager, expired_jwt_token, error_type, monkeypatch):
    """Auth-denial errors are surfaced instead of falling back to an expired cached token."""
    auth_token_manager._token = expired_jwt_token
    auth_token_manager._expiry = exp_time(expired_jwt_token)
    auth_token_manager._retry_after = 0.0
    auth_token_manager._backoff = auth_token_manager.FAILURE_BACKOFF_BASE
    error = error_type("credentials rejected")

    async def fail_auth_token_get(*args, **kwargs):
        raise error

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", fail_auth_token_get)

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    with pytest.raises(error_type, match="credentials rejected"):
        await wrapped_get_token.aio()
    assert auth_token_manager._token == expired_jwt_token
    assert auth_token_manager._retry_after == 0.0
    assert auth_token_manager._backoff == auth_token_manager.FAILURE_BACKOFF_BASE
    assert not auth_token_manager._in_backoff()


@pytest.mark.asyncio
async def test_get_token_failure_backoff_grows_exponentially(
    auth_token_manager, expired_jwt_token, valid_jwt_token, monkeypatch
):
    """Consecutive refresh failures increase the cooldown, which resets after success."""
    now = 1_000.0
    monkeypatch.setattr(time, "time", lambda: now)
    auth_token_manager._token = expired_jwt_token
    auth_token_manager._set_expiry(0.0)
    should_fail = True

    async def auth_token_get(*args, **kwargs):
        if should_fail:
            raise RuntimeError("auth server unavailable")
        return api_pb2.AuthTokenGetResponse(token=valid_jwt_token)

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", auth_token_get)

    expected_cooldowns = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    for expected_cooldown in expected_cooldowns:
        auth_token_manager._retry_after = 0.0
        with pytest.raises(RuntimeError, match="auth server unavailable"):
            await auth_token_manager._refresh_token()
        assert auth_token_manager._retry_after == now + expected_cooldown

    should_fail = False
    auth_token_manager._retry_after = 0.0
    await auth_token_manager._refresh_token()
    assert auth_token_manager._backoff == auth_token_manager.FAILURE_BACKOFF_BASE

    should_fail = True
    auth_token_manager._set_expiry(0.0)
    auth_token_manager._retry_after = 0.0
    with pytest.raises(RuntimeError, match="auth server unavailable"):
        await auth_token_manager._refresh_token()
    assert auth_token_manager._retry_after == now + auth_token_manager.FAILURE_BACKOFF_BASE


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
    assert auth_token_manager._expiry <= time.time() + auth_token_manager.DEFAULT_EXPIRY_OFFSET
    # And should schedule the refresh partway through that lifetime.
    assert time.time() < auth_token_manager._refresh_at < auth_token_manager._expiry


@pytest.mark.asyncio
async def test_get_token_empty_response(auth_token_manager, servicer):
    """Test handling of empty token response."""
    servicer.auth_token = ""

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    with pytest.raises(ExecutionError):
        await wrapped_get_token.aio()


@pytest.mark.asyncio
async def test_get_token_none_response(auth_token_manager, servicer):
    """Test handling of None token response."""
    servicer.auth_token = None

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    with pytest.raises(ExecutionError):
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
async def test_concurrent_refresh(auth_token_manager, token_due_for_refresh, valid_jwt_token, servicer):
    """Test that when get_token is called concurrently, test that old but valid token is returned."""
    # Set up token that needs refresh
    auth_token_manager._token = "old.but.valid.token"
    auth_token_manager._expiry = exp_time(token_due_for_refresh)
    auth_token_manager._refresh_at = time.time() - 1

    # Set up new token in servicer
    servicer.auth_token = valid_jwt_token
    servicer.auth_token_delay = 0.5

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


@pytest.mark.asyncio
async def test_concurrent_fetch_failure_makes_one_request(auth_token_manager, monkeypatch):
    """Concurrent callers with no cached token share one failing request instead of one request each."""
    calls = 0

    async def fail_auth_token_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.1)
        raise RuntimeError("auth server unavailable")

    monkeypatch.setattr(auth_token_manager._stub, "AuthTokenGet", fail_auth_token_get)

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    results = await asyncio.gather(*[wrapped_get_token.aio() for _ in range(5)], return_exceptions=True)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert calls == 1


@pytest.mark.asyncio
async def test_cancelling_a_caller_does_not_cancel_the_refresh(auth_token_manager, valid_jwt_token, servicer):
    """The refresh runs to completion for the remaining callers when the caller that started it is cancelled."""
    servicer.auth_token = valid_jwt_token
    servicer.auth_token_delay = 0.5

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    first = asyncio.create_task(wrapped_get_token.aio())
    await asyncio.sleep(0.1)
    second = asyncio.create_task(wrapped_get_token.aio())
    await asyncio.sleep(0.1)
    first.cancel()

    assert await second == valid_jwt_token
    assert servicer.auth_tokens_generated == 1


def test_decode_jwt_valid(valid_jwt_token):
    """Test JWT decoding with valid token."""
    decoded = _AuthTokenManager._decode_jwt(valid_jwt_token)
    assert "exp" in decoded
    assert "type" in decoded
    assert decoded["type"] == "valid"


def test_decode_jwt_without_exp(token_without_exp):
    """Test JWT decoding with token that has no exp claim."""
    decoded = _AuthTokenManager._decode_jwt(token_without_exp)
    assert "exp" not in decoded
    assert "type" in decoded
    assert decoded["type"] == "without_exp"


def test_decode_jwt_invalid_format():
    """Test JWT decoding with invalid token format."""
    with pytest.raises(ValueError):
        _AuthTokenManager._decode_jwt("invalid.token")


def test_needs_refresh_true(auth_token_manager):
    """Test _needs_refresh returns True once the refresh point has passed."""
    auth_token_manager._refresh_at = time.time() - 1
    assert auth_token_manager._needs_refresh() is True


def test_needs_refresh_false(auth_token_manager):
    """Test _needs_refresh returns False before the refresh point."""
    auth_token_manager._refresh_at = time.time() + 600
    assert auth_token_manager._needs_refresh() is False


def test_set_expiry_of_expired_token_refreshes_immediately(auth_token_manager):
    """Test an already-expired token is due for refresh rather than getting a negative delay."""
    auth_token_manager._set_expiry(time.time() - 100)
    assert auth_token_manager._needs_refresh() is True


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
    exp = int(time.time()) + 3600
    tokens = [
        jwt.encode({"exp": exp, "name": "t0"}, SECRET_KEY, algorithm="HS256"),
        jwt.encode({"exp": exp, "name": "t1"}, SECRET_KEY, algorithm="HS256"),
        jwt.encode({"exp": exp, "name": "t2"}, SECRET_KEY, algorithm="HS256"),
    ]

    @synchronize_api
    async def wrapped_get_token():
        return await auth_token_manager.get_token()

    # First call
    servicer.auth_token = tokens[0]
    token0 = await wrapped_get_token.aio()
    assert token0 == tokens[0]

    # Expire the token
    auth_token_manager._set_expiry(time.time() - 100)

    # Second call
    servicer.auth_token = tokens[1]
    token1 = await wrapped_get_token.aio()
    assert token1 == tokens[1]

    # Expire again
    auth_token_manager._set_expiry(time.time() - 100)

    # Third call
    servicer.auth_token = tokens[2]
    token2 = await wrapped_get_token.aio()
    assert token2 == tokens[2]


def exp_time(token: str):
    return jwt.decode(token, options={"verify_signature": False})["exp"]
