# Copyright Modal Labs 2025
import asyncio
import base64
import json
import time
from typing import Any

from modal._utils.grpc_utils import DEFAULT_MAX_RETRIES, Retry
from modal.exception import AuthError, ExecutionError, PermissionDeniedError
from modal_proto import api_pb2, modal_api_grpc

from .logger import logger

AUTH_DENIED_EXCEPTIONS = (AuthError, PermissionDeniedError)


class _AuthTokenManager:
    """Handles fetching and refreshing of the input plane auth token."""

    # Start refreshing this many seconds before the token expires
    REFRESH_WINDOW = 5 * 60
    # Bound each AuthTokenGet attempt so a refresh can't hang.
    AUTH_TOKEN_TIMEOUT = 5.0
    AUTH_TOKEN_RETRY_TOTAL_TIMEOUT = 3 * AUTH_TOKEN_TIMEOUT
    # After a failed refresh, wait before hitting the server again, growing exponentially with
    # consecutive failures between these bounds (in seconds).
    FAILURE_BACKOFF_BASE = 0.5
    FAILURE_BACKOFF_MAX = 60.0
    # If the token doesn't have an expiry field, default to current time plus this value (not expected).
    DEFAULT_EXPIRY_OFFSET = 20 * 60

    def __init__(self, stub: "modal_api_grpc.ModalClientModal"):
        self._stub = stub
        self._token = ""
        self._expiry = 0.0
        self._retry_after = 0.0
        self._backoff = self.FAILURE_BACKOFF_BASE
        self._lock: asyncio.Lock | None = None

    async def get_token(self) -> str:
        """
        When called, the AuthTokenManager can be in one of three states:
        1. Has a valid cached token. It is returned to the caller.
        2. Has no cached token, or the token is expired. We fetch a new one and cache it. If `get_token` is called
        concurrently by multiple coroutines, all requests will block until the token has been fetched. But only one
        coroutine will actually make a request to the control plane to fetch the new token. This ensures we do not hit
        the control plane with more requests than needed.
        3. Has a valid cached token, but it is going to expire in the next 5 minutes. In this case we fetch a new token
        and cache it. If `get_token` is called concurrently, only one request will fetch the new token, and the others
        will be given the old (but still valid) token - i.e. they will not block.
        """
        if not self._token or self._is_expired():
            # We either have no token or it is expired - block everyone until we get a new token
            await self._try_refresh_token()
        elif self._should_refresh():
            # The token hasn't expired yet, but will soon, so it needs a refresh.
            lock = await self._get_lock()
            if lock.locked():
                # The lock is taken, so someone else is refreshing. Continue to use the old token.
                return self._token
            else:
                # The lock is not taken, so we need to fetch a new token.
                await self._try_refresh_token()

        return self._token

    async def _try_refresh_token(self):
        try:
            await self._refresh_token()
        except AUTH_DENIED_EXCEPTIONS:
            raise
        except Exception as e:
            if not self._token:
                raise
            logger.warning("Auth token refresh failed; falling back to cached token: %s", e)

    async def _refresh_token(self):
        """
        Fetch a new token from the control plane. If called concurrently, only one coroutine will make a request for a
        new token. The others will block on a lock, until the first coroutine has fetched the new token.
        """
        lock = await self._get_lock()
        async with lock:
            # Double check inside lock - maybe another coroutine refreshed already. This happens the first time we fetch
            # the token. The first coroutine will fetch the token, while the others block on the lock, waiting for the
            # new token. Once we have a new token, the other coroutines will unblock and return from here.
            if self._token and not self._should_refresh():
                return
            await self._fetch_token()

    async def _fetch_token(self):
        """Make the AuthTokenGet request and cache its token, or arm the failure backoff."""
        try:
            resp: api_pb2.AuthTokenGetResponse = await self._stub.AuthTokenGet(
                api_pb2.AuthTokenGetRequest(),
                # No cached token to fall back on, so a failure is user-visible: retry transient errors.
                # Otherwise one attempt, and _retry_after handles the cooldown.
                retry=Retry(
                    attempt_timeout=self.AUTH_TOKEN_TIMEOUT,
                    total_timeout=self.AUTH_TOKEN_RETRY_TOTAL_TIMEOUT,
                    max_retries=0 if self._token else DEFAULT_MAX_RETRIES,
                ),
            )
            if not resp.token:
                # Not expected
                raise ExecutionError(
                    "Internal error: Did not receive auth token from server. Please contact Modal support."
                )
            if exp := self._decode_jwt(resp.token).get("exp"):
                expiry = float(exp)
            else:
                # This should never happen.
                logger.warning("x-modal-auth-token does not contain exp field")
                expiry = time.time() + self.DEFAULT_EXPIRY_OFFSET
            self._token = resp.token
            self._expiry = expiry
            self._retry_after = 0.0
            self._backoff = self.FAILURE_BACKOFF_BASE
        except AUTH_DENIED_EXCEPTIONS:
            raise
        except Exception as e:
            # Back off (exponentially on consecutive failures) so we don't hammer a struggling server.
            self._retry_after = time.time() + self._backoff
            self._backoff = min(self._backoff * 2, self.FAILURE_BACKOFF_MAX)
            raise

    async def _get_lock(self) -> asyncio.Lock:
        # Note: this function runs no async code but is marked as async to ensure it's
        # being run inside the synchronicity event loop and binds the lock to the
        # correct event loop on Python 3.9 which eagerly assigns event loops on
        # constructions of locks
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    def _decode_jwt(token: str) -> dict[str, Any]:
        """
        Decodes a JWT into a dict without verifying signature. We do this manually instead of using a library to avoid
        adding another dependency to the client.
        """
        try:
            payload = token.split(".")[1]
            padding = "=" * (-len(payload) % 4)
            decoded_bytes = base64.urlsafe_b64decode(payload + padding)
            return json.loads(decoded_bytes)
        except Exception as e:
            raise ValueError("Internal error: Cannot parse auth token. Please contact Modal support.") from e

    def _in_backoff(self):
        return time.time() < self._retry_after

    def _should_refresh(self):
        # Fetch only when the token is stale/expiring and we're not in a post-failure backoff.
        return self._needs_refresh() and not self._in_backoff()

    def _needs_refresh(self):
        return time.time() >= (self._expiry - self.REFRESH_WINDOW)

    def _is_expired(self):
        return time.time() >= self._expiry
