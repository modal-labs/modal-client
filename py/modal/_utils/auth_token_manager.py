# Copyright Modal Labs 2025
import asyncio
import base64
import json
import random
import time
from typing import Any

from modal._utils.grpc_utils import DEFAULT_MAX_RETRIES, Retry
from modal.exception import AuthError, ExecutionError, PermissionDeniedError
from modal_proto import api_pb2, modal_api_grpc

from .logger import logger

AUTH_DENIED_EXCEPTIONS = (AuthError, PermissionDeniedError)


class _AuthTokenManager:
    """Handles fetching and refreshing of the input plane auth token."""

    # Fraction of the token's lifetime to use before refreshing.
    REFRESH_FRACTION = 0.5
    # Width of the random band added on top of REFRESH_FRACTION, so that clients issued
    # same-lifetime tokens at the same moment (a fan-out of containers starting together)
    # spread their refreshes out instead of refreshing in lockstep every cycle. The band
    # only ever delays a refresh, so the margin before expiry stays at no less than
    # 1 - REFRESH_FRACTION - REFRESH_JITTER of the lifetime.
    REFRESH_JITTER = 0.1
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
        self._refresh_at = 0.0
        self._retry_after = 0.0
        self._backoff = self.FAILURE_BACKOFF_BASE
        self._refresh_task: asyncio.Task | None = None

    async def get_token(self) -> str:
        """
        When called, the AuthTokenManager can be in one of three states:
        1. Has a valid cached token. It is returned to the caller.
        2. Has no cached token, or the token is expired. We fetch a new one and cache it. If `get_token` is called
        concurrently by multiple coroutines, all requests will block until the token has been fetched. But only one
        coroutine will actually make a request to the control plane to fetch the new token. This ensures we do not hit
        the control plane with more requests than needed.
        3. Has a valid cached token that has used up its refresh fraction of its lifetime. In this case we fetch a new
        token and cache it. If `get_token` is called concurrently, only one request will fetch the new token, and the
        others will be given the old (but still valid) token - i.e. they will not block.
        """
        if not self._token or self._is_expired():
            # We either have no token or it is expired - block everyone until we get a new token
            await self._try_refresh_token()
        elif self._should_refresh():
            # The token is still valid, but has used up its refresh fraction of its lifetime.
            if self._refresh_task is not None:
                # A refresh is in flight, so someone else is refreshing. Continue to use the old token.
                return self._token
            else:
                # Nobody is refreshing, so we need to fetch a new token.
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
        Fetch a new token from the control plane. Concurrent callers share a single request: the first caller starts it
        and the others await the same task, so a burst of callers results in one RPC whether it succeeds or fails. The
        task is owned by the event loop rather than by the caller that started it, so it runs to completion even if that
        caller goes away.
        """
        if self._refresh_task is None:
            # Maybe another coroutine refreshed already, or we're cooling down after a failure.
            if self._token and not self._should_refresh():
                return
            # Note: the task is created in async code so it binds to the synchronicity event loop.
            self._refresh_task = asyncio.create_task(self._fetch_token())
            self._refresh_task.add_done_callback(self._refresh_task_done)
        # Shielded so a cancelled caller doesn't cancel the refresh for everyone else.
        await asyncio.shield(self._refresh_task)

    def _refresh_task_done(self, task: "asyncio.Task"):
        if self._refresh_task is task:
            self._refresh_task = None
        if not task.cancelled():
            # Retrieve the exception, if any, so a failure isn't logged as unhandled when no caller is waiting.
            task.exception()

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
            self._set_expiry(expiry)
            self._retry_after = 0.0
            self._backoff = self.FAILURE_BACKOFF_BASE
        except AUTH_DENIED_EXCEPTIONS:
            raise
        except Exception as e:
            # Back off (exponentially on consecutive failures) so we don't hammer a struggling server.
            self._retry_after = time.time() + self._backoff
            self._backoff = min(self._backoff * 2, self.FAILURE_BACKOFF_MAX)
            raise

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

    def _set_expiry(self, expiry: float) -> None:
        """Record a new expiry and schedule the proactive refresh a jittered fraction of the way through.

        The delay is measured from the remaining lifetime as this client sees it, so a client whose clock is
        skewed relative to the issuer still keeps a margin proportional to the lifetime it observes.
        """
        now = time.time()
        self._expiry = expiry
        remaining = max(expiry - now, 0.0)
        fraction = self.REFRESH_FRACTION + random.random() * self.REFRESH_JITTER
        self._refresh_at = now + remaining * fraction

    def _needs_refresh(self):
        return time.time() >= self._refresh_at

    def _is_expired(self):
        return time.time() >= self._expiry
