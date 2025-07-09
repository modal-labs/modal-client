# Copyright Modal Labs 2025
import asyncio
import base64
import json
import time
from typing import Any

from modal.exception import RemoteError
from modal_proto import api_pb2, modal_api_grpc

from .grpc_utils import retry_transient_errors
from .logger import logger


class AuthTokenManager:
    """ Handles fetching and refreshing of the input plane auth token. """

    # Start refreshing this many seconds before the token expires
    REFRESH_WINDOW = 5 * 60
    # If the token doesn't have an expiry use this default (not expected)
    DEFAULT_EXPIRY = 20 * 60

    def __init__(self, stub: "modal_api_grpc.ModalClientModal"):
        self._stub = stub
        self._token = ""
        self._expiry = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self):
        """
        If we do not have a token, or if the current one is expired, fetch a new token and cache it.
        If the cached token will expire in the next 5 minutes, fetch a new one. If called concurrently, only one
        coroutine will fetch a new token, and the others will continue to use the old token without blocking.
        """
        if self._token is None or self._is_expired():
            # We either have no token or it is expired - block everyone until we get a new token
            await self._refresh_token()
        elif self._needs_refresh():
            # The token hasn't expired yet, but will soon, so it needs a refresh.
            if self._lock.locked():
                # The lock is taken, so someone else is refreshing. Continue to use the old token.
                return self._token
            else:
                # The lock is not taken, so we need to fetch a new token.
                await self._refresh_token()

        return self._token

    async def _refresh_token(self):
        """
        Fetch a new token from the control plane. If called concurrently, only one coroutine will make a request for a
        new token. The others will block on a lock, until the first coroutine has fetched the new token.
        """
        async with self._lock:
            # Double check inside lock: maybe another coroutine refreshed already
            if self._token and not self._needs_refresh():
                return
            resp: api_pb2.AuthTokenGetResponse = await retry_transient_errors(
                self._stub.AuthTokenGet, api_pb2.AuthTokenGetRequest()
            )
            if not resp.token:
                # Not expected
                raise RemoteError("Did not receive auth token from server")

            self._token = resp.token
            if exp := self._decode_jwt(resp.token).get("exp"):
                self._expiry = float(exp)
            else:
                # This should never happen.
                logger.warning("x-modal-auth-token does not contain exp field")
                # We'll use the token, and set the expiry to 20 min from now.
                self._expiry = time.time() + self.DEFAULT_EXPIRY

    @staticmethod
    def _decode_jwt(token: str) -> dict[str, Any]:
        """Decodes a JWT into a dict without verifying signature."""
        try:
            payload = token.split(".")[1]
            padding = "=" * (-len(payload) % 4)
            decoded_bytes = base64.urlsafe_b64decode(payload + padding)
            return json.loads(decoded_bytes)
        except Exception as e:
            raise ValueError("Cannot parse auth token") from e

    def _needs_refresh(self):
        return time.time() >= (self._expiry - self.REFRESH_WINDOW)

    def _is_expired(self):
        return time.time() >= self._expiry
