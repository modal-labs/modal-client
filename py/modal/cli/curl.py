# Copyright Modal Labs 2026
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import click

from modal._utils.async_utils import synchronizer
from modal._utils.curl_utils import endpoint_cache_host, find_url
from modal.client import _Client
from modal_proto import api_pb2

from ._help import ModalCommand

_FLASH_ENDPOINT_AUTH_CACHE_TTL_SECONDS = 2 * 60
_FLASH_ENDPOINT_AUTH_REFRESH_WINDOW_SECONDS = 15


def _cache_file_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache_root / "modal" / "curl-flash-auth-tokens.json"


def _cache_key(url: str) -> str:
    return endpoint_cache_host(url)


def _decode_jwt_exp(token: str) -> float | None:
    try:
        payload = token.split(".")[1]
        padding = "=" * (-len(payload) % 4)
        decoded_bytes = base64.urlsafe_b64decode(payload + padding)
        exp = json.loads(decoded_bytes).get("exp")
        return float(exp) if exp else None
    except Exception:
        return None


def _cache_expiry(token: str, now: float) -> float:
    if exp := _decode_jwt_exp(token):
        return min(exp - _FLASH_ENDPOINT_AUTH_REFRESH_WINDOW_SECONDS, now + _FLASH_ENDPOINT_AUTH_CACHE_TTL_SECONDS)
    return now + _FLASH_ENDPOINT_AUTH_CACHE_TTL_SECONDS


def _read_cached_token(cache_key: str, now: float) -> str | None:
    try:
        cache_data = json.loads(_cache_file_path().read_text())
        entry = cache_data.get(cache_key, {})
        if entry.get("expires_at", 0) > now:
            return entry.get("token") or None
    except Exception:
        return None
    return None


def _write_cached_token(cache_key: str, token: str, now: float) -> None:
    if not token:
        return
    cache_path = _cache_file_path()
    try:
        cache_data = json.loads(cache_path.read_text())
    except Exception:
        cache_data = {}

    cache_data = {
        key: value for key, value in cache_data.items() if isinstance(value, dict) and value.get("expires_at", 0) > now
    }
    cache_data[cache_key] = {"token": token, "expires_at": _cache_expiry(token, now)}

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        with os.fdopen(os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(cache_data, f)
        os.replace(tmp_path, cache_path)
        cache_path.chmod(0o600)
    except Exception:
        pass


async def _get_flash_endpoint_auth_token(client: _Client, url: str, cache_key: str, now: float) -> str:
    resp = await client.stub.CurlGetAuthToken(api_pb2.CurlAuthTokenRequest(url=url))
    _write_cached_token(cache_key, resp.token, now)
    return resp.token


def _run_curl(curl_args: tuple[str, ...], token: str | None = None) -> None:
    cmd = ["curl"]
    if token:
        cmd += ["-H", f"Modal-Authorization: Bearer {token}"]
    cmd += list(curl_args)
    sys.exit(subprocess.call(cmd))


@click.command(
    "curl",
    cls=ModalCommand,
    no_args_is_help=True,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("curl_args", nargs=-1, required=True, type=click.UNPROCESSED)
@synchronizer.create_blocking
async def curl(curl_args: tuple[str, ...]):
    """Send an authenticated request to a Modal endpoint.

    Experimental: This command may change or be removed in the future.

    This command allows you to send authenticated requests without including proxy token
    headers. Authentication is managed via your local Modal API credentials. API-based
    authentication adds latency to requests, so this utility is recommended only for
    experimentation and debugging purposes.

    All arguments are passed directly to `curl`, which must be installed locally.

    Examples:

    ```bash
    modal curl https://user--my-app.us-west.modal.direct
    modal curl -X GET https://user--my-app.us-west.modal.direct
    ```

    """
    url = find_url(curl_args)
    if url is None:
        _run_curl(curl_args)
        return

    now = time.time()
    cache_key = _cache_key(url)
    if token := _read_cached_token(cache_key, now):
        _run_curl(curl_args, token)
        return

    client = await _Client.from_env()
    try:
        token = await _get_flash_endpoint_auth_token(client, url, cache_key, now)
    except Exception as e:
        raise click.ClickException(f"Error getting proxy token: {e}") from e

    _run_curl(curl_args, token)
