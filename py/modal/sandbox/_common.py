# Copyright Modal Labs 2026
import asyncio
import enum
import re
import time
import typing
from collections.abc import Awaitable, Sequence
from typing import TYPE_CHECKING, Any

from modal_proto import api_pb2

from .._image import _Image
from ..client import _Client
from ..exception import (
    InvalidError,
)
from ..types import SandboxRuntime


async def _gather_load_with_timings(
    load_coros: Sequence[Awaitable[Any]],
) -> list[tuple[str, float]]:
    """Await all loader coroutines concurrently and return [(object_id, elapsed_seconds)] per load."""
    timings: list[tuple[str, float]] = []

    async def timed(coro: Awaitable[Any]) -> None:
        start = time.monotonic()
        obj = await coro
        timings.append((obj.object_id, time.monotonic() - start))

    await asyncio.gather(*(timed(c) for c in load_coros))
    return timings


def _format_sandbox_create_timing_log(
    sandbox_id: str,
    total_seconds: float,
    rpc_seconds: float,
    dep_timings: Sequence[tuple[str, float]],
) -> str:
    """Format the Sandbox create debug log line, listing the slowest deps first."""
    if dep_timings:
        deps_sorted = sorted(dep_timings, key=lambda t: t[1], reverse=True)
        shown = deps_sorted[:10]
        dep_summary = ", ".join(f"{label}: {elapsed:.2f}s" for label, elapsed in shown)
        if len(deps_sorted) > 10:
            dep_summary += f", +{len(deps_sorted) - 10} more"
    else:
        dep_summary = "none"
    return (
        f"Sandbox {sandbox_id} created in {total_seconds:.2f}s "
        f"(create rpc: {rpc_seconds:.2f}s; dependencies: {dep_summary})"
    )


# The maximum number of bytes that can be passed to an exec on Linux.
# Though this is technically a 'server side' limit, it is unlikely to change.
# getconf ARG_MAX will show this value on a host.
#
# By probing in production, the limit is 131072 bytes (2**17).
# We need some bytes of overhead for the rest of the command line besides the args,
# e.g. 'runsc exec ...'. So we use 2**16 as the limit.
ARG_MAX_BYTES = 2**16
TTL_NO_EXPIRY_SENTINEL = -1


_SECRET_KEYNAME_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_sandbox_env(env: dict[str, str]) -> None:
    for key in env:
        if not key:
            raise InvalidError("Secret key name cannot be empty")
        if not _SECRET_KEYNAME_REGEX.match(key):
            raise InvalidError(
                f"Secret key name {key!r} is invalid for environment variables. "
                "Only letters, numbers, and underscores are allowed."
            )


def _validate_sandbox_runtime(runtime: SandboxRuntime | None) -> None:
    runtimes = typing.get_args(SandboxRuntime)
    if runtime is not None and runtime not in runtimes:
        raise InvalidError(f"runtime must be one of {list(runtimes)}, got {runtime!r}")


def _ttl_to_wire_ttl(ttl: int | None) -> int:
    """Convert a TTL value to the wire format, validating the input."""
    if ttl is None:
        return TTL_NO_EXPIRY_SENTINEL
    if ttl <= 0:
        raise InvalidError("ttl must be positive, or None to disable expiry")
    return ttl


_V1_SANDBOX_ID_ALPHABET = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MIN_LENGTH = 16
_CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MAX_LENGTH = 512


def _validate_experimental_encryption_key(key: bytes | None) -> bytes | None:
    if key is None:
        return None
    if not isinstance(key, bytes):
        raise TypeError("_experimental_encryption_key must be bytes")
    if len(key) == 0:
        raise InvalidError("_experimental_encryption_key must not be empty")
    if len(key) < _CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MIN_LENGTH:
        raise InvalidError(
            f"_experimental_encryption_key must be at least {_CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MIN_LENGTH} bytes"
        )
    if len(key) > _CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MAX_LENGTH:
        raise InvalidError(
            f"_experimental_encryption_key must be at most {_CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MAX_LENGTH} bytes"
        )
    return key


def _image_id_for_mount(image: _Image, method_name: str) -> str:
    if not isinstance(image, _Image):
        raise TypeError(f"{method_name}(image=...) expects an Image object, got {image!r}")

    if image._mount_layers:
        raise InvalidError(
            f"{method_name}() only supports pre-built images. When using `add_local*` methods, "
            "specify `copy=True` and call `.build()` before passing the image to `mount_image()`:\n\nE.g.\n"
            'img = modal.Image.debian_slim().add_local_file("foo", "/foo", copy=True).build(app)\n'
            f"{method_name}(path, img)"
        )
    if image._is_empty:
        return ""
    if image._object_id:
        return image._object_id
    raise InvalidError(
        f"{method_name}() currently only supports Images that are either:\n"
        "- prebuilt using `image.build()`\n"
        "- referenced by id, e.g. `Image.from_id()`\n"
        "- filesystem/directory snapshots e.g. created by `.snapshot_directory()` "
        "or `.snapshot_filesystem()`\n"
    )


if TYPE_CHECKING:
    import modal.app


class SandboxVersion(enum.Enum):
    V1 = 1
    V2 = 2


def _is_v1_sandbox_id(sandbox_id: str) -> bool:
    prefix, separator, suffix = sandbox_id.partition("-")
    return (
        prefix == "sb"
        and separator == "-"
        and len(suffix) == 22
        and all(ch in _V1_SANDBOX_ID_ALPHABET for ch in suffix)
    )


def _get_sandbox_version(sandbox_id: str) -> SandboxVersion:
    # IDs that don't match the V1 shape are assumed to be V2, so that Sandbox
    # IDs minted in newer formats route to the newer backend instead of
    # failing client-side.
    if _is_v1_sandbox_id(sandbox_id):
        return SandboxVersion.V1
    return SandboxVersion.V2


def _result_returncode(result: api_pb2.GenericResult | None) -> int | None:
    if result is None or result.status == api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED:
        return None
    if result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
        return 124
    if result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED:
        return 137
    return result.exitcode


def _validate_exec_args(args: Sequence[str]) -> None:
    # Entrypoint args must be strings.
    if not all(isinstance(arg, str) for arg in args):
        raise InvalidError("All entrypoint arguments must be strings")
    # Avoid "[Errno 7] Argument list too long" errors.
    total_arg_len = sum(len(arg) for arg in args)
    if total_arg_len > ARG_MAX_BYTES:
        raise InvalidError(
            f"Total length of CMD arguments cannot exceed {ARG_MAX_BYTES} bytes (ARG_MAX). Got {total_arg_len} bytes."
        )


def _build_outbound_network_access(
    block_network: bool,
    outbound_cidr_allowlist: Sequence[str] | None,
    outbound_domain_allowlist: Sequence[str] | None,
) -> api_pb2.NetworkAccess:
    """Build the outbound `NetworkAccess` for the given params.

    Shared by `Sandbox.create` and the sidecar create path. When nothing is
    specified, defaults to open network access.
    """
    if block_network:
        if outbound_cidr_allowlist is not None:
            raise InvalidError("`outbound_cidr_allowlist` cannot be used when `block_network` is enabled")
        if outbound_domain_allowlist is not None:
            raise InvalidError("`outbound_domain_allowlist` cannot be used when `block_network` is enabled")
        return api_pb2.NetworkAccess(network_access_type=api_pb2.NetworkAccess.NetworkAccessType.BLOCKED)
    if outbound_cidr_allowlist is None and outbound_domain_allowlist is None:
        return api_pb2.NetworkAccess(network_access_type=api_pb2.NetworkAccess.NetworkAccessType.OPEN)
    return api_pb2.NetworkAccess(
        network_access_type=api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST,
        allowed_cidrs=list(outbound_cidr_allowlist or []),
        allowed_domains=list(outbound_domain_allowlist or []),
    )


def _resolve_app_id_and_client(
    app: "modal.app._App | None",
    client: "_Client | None",
) -> "tuple[str | None, _Client | None]":
    """Resolve the App id and client for Sandbox creation, validating that an App is available."""
    from ..app import _App

    if app is not None:
        if app.app_id is None:
            raise ValueError(
                "App has not been initialized yet. To create an App lazily, use `App.lookup`: \n"
                "app = modal.App.lookup('my-app', create_if_missing=True)\n"
                "modal.Sandbox.create('echo', 'hi', app=app)\n"
                "In order to initialize an existing `App` object, refer to our docs: https://modal.com/docs/guide/apps"
            )
        app_id = app.app_id
        app_client = app._client
    elif (container_app := _App._get_container_app()) is not None:
        app_id = container_app.app_id
        app_client = container_app._client
    else:
        raise InvalidError(
            "Sandboxes require an App when created outside of a Modal container.\n\n"
            "Run an ephemeral App (`with app.run(): ...`), or reference a deployed App using `App.lookup`:\n\n"
            "```\n"
            'app = modal.App.lookup("sandbox-app", create_if_missing=True)\n'
            "sb = modal.Sandbox.create(..., app=app)\n"
            "```",
        )

    return app_id, client or app_client
