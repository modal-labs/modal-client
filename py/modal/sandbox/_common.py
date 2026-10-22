# Copyright Modal Labs 2026
"""Code shared by the Sandbox facade and its backend implementations, including the implementation interface."""

import asyncio
import contextlib
import dataclasses
import enum
import re
import time
import typing
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Protocol

from grpclib import Status

from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._image import _Image
from .._load_context import LoadContext
from .._object import _Object
from .._outbound_policy import _OutboundPolicy, _validate_compatible_network_access
from .._output.pty import get_pty_info
from .._resolver import Resolver
from .._resources import convert_fn_config_to_resources_config
from .._tunnel import Tunnel
from .._utils.async_utils import TaskContext
from .._utils.grpc_utils import Retry
from .._utils.mount_utils import validate_network_file_systems, validate_volumes, validate_volumes_by_object_id
from .._utils.name_utils import check_object_name
from ..client import _Client
from ..cloud_bucket_mount import _CloudBucketMount
from ..config import config, logger
from ..container_process import _ContainerProcess
from ..exception import (
    ConflictError,
    ExecutionError,
    InternalError,
    InvalidError,
    NotFoundError,
    SandboxTerminatedError,
    SandboxTimeoutError,
    ServiceError,
    SnapshotCreationError,
    TimeoutError,
)
from ..io_streams import StreamReader, StreamWriter
from ..mount import _Mount
from ..network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from ..proxy import _Proxy
from ..secret import _resolvable_secrets, _Secret, _secret_sources
from ..stream_type import StreamType
from ..types import SandboxRuntime
from ..volume import _Volume, _volume_to_mount_proto
from ._snapshot import _SandboxSnapshot
from ._task_command_router_client import TaskCommandRouterClient

if TYPE_CHECKING:
    import modal.app

    from ._sandbox import Probe


_default_image: _Image = _Image.debian_slim()


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


@contextlib.asynccontextmanager
async def sandbox_create_context(
    app: "modal.app._App | None", client: _Client | None
) -> AsyncIterator[tuple[_Client, LoadContext]]:
    """Resolve the App and client of a new Sandbox, and open the load context its dependencies load in."""
    app_id, client = _resolve_app_id_and_client(app, client)
    if client is None:
        client = await _Client.from_env()
    async with TaskContext() as tc:
        yield client, LoadContext(client=client, app_id=app_id, task_context=tc)


def default_pty_info() -> api_pb2.PTYInfo:
    return get_pty_info(shell=True, no_terminate_on_idle_stdin=True)


_EXIT_SNAPSHOT_NOT_FOUND_ERROR_CODES = frozenset(
    (
        api_pb2.SandboxGetExitSnapshotResponse.ERROR_CODE_TIMEOUT,
        api_pb2.SandboxGetExitSnapshotResponse.ERROR_CODE_FILESYSTEM_INCONSISTENT,
    )
)

# Exit snapshot polling holds a server-side long poll for at most _EXIT_SNAPSHOT_LONG_POLL_TIMEOUT seconds and
# then re-polls from the client, so a request is never held open indefinitely however long the caller waits.
_EXIT_SNAPSHOT_LONG_POLL_TIMEOUT = 10.0
# Margin added to each poll's network deadline over the server-side hold, covering the
# response's return trip so an answer sent just before the hold expires isn't lost to the
# client-side deadline.
_EXIT_SNAPSHOT_POLL_DEADLINE_MARGIN = 5.0

# How long to wait between task ID lookups.
_TASK_ID_POLL_INTERVAL: float = 0.5


def absolute_path_bytes(path: PurePosixPath | str, what: str) -> bytes:
    posix_path = PurePosixPath(path)
    if not posix_path.is_absolute():
        raise InvalidError(f"{what} path must be absolute; got: {posix_path}")
    return posix_path.as_posix().encode("utf8")


def split_volumes(
    volumes: Mapping[str | PurePosixPath, _Volume | _CloudBucketMount],
) -> tuple[list[tuple[str, _Volume]], list[tuple[str, _CloudBucketMount]]]:
    """Validate the volume mapping of a new Sandbox and split it into Volumes and cloud bucket mounts."""
    mounted_objects = validate_volumes(volumes)
    validated_volumes = [(k, v) for k, v in mounted_objects if isinstance(v, _Volume)]
    cloud_bucket_mounts = [(k, v) for k, v in mounted_objects if isinstance(v, _CloudBucketMount)]
    return validated_volumes, cloud_bucket_mounts


def scheduler_placement(region: str | Sequence[str] | None) -> api_pb2.SchedulerPlacement | None:
    if not region:
        return None
    regions = [region] if isinstance(region, str) else list(region)
    return api_pb2.SchedulerPlacement(regions=regions)


def open_port_specs(
    encrypted_ports: Sequence[int], unencrypted_ports: Sequence[int], h2_ports: Sequence[int]
) -> list[api_pb2.PortSpec]:
    open_ports = [api_pb2.PortSpec(port=port, unencrypted=False) for port in encrypted_ports]
    open_ports.extend([api_pb2.PortSpec(port=port, unencrypted=True) for port in unencrypted_ports])
    open_ports.extend(
        [api_pb2.PortSpec(port=port, unencrypted=False, tunnel_type=api_pb2.TUNNEL_TYPE_H2) for port in h2_ports]
    )
    return open_ports


def tunnels_from_protos(tunnels: Iterable[api_pb2.TunnelData]) -> dict[int, Tunnel]:
    return {t.container_port: Tunnel(t.host, t.port, t.unencrypted_host, t.unencrypted_port) for t in tunnels}


def tag_protos(tags: Mapping[str, str] | None) -> list[api_pb2.SandboxTag]:
    return [api_pb2.SandboxTag(tag_name=k, tag_value=v) for k, v in tags.items()] if tags else []


def tags_from_protos(tags: Iterable[api_pb2.SandboxTag]) -> dict[str, str]:
    return {tag.tag_name: tag.tag_value for tag in tags}


@dataclasses.dataclass
class SandboxCreateArgs:
    """The arguments of `Sandbox.create`, with its defaults, exactly as requested.

    Nothing is validated here; `SandboxCreateCommon` validates and derives what the backends share.
    """

    args: Sequence[str] = ()
    image: _Image | None = None
    secrets: Collection[_Secret] | None = None
    env: dict[str, str | None] | None = None
    name: str | None = None
    timeout: int | None = 300
    idle_timeout: int | None = None
    workdir: str | None = None
    gpu: str | None = None
    cloud: str | None = None
    region: str | Sequence[str] | None = None
    cpu: float | tuple[float, float] | None = None
    memory: int | tuple[int, int] | None = None
    mounts: Sequence[_Mount] = ()
    network_file_systems: Mapping[str | PurePosixPath, _NetworkFileSystem] = dataclasses.field(default_factory=dict)
    block_network: bool = False
    outbound_cidr_allowlist: Sequence[str] | None = None
    outbound_domain_allowlist: Sequence[str] | None = None
    outbound_policy: _OutboundPolicy | None = None
    inbound_cidr_allowlist: Sequence[str] | None = None
    i6pn: bool = False
    volumes: Mapping[str | PurePosixPath, _Volume | _CloudBucketMount] = dataclasses.field(default_factory=dict)
    pty: bool = False
    pty_info: api_pb2.PTYInfo | None = None
    encrypted_ports: Sequence[int] = ()
    h2_ports: Sequence[int] = ()
    unencrypted_ports: Sequence[int] = ()
    proxy: _Proxy | None = None
    readiness_probe: "Probe | None" = None
    experimental_options: dict[str, Any] | None = None
    tags: dict[str, str] | None = None
    enable_snapshot: bool = False
    verbose: bool = False
    custom_domain: str | None = None
    include_oidc_identity_token: bool = False
    runtime: SandboxRuntime | None = None
    persistent: bool = False


class SandboxCreateCommon:
    """Validation, dependency loading and definition building shared by every Sandbox backend.

    `secrets` are the Secrets the backend loads and references by id. Each backend decides which of
    the requested Secrets those are (and how it sends the rest) before constructing this.
    """

    def __init__(self, create_args: SandboxCreateArgs, *, secrets: Sequence[_Secret]) -> None:
        a = create_args
        if a.timeout is None:
            raise InvalidError("`timeout=None` is only supported for experimental persistent Sandboxes.")
        _validate_exec_args(a.args)
        if a.name is not None:
            check_object_name(a.name, "Sandbox")
        if a.workdir is not None and not a.workdir.startswith("/"):
            raise InvalidError(f"workdir must be an absolute path, got: {a.workdir}")
        _validate_sandbox_runtime(a.runtime)
        if isinstance(a.gpu, list):
            raise InvalidError(
                "Sandboxes do not support configuring a list of GPUs. "
                "Specify a single GPU configuration, e.g. gpu='a10g'"
            )

        if a.block_network and (a.encrypted_ports or a.h2_ports or a.unencrypted_ports):
            raise InvalidError("Cannot specify open ports when `block_network` is enabled")
        if a.outbound_policy is not None:
            a.outbound_policy._validate()
        _validate_compatible_network_access(a.outbound_policy, a.block_network, a.outbound_domain_allowlist)
        network_access = _build_outbound_network_access(
            a.block_network, a.outbound_cidr_allowlist, a.outbound_domain_allowlist
        )
        if a.block_network and a.inbound_cidr_allowlist is not None:
            raise InvalidError("`inbound_cidr_allowlist` cannot be used when `block_network` is enabled")

        self.create_args = a
        self.timeout = a.timeout
        self.secrets = secrets
        self.image = a.image or _default_image
        self.network_file_systems = validate_network_file_systems(a.network_file_systems)
        self.volumes, self.cloud_bucket_mounts = split_volumes(a.volumes)
        self.network_access = network_access
        self.open_ports = open_port_specs(a.encrypted_ports, a.unencrypted_ports, a.h2_ports)
        self.pty_info = default_pty_info() if a.pty else a.pty_info
        self.scheduler_placement = scheduler_placement(a.region)

    async def load_dependencies(
        self,
        resolver: Resolver,
        load_context: LoadContext,
        *,
        load_ephemeral_bucket_secrets: bool,
    ) -> list[tuple[str, float]]:
        """Load the objects the Sandbox depends on, and return how long each load took."""
        a = self.create_args
        deps: list[_Object] = []
        # An already-hydrated image (e.g. one returned by `Sandbox.snapshot_directory`) has nothing to load.
        if not self.image._is_hydrated:
            deps.append(self.image)
        deps.extend(a.mounts)
        deps.extend(self.secrets)
        if a.outbound_policy:
            deps.extend(a.outbound_policy._secrets())
        deps.extend(nfs for _, nfs in self.network_file_systems)
        deps.extend(volume for _, volume in self.volumes)
        for _, cloud_bucket_mount in self.cloud_bucket_mounts:
            secret = cloud_bucket_mount.secret
            if secret and (load_ephemeral_bucket_secrets or not secret._is_ephemeral):
                deps.append(secret)
        if a.proxy:
            deps.append(a.proxy)
        dep_timings = await _gather_load_with_timings([resolver.load(dep, load_context) for dep in deps])

        # Validate that the same volume (by object_id) isn't mounted at multiple paths
        validate_volumes_by_object_id(self.volumes)
        return dep_timings

    def definition(self, *, cloud_bucket_mounts: Sequence[api_pb2.CloudBucketMount]) -> api_pb2.Sandbox:
        """Build the definition of the Sandbox once its dependencies are loaded."""
        a = self.create_args
        image = self.image
        return api_pb2.Sandbox(
            entrypoint_args=a.args,
            image_id=image.object_id,
            mount_ids=[mount.object_id for mount in a.mounts] + [mount.object_id for mount in image._mount_layers],
            secret_ids=[secret.object_id for secret in self.secrets],
            # Ephemeral disk requests are not supported on Sandboxes.
            resources=convert_fn_config_to_resources_config(cpu=a.cpu, memory=a.memory, gpu=a.gpu, ephemeral_disk=None),
            timeout_secs=self.timeout,
            idle_timeout_secs=a.idle_timeout,
            workdir=a.workdir,
            cloud_provider_str=a.cloud if a.cloud else None,  # pyright: ignore[reportArgumentType]
            runtime=a.runtime or config.get("function_runtime"),
            runtime_debug=config.get("function_runtime_debug"),
            pty_info=self.pty_info,
            scheduler_placement=self.scheduler_placement,
            worker_id=config.get("worker_id"),
            open_ports=api_pb2.PortSpecs(ports=self.open_ports),
            network_access=self.network_access,
            experimental_outbound_policy=a.outbound_policy._to_proto() if a.outbound_policy else None,
            proxy_id=(a.proxy.object_id if a.proxy else None),
            readiness_probe=(a.readiness_probe._to_proto() if a.readiness_probe else None),
            enable_snapshot=a.enable_snapshot,
            verbose=a.verbose,
            name=a.name,
            experimental_options_v2=(
                {k: str(v) for k, v in a.experimental_options.items()} if a.experimental_options else None
            ),
            custom_domain=a.custom_domain,  # pyright: ignore[reportArgumentType]
            include_oidc_identity_token=a.include_oidc_identity_token,
            inbound_cidr_allowlist=(list(a.inbound_cidr_allowlist) if a.inbound_cidr_allowlist is not None else []),
            # Relies on dicts being ordered (true as of Python 3.6).
            volume_mounts=[_volume_to_mount_proto(path, volume) for path, volume in self.volumes],
            nfs_mounts=network_file_system_mount_protos(self.network_file_systems),
            cloud_bucket_mounts=cloud_bucket_mounts,
            i6pn_enabled=a.i6pn,
        )


def task_id_poll_deadline(timeout: float | None) -> float | None:
    return time.monotonic() + timeout if timeout is not None else None


def raise_unless_retryable_task_id_error(exc: Exception, *, retry_transient: bool, deadline: float | None) -> None:
    """Re-raise an error from polling for a task ID, unless polling should continue after it."""
    # Keep polling through transient server or network errors until the deadline.
    transient = isinstance(exc, (InternalError, ConnectionError)) or (
        isinstance(exc, ServiceError) and exc._grpc_status in (Status.UNAVAILABLE, Status.DEADLINE_EXCEEDED)
    )
    if not retry_transient or deadline is None or not transient:
        raise exc
    if deadline - time.monotonic() <= _TASK_ID_POLL_INTERVAL:
        # A server error is not evidence of missing capacity, so surface it as is.
        if isinstance(exc, InternalError):
            raise exc
        raise TimeoutError("Sandbox was not scheduled within the timeout.") from exc


def task_id_for_logs(resp: api_pb2.SandboxGetTaskIdResponse) -> str:
    if not resp.task_id:
        raise ExecutionError("Sandbox task_id is not available yet.")
    return resp.task_id


class ExitSnapshotPoll:
    """Builds the requests and handles the responses of one wait for the exit snapshot of a Sandbox."""

    def __init__(self, sandbox_id: str, timeout: float | None) -> None:
        self._sandbox_id = sandbox_id
        self._timeout = timeout
        self._deadline = None if timeout is None else time.monotonic() + timeout

    def next_request(self) -> tuple[api_pb2.SandboxGetExitSnapshotRequest, Retry]:
        if self._deadline is None:
            remaining = None
            request_timeout = _EXIT_SNAPSHOT_LONG_POLL_TIMEOUT
        else:
            remaining = max(0.0, self._deadline - time.monotonic())
            request_timeout = min(_EXIT_SNAPSHOT_LONG_POLL_TIMEOUT, remaining)
        req = api_pb2.SandboxGetExitSnapshotRequest(sandbox_id=self._sandbox_id, timeout=request_timeout)
        retry = Retry(
            attempt_timeout=request_timeout + _EXIT_SNAPSHOT_POLL_DEADLINE_MARGIN,
            attempt_timeout_floor=_EXIT_SNAPSHOT_POLL_DEADLINE_MARGIN,
            total_timeout=remaining,
        )
        return req, retry

    def timeout_error(self) -> TimeoutError:
        return TimeoutError(
            f"timed out waiting for exit snapshot for Sandbox {self._sandbox_id}: "
            f"the operation could not complete within the provided timeout of {self._timeout} seconds"
        )

    def handle_response(self, resp: api_pb2.SandboxGetExitSnapshotResponse, client: _Client) -> _Image | None:
        """Return the exit snapshot image, or None if it is still pending."""
        outcome = resp.WhichOneof("outcome")

        if outcome == "success":
            if not resp.success.image_id:
                raise InternalError("Exit snapshot result is missing image ID")
            return _Image._new_hydrated(resp.success.image_id, client, None)

        if outcome == "error":
            if resp.error.error_code in _EXIT_SNAPSHOT_NOT_FOUND_ERROR_CODES:
                message = resp.error.message or "No exit snapshot image will be produced"
                raise SnapshotCreationError(message)
            message = resp.error.message or "Exit snapshot state is malformed"
            raise InternalError(message)

        if outcome != "pending":
            raise InternalError("Exit snapshot response is missing an outcome")

        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise self.timeout_error()
        return None


class SandboxImpl(Protocol):
    """Implements the operations of a `_Sandbox` for one type of Sandbox backend.

    Every `_Sandbox` delegates its operations to an implementation, which is selected
    from the Sandbox type when the handle is created or looked up.
    """

    client: _Client
    sandbox_id: str
    # The Sandbox's final result once known; `None` while it is running.
    result: api_pb2.GenericResult | None

    def __init__(self, client: _Client, sandbox_id: str, *, result: api_pb2.GenericResult | None = None) -> None: ...

    @classmethod
    async def create(
        cls, create_args: SandboxCreateArgs, *, app: "modal.app._App | None", client: _Client | None
    ) -> "tuple[SandboxImpl, api_pb2.SandboxHandleMetadata]":
        """Create the Sandbox described by `create_args`.

        Returns the implementation backing the new Sandbox, holding its initial state, and the handle
        metadata the backend returned; `_Sandbox._from_impl` turns them into a handle.
        """
        ...

    def clone(self) -> "SandboxImpl":
        """Create an implementation of the same type for another handle to the same Sandbox."""
        ...

    @classmethod
    async def lookup_by_name(
        cls, client: _Client, req: api_pb2.SandboxGetFromNameRequest
    ) -> api_pb2.SandboxGetFromNameResponse: ...

    @classmethod
    async def lookup_by_id(cls, client: _Client, sandbox_id: str) -> api_pb2.SandboxWaitResponse: ...

    @classmethod
    async def list_page(cls, client: _Client, req: api_pb2.SandboxListRequest) -> api_pb2.SandboxListResponse: ...

    @classmethod
    async def restore_from_memory_snapshot(
        cls,
        snapshot: _SandboxSnapshot,
        client: _Client,
        *,
        override_type: "api_pb2.SandboxRestoreRequest.SandboxNameOverrideType.ValueType",
        name_override: str | None,
    ) -> "tuple[SandboxImpl, api_pb2.SandboxHandleMetadata]": ...

    def make_stdio(self, check_open: Callable[[], None]) -> tuple[StreamReader[str], StreamReader[str], StreamWriter]:
        """Create the stdout, stderr and stdin streams of the Sandbox entrypoint."""
        ...

    async def get_tags(self) -> dict[str, str]: ...

    async def set_tags(self, tags: dict[str, str]) -> None: ...

    async def set_name(self, name: str) -> None: ...

    async def create_connect_token(
        self, req: api_pb2.SandboxCreateConnectTokenRequest
    ) -> api_pb2.SandboxCreateConnectTokenResponse: ...

    async def terminate(self) -> None: ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def close(self) -> None:
        """Release connections held for the Sandbox once in-flight operations complete."""
        ...

    async def wait(self, raise_on_termination: bool) -> None:
        # Use the unchecked client to allow `wait` to work with a detached sandbox
        ...

    async def poll(self) -> None: ...

    async def tunnels(self, timeout: int) -> dict[int, Tunnel]: ...

    async def get_task_id(
        self, raise_if_task_complete: bool = False, timeout: float | None = None, retry_transient: bool = False
    ) -> str: ...

    async def resolve_task_id_for_logs(self) -> str:
        """Try to resolve the task ID in one shot for historical log fetches.

        For a historical log api, we should not wait for the task to be created.
        If the Sandbox created logs worth fetching, the task ID should be available.
        """
        ...

    async def get_exit_snapshot(self, timeout: float | None) -> _Image: ...

    async def snapshot_memory(self) -> _SandboxSnapshot: ...

    async def get_command_router_client(self, task_id: str) -> TaskCommandRouterClient: ...

    async def wait_until_ready(self, timeout: int) -> None: ...

    async def set_network_access(self, network_access: api_pb2.NetworkAccess) -> None: ...

    async def update_outbound_policy(self, outbound_policy: _OutboundPolicy) -> None: ...

    async def snapshot_filesystem(self, timeout: int, *, ttl: int | None, container_id: str = "") -> _Image: ...

    async def mount_image(
        self,
        path: PurePosixPath | str,
        image_id: str,
        *,
        encryption_key: bytes | None,
        container_id: str = "",
    ) -> None: ...

    async def unmount_image(self, path: PurePosixPath | str, *, container_id: str = "") -> None: ...

    async def snapshot_directory(
        self,
        path: PurePosixPath | str,
        *,
        timeout: int,
        ttl: int | None,
        encryption_key: bytes | None,
        container_id: str = "",
    ) -> _Image: ...

    async def reload_volumes(self, *, timeout: int, container_id: str = "") -> None:
        # Reload one container's Volumes; empty `container_id` targets the main container.
        ...

    async def exec(
        self,
        *args: str,
        pty_info: api_pb2.PTYInfo | None,
        stdout: StreamType,
        stderr: StreamType,
        timeout: int | None,
        workdir: str | None,
        env: dict[str, str | None] | None,
        secrets: Collection[_Secret] | None,
        text: bool,
        bufsize: Literal[-1, 1],
        container_id: str | None,
    ) -> _ContainerProcess[bytes] | _ContainerProcess[str]: ...

    def validate_sidecar_create(
        self,
        *,
        cloud_bucket_mounts: Sequence[tuple[str, _CloudBucketMount]],
        has_proxy: bool,
        include_oidc_identity_token: bool,
    ) -> None:
        """Reject sidecar options that `create_sidecar` cannot provide for this Sandbox."""
        ...

    async def create_sidecar(
        self,
        *args: str,
        name: str,
        image_id: str,
        secret_ids: Sequence[str],
        secret_sources: Sequence[api_pb2.SecretSource],
        workdir: str | None,
        volume_mounts: Sequence[api_pb2.VolumeMount],
        cloud_bucket_mounts: Sequence[tuple[str, _CloudBucketMount]],
        network_access: api_pb2.NetworkAccess,
        include_oidc_identity_token: bool,
        proxy_id: str | None,
        pty_info: api_pb2.PTYInfo | None,
        memory_reserve_consume_mib: int | None,
    ) -> tuple[str, str]:
        """Create a sidecar container and return its ID and name.

        Options rejected by `validate_sidecar_create` are ignored here.
        """
        ...

    async def get_sidecar(self, name: str, include_terminated: bool) -> sr_pb2.TaskContainerInfo: ...

    async def list_sidecars(self, include_terminated: bool) -> list[sr_pb2.TaskContainerInfo]: ...

    async def wait_sidecar(self, container_id: str, timeout: float) -> api_pb2.GenericResult | None:
        """Wait up to `timeout` seconds for a sidecar to exit, returning its result if it has."""
        ...

    async def terminate_sidecar(self, container_id: str) -> None: ...


# Stateless helpers for code reuse between `SandboxImplV1` and `SandboxImplV2`. They don't mirror the
# `SandboxImpl` protocol; each implementation decides which helpers to call. The `*_via_tcr` helpers
# implement operations served by the Sandbox's task command router on top of the implementation's
# `get_task_id` and `get_command_router_client`.


# `Sandbox.wait` re-polls a server-side long poll of this many seconds. Each poll gets a network deadline
# with a margin over the hold, so a poll the server never answers fails and is retried instead of hanging.
_WAIT_LONG_POLL_TIMEOUT = 10.0
_WAIT_POLL_DEADLINE_MARGIN = 5.0


def wait_poll_request(sandbox_id: str) -> tuple[api_pb2.SandboxWaitRequest, Retry]:
    """Build one `Sandbox.wait` long poll and the retry policy that bounds it."""
    req = api_pb2.SandboxWaitRequest(sandbox_id=sandbox_id, timeout=_WAIT_LONG_POLL_TIMEOUT)
    return req, Retry(attempt_timeout=_WAIT_LONG_POLL_TIMEOUT + _WAIT_POLL_DEADLINE_MARGIN)


def raise_for_wait_result(sandbox_id: str, result: api_pb2.GenericResult, raise_on_termination: bool) -> None:
    logger.debug(f"Sandbox {sandbox_id} wait completed with status {result.status}")
    if result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
        raise SandboxTimeoutError()
    elif result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED and raise_on_termination:
        raise SandboxTerminatedError()


def tunnels_from_response(resp: api_pb2.SandboxGetTunnelsResponse) -> dict[int, Tunnel]:
    # If we couldn't get the tunnels in time, report the timeout.
    if resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
        raise SandboxTimeoutError()
    return tunnels_from_protos(resp.tunnels)


def task_id_from_response(
    resp: api_pb2.SandboxGetTaskIdResponse, *, raise_if_task_complete: bool, deadline: float | None
) -> str:
    """Return the task ID of the Sandbox, or an empty string if it is not scheduled yet."""
    if not resp.task_id and raise_if_task_complete and resp.HasField("task_result"):
        msg = resp.task_result.exception or "Sandbox already finished"
        raise ConflictError(msg)
    if not resp.task_id and deadline is not None and deadline - time.monotonic() <= _TASK_ID_POLL_INTERVAL:
        raise TimeoutError("Sandbox was not scheduled within the timeout.")
    return resp.task_id


def new_snapshot_handle(
    client: _Client, snapshot_id: str, metadata: api_pb2.SandboxSnapshotHandleMetadata
) -> _SandboxSnapshot:
    """Return a hydrated handle for a memory snapshot of the Sandbox."""

    async def _load(
        self: _SandboxSnapshot, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None
    ):
        # we eagerly hydrate the sandbox snapshot below
        pass

    rep = "SandboxSnapshot()"
    # TODO: use ._new_hydrated instead
    obj = _SandboxSnapshot._from_loader(_load, rep, hydrate_lazily=True, load_context_overrides=LoadContext.empty())
    obj._hydrate(snapshot_id, client, metadata)
    return obj


async def command_router(impl: SandboxImpl) -> tuple[str, TaskCommandRouterClient]:
    task_id = await impl.get_task_id()
    return task_id, await impl.get_command_router_client(task_id)


async def wait_until_ready_via_tcr(impl: SandboxImpl, timeout: int) -> None:
    task_id = await impl.get_task_id(raise_if_task_complete=True)
    try:
        command_router_client = await impl.get_command_router_client(task_id)
    except NotFoundError as e:
        # We do this to maintain backwards compatibility within wait_until_ready.
        # The V1 implementation would raise ConflictError instead of NotFoundError
        # if the sandbox was terminated, so we do the same for V2.
        raise ConflictError(str(e)) from e
    await command_router_client.sandbox_wait_until_ready(task_id, timeout=timeout)


async def set_network_access_via_tcr(impl: SandboxImpl, network_access: api_pb2.NetworkAccess) -> None:
    task_id, command_router_client = await command_router(impl)
    req = sr_pb2.TaskSetNetworkAccessRequest(task_id=task_id, network_access=network_access)
    await command_router_client.set_network_access(req)


async def update_outbound_policy_via_tcr(impl: SandboxImpl, client: _Client, outbound_policy: _OutboundPolicy) -> None:
    async def hydrate_policy_secrets() -> None:
        await TaskContext.gather(*(secret.hydrate(client=client) for secret in outbound_policy._secrets()))

    _, (task_id, command_router_client) = await TaskContext.gather(hydrate_policy_secrets(), command_router(impl))
    req = sr_pb2.TaskSetExperimentalOutboundPolicyRequest(
        task_id=task_id, experimental_outbound_policy=outbound_policy._to_proto()
    )
    await command_router_client.set_outbound_policy(req)


async def snapshot_filesystem_via_tcr(
    impl: SandboxImpl, client: _Client, timeout: int, *, ttl: int | None, container_id: str = ""
) -> _Image:
    wire_ttl_seconds = _ttl_to_wire_ttl(ttl)
    task_id, command_router_client = await command_router(impl)
    req = sr_pb2.TaskSnapshotFilesystemRequest(
        task_id=task_id,
        snapshot_id=str(uuid.uuid4()),
        ttl_seconds=wire_ttl_seconds,
        container_id=container_id,
    )
    res = await command_router_client.snapshot_filesystem(req, timeout=float(timeout))
    return _Image._new_hydrated(res.image_id, client, None)


_PERSISTENT_ONLY_MESSAGE = "`{}` requires a persistent Sandbox, created with `_experimental_persistent=True`."


async def mount_image_via_tcr(
    impl: SandboxImpl,
    path: PurePosixPath | str,
    image_id: str,
    *,
    encryption_key: bytes | None,
    container_id: str = "",
) -> None:
    path_bytes = absolute_path_bytes(path, "Mount")
    task_id, command_router_client = await command_router(impl)
    req = sr_pb2.TaskMountDirectoryRequest(
        task_id=task_id,
        path=path_bytes,
        image_id=image_id,
        customer_supplied_encryption_key=_validate_experimental_encryption_key(encryption_key),
        container_id=container_id,
    )
    await command_router_client.mount_image(req)


async def unmount_image_via_tcr(impl: SandboxImpl, path: PurePosixPath | str, *, container_id: str = "") -> None:
    path_bytes = absolute_path_bytes(path, "Unmount")
    task_id, command_router_client = await command_router(impl)
    req = sr_pb2.TaskUnmountDirectoryRequest(task_id=task_id, path=path_bytes, container_id=container_id)
    await command_router_client.unmount_image(req)


async def snapshot_directory_via_tcr(
    impl: SandboxImpl,
    client: _Client,
    path: PurePosixPath | str,
    *,
    timeout: int,
    ttl: int | None,
    encryption_key: bytes | None,
    container_id: str = "",
) -> _Image:
    wire_ttl_seconds = _ttl_to_wire_ttl(ttl)
    path_bytes = absolute_path_bytes(path, "Snapshot")
    task_id, command_router_client = await command_router(impl)
    req = sr_pb2.TaskSnapshotDirectoryRequest(
        task_id=task_id,
        path=path_bytes,
        snapshot_id=str(uuid.uuid4()),
        ttl_seconds=wire_ttl_seconds,
        customer_supplied_encryption_key=_validate_experimental_encryption_key(encryption_key),
        container_id=container_id,
    )
    res = await command_router_client.snapshot_directory(req, timeout=float(timeout))
    return _Image._new_hydrated(res.image_id, client, None)


async def reload_volumes_via_tcr(impl: SandboxImpl, *, timeout: int, container_id: str = "") -> None:
    # Reload one container's Volumes; empty `container_id` targets the main container.
    if timeout <= 0:
        raise InvalidError("The `timeout` argument to `reload_volumes` must be positive.")
    task_id, command_router_client = await command_router(impl)
    await command_router_client.reload_volumes(task_id, timeout=float(timeout), container_id=container_id)


async def exec_via_tcr(
    impl: SandboxImpl,
    client: _Client,
    *args: str,
    pty_info: api_pb2.PTYInfo | None,
    stdout: StreamType,
    stderr: StreamType,
    timeout: int | None,
    workdir: str | None,
    env: dict[str, str | None] | None,
    secrets: Collection[_Secret] | None,
    text: bool,
    bufsize: Literal[-1, 1],
    container_id: str | None,
) -> _ContainerProcess[bytes] | _ContainerProcess[str]:
    if workdir is not None and not workdir.startswith("/"):
        raise InvalidError(f"workdir must be an absolute path, got: {workdir}")
    _validate_exec_args(args)

    secrets = list(secrets or [])
    resolvable_secrets = _resolvable_secrets(secrets)

    # Force explicit secret resolution so we can pass the secret IDs to the backend.
    secret_coros = [secret.hydrate(client=client) for secret in resolvable_secrets]
    await TaskContext.gather(*secret_coros)

    env_dict = {k: v for k, v in (env or {}).items() if v is not None}

    task_id = await impl.get_task_id(raise_if_task_complete=True)

    # NB: This must come after the task ID is set, since the sandbox must be
    # scheduled before we can create a router client.
    command_router_client = await impl.get_command_router_client(task_id)

    # Generate a random process ID to use as a combination of idempotency key/process identifier.
    process_id = str(uuid.uuid4())
    if stdout == StreamType.PIPE:
        stdout_config = sr_pb2.TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_PIPE
    elif stdout == StreamType.DEVNULL:
        stdout_config = sr_pb2.TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_DEVNULL
    elif stdout == StreamType.STDOUT:
        # Stream stdout to the client so that it can be printed locally in the reader.
        stdout_config = sr_pb2.TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_PIPE
    else:
        raise ValueError("Unsupported StreamType for stdout")

    if stderr == StreamType.PIPE:
        stderr_config = sr_pb2.TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_PIPE
    elif stderr == StreamType.DEVNULL:
        stderr_config = sr_pb2.TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_DEVNULL
    elif stderr == StreamType.STDOUT:
        # Stream stderr to the client so that it can be printed locally in the reader.
        stderr_config = sr_pb2.TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_PIPE
    else:
        raise ValueError("Unsupported StreamType for stderr")

    # Start the process.
    start_req = sr_pb2.TaskExecStartRequest(
        task_id=task_id,
        exec_id=process_id,
        command_args=args,
        stdout_config=stdout_config,
        stderr_config=stderr_config,
        timeout_secs=timeout,
        workdir=workdir,
        secret_sources=_secret_sources(secrets, env_dict),
        pty_info=pty_info,
        runtime_debug=config.get("function_runtime_debug"),
        container_id=container_id or "",
    )
    _ = await command_router_client.exec_start(start_req)

    return _ContainerProcess(
        process_id,
        task_id,
        client,
        command_router_client=command_router_client,
        stdout=stdout,
        stderr=stderr,
        text=text,
        by_line=bufsize == 1,
        exec_deadline=time.monotonic() + int(timeout) if timeout else None,
    )


async def create_sidecar_via_tcr(
    impl: SandboxImpl,
    *args: str,
    name: str,
    image_id: str,
    secret_sources: Sequence[api_pb2.SecretSource],
    workdir: str | None,
    volume_mounts: Sequence[api_pb2.VolumeMount],
    network_access: api_pb2.NetworkAccess,
    pty_info: api_pb2.PTYInfo | None,
    memory_reserve_consume_mib: int | None,
) -> tuple[str, str]:
    """Create a sidecar container through the task command router and return its ID and name.

    Cloud bucket mounts, proxies and OIDC identity tokens are not supported on this path;
    `validate_sidecar_create` rejects them before creation.
    """
    task_id, command_router_client = await command_router(impl)
    create_resp = await command_router_client.container_create(
        sr_pb2.TaskContainerCreateRequest(
            task_id=task_id,
            container_name=name,
            image_id=image_id,
            args=list(args),
            workdir=workdir or "",
            secret_sources=secret_sources,
            volume_mounts=volume_mounts,
            network_access=network_access,
            pty_info=pty_info,
            memory_reserve_consume_mib=memory_reserve_consume_mib,
        )
    )
    return create_resp.container_id, create_resp.container_name or name


async def get_sidecar_via_tcr(impl: SandboxImpl, name: str, include_terminated: bool) -> sr_pb2.TaskContainerInfo:
    task_id, command_router_client = await command_router(impl)
    resp = await command_router_client.container_get(
        sr_pb2.TaskContainerGetRequest(task_id=task_id, container_name=name, include_terminated=include_terminated)
    )
    return resp.container


async def list_sidecars_via_tcr(impl: SandboxImpl, include_terminated: bool) -> list[sr_pb2.TaskContainerInfo]:
    task_id, command_router_client = await command_router(impl)
    resp = await command_router_client.container_list(
        sr_pb2.TaskContainerListRequest(task_id=task_id, include_terminated=include_terminated)
    )
    return list(resp.containers)


async def wait_sidecar_via_tcr(impl: SandboxImpl, container_id: str, timeout: float) -> api_pb2.GenericResult | None:
    """Wait up to `timeout` seconds for a sidecar to exit, returning its result if it has."""
    task_id, command_router_client = await command_router(impl)
    resp = await command_router_client.container_wait(
        sr_pb2.TaskContainerWaitRequest(task_id=task_id, container_id=container_id, timeout=timeout)
    )
    return resp.result if resp.result.status else None


async def terminate_sidecar_via_tcr(impl: SandboxImpl, container_id: str) -> None:
    task_id, command_router_client = await command_router(impl)
    await command_router_client.container_terminate(
        sr_pb2.TaskContainerTerminateRequest(task_id=task_id, container_id=container_id)
    )
