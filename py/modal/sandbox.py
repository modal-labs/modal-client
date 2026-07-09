# Copyright Modal Labs 2022
import asyncio
import builtins
import enum
import json
import logging
import os
import re
import time
import typing
import uuid
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Collection, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Union, overload

from modal.secret import _split_env_dict_and_resolvable_secrets

from ._output.pty import get_pty_info
from .config import config, logger

if TYPE_CHECKING:
    import _typeshed

from google.protobuf.message import Message

from modal._tunnel import Tunnel
from modal.cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from modal.mount import _Mount
from modal.volume import _Volume, _volume_to_mount_proto
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from ._image import _Image
from ._load_context import LoadContext
from ._object import _get_environment_name, _Object
from ._resolver import Resolver
from ._resources import convert_fn_config_to_resources_config
from ._utils.async_utils import TaskContext, synchronize_api
from ._utils.deprecation import deprecation_warning
from ._utils.mount_utils import (
    validate_network_file_systems,
    validate_only_modal_volumes,
    validate_volumes,
    validate_volumes_by_object_id,
)
from ._utils.name_utils import check_object_name
from ._utils.task_command_router_client import TaskCommandRouterClient
from .client import _Client
from .container_process import _ContainerProcess
from .exception import (
    ClientClosed,
    ConflictError,
    ExecutionError,
    InvalidError,
    NotFoundError,
    SandboxTerminatedError,
    SandboxTimeoutError,
)
from .file_io import _FileIO, ls, mkdir, rm, watch
from .io_streams import (
    StreamReader,
    StreamWriter,
    _StreamReader,
    _StreamReaderThroughSandboxCommandRouterParams,
    _StreamReaderThroughServerParams,
    _StreamWriter,
    _StreamWriterThroughCommandRouterSandboxParams,
    _StreamWriterThroughServerParams,
)
from .network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from .proxy import _Proxy
from .sandbox_fs import _SandboxFilesystem
from .secret import _Secret
from .snapshot import _SandboxSnapshot
from .stream_type import StreamType
from .types import FileWatchEvent, FileWatchEventType, SandboxConnectCredentials

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


def _ttl_to_wire_ttl(ttl: int | None) -> int:
    """Convert a TTL value to the wire format, validating the input."""
    if ttl is None:
        return TTL_NO_EXPIRY_SENTINEL
    if ttl <= 0:
        raise InvalidError("ttl must be positive, or None to disable expiry")
    return ttl


_V1_SANDBOX_ID_ALPHABET = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_ULID_ALPHABET = frozenset("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
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


def _is_v2_sandbox_id(sandbox_id: str) -> bool:
    prefix, separator, suffix = sandbox_id.partition("-")
    return (
        prefix == "sb"
        and separator == "-"
        and len(suffix) == 26
        and suffix[0] in "01234567"
        and all(ch in _ULID_ALPHABET for ch in suffix)
    )


def _get_sandbox_version(sandbox_id: str) -> SandboxVersion:
    if _is_v2_sandbox_id(sandbox_id):
        return SandboxVersion.V2
    if _is_v1_sandbox_id(sandbox_id):
        return SandboxVersion.V1
    raise InvalidError(f"Invalid Sandbox ID: {sandbox_id!r}")


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


class DefaultSandboxNameOverride(str):
    """A singleton class that represents the default sandbox name override.

    It is used to indicate that the sandbox name should not be overridden.
    """

    def __repr__(self) -> str:
        # NOTE: this must match the instance var name below in order for type stubs to work 😬
        return "_DEFAULT_SANDBOX_NAME_OVERRIDE"


_DEFAULT_SANDBOX_NAME_OVERRIDE = DefaultSandboxNameOverride()


@dataclass(frozen=True)
class Probe:
    """Probe configuration for the Sandbox Readiness Probe.

    Examples:
        ```python notest
        # Wait until a file exists.
        readiness_probe = modal.Probe.with_exec(
            "sh", "-c", "test -f /tmp/ready",
        )

        # Wait until a TCP port is accepting connections.
        readiness_probe = modal.Probe.with_tcp(8080)

        app = modal.App.lookup('sandbox-readiness-probe', create_if_missing=True)
        sandbox = modal.Sandbox.create(
            "python3", "-m", "http.server", "8080",
            readiness_probe=readiness_probe,
            app=app,
        )
        sandbox.wait_until_ready()
        ```
    """

    tcp_port: int | None = None
    exec_argv: tuple[str, ...] | None = None
    interval_ms: int = 100

    def __post_init__(self):
        if (self.tcp_port is None) == (self.exec_argv is None):
            raise InvalidError("Probe must be created with Probe.with_tcp(...) or Probe.with_exec(...)")

    @classmethod
    def with_tcp(cls, port: int, *, interval_ms: int = 100) -> "Probe":
        if not isinstance(port, int):
            raise InvalidError("Probe.with_tcp() expects an integer `port`")
        if port <= 0 or port > 65535:
            raise InvalidError(f"Probe.with_tcp() expects `port` in [1, 65535], got {port}")
        if not isinstance(interval_ms, int):
            raise InvalidError("Probe.with_tcp() expects an integer `interval_ms`")
        if interval_ms <= 0:
            raise InvalidError(f"Probe.with_tcp() expects `interval_ms` > 0, got {interval_ms}")
        return cls(tcp_port=port, interval_ms=interval_ms)

    @classmethod
    def with_exec(cls, *argv: str, interval_ms: int = 100) -> "Probe":
        if len(argv) == 0:
            raise InvalidError("Probe.with_exec() requires at least one argument")
        if not all(isinstance(arg, str) for arg in argv):
            raise InvalidError("Probe.with_exec() expects all arguments to be strings")
        if not isinstance(interval_ms, int):
            raise InvalidError("Probe.with_exec() expects an integer `interval_ms`")
        if interval_ms <= 0:
            raise InvalidError(f"Probe.with_exec() expects `interval_ms` > 0, got {interval_ms}")
        return cls(exec_argv=tuple(argv), interval_ms=interval_ms)

    def _to_proto(self) -> api_pb2.Probe:
        if self.tcp_port is not None:
            return api_pb2.Probe(tcp_port=self.tcp_port, interval_ms=self.interval_ms)
        if self.exec_argv is not None:
            return api_pb2.Probe(
                exec_command=api_pb2.Probe.ExecCommand(argv=list(self.exec_argv)),
                interval_ms=self.interval_ms,
            )
        raise InvalidError("Probe must be created with Probe.with_tcp(...) or Probe.with_exec(...)")


class _Sandbox(_Object, type_prefix="sb"):
    """A `Sandbox` object lets you interact with a running sandbox. This API is similar to Python's
    [asyncio.subprocess.Process](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.subprocess.Process).

    Refer to the [guide](https://modal.com/docs/guide/sandbox) on how to spawn and use sandboxes.
    """

    _result: api_pb2.GenericResult | None
    _stdout: _StreamReader[str]
    _stderr: _StreamReader[str]
    _stdin: _StreamWriter
    _task_id: str | None
    _tunnels: dict[int, Tunnel] | None
    _enable_snapshot: bool
    _command_router_client: TaskCommandRouterClient | None
    _attached: bool
    _filesystem: _SandboxFilesystem | None
    _is_v2: bool = False

    @staticmethod
    def _default_pty_info() -> api_pb2.PTYInfo:
        return get_pty_info(shell=True, no_terminate_on_idle_stdin=True)

    @staticmethod
    def _new(
        args: Sequence[str],
        image: _Image,
        secrets: Collection[_Secret],
        name: str | None = None,
        timeout: int = 300,
        idle_timeout: int | None = None,
        workdir: str | None = None,
        gpu: str | None = None,
        cloud: str | None = None,
        region: str | Sequence[str] | None = None,
        cpu: float | None = None,
        memory: int | tuple[int, int] | None = None,
        mounts: Sequence[_Mount] = (),
        network_file_systems: dict[str | os.PathLike, _NetworkFileSystem] = {},
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        inbound_cidr_allowlist: Sequence[str] | None = None,
        volumes: dict[str | os.PathLike, _Volume | _CloudBucketMount] = {},
        pty: bool = False,
        pty_info: api_pb2.PTYInfo | None = None,  # deprecated
        encrypted_ports: Sequence[int] = [],
        h2_ports: Sequence[int] = [],
        unencrypted_ports: Sequence[int] = [],
        proxy: _Proxy | None = None,
        readiness_probe: Probe | None = None,
        experimental_options: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        enable_snapshot: bool = False,
        verbose: bool = False,
        custom_domain: str | None = None,
        include_oidc_identity_token: bool = False,
    ) -> "_Sandbox":
        """mdmd:hidden"""

        validated_network_file_systems = validate_network_file_systems(network_file_systems)

        if isinstance(gpu, list):
            raise InvalidError(
                "Sandboxes do not support configuring a list of GPUs. "
                "Specify a single GPU configuration, e.g. gpu='a10g'"
            )

        if workdir is not None and not workdir.startswith("/"):
            raise InvalidError(f"workdir must be an absolute path, got: {workdir}")

        # Validate volumes
        validated_volumes = validate_volumes(volumes)
        cloud_bucket_mounts = [(k, v) for k, v in validated_volumes if isinstance(v, _CloudBucketMount)]
        validated_volumes = [(k, v) for k, v in validated_volumes if isinstance(v, _Volume)]

        scheduler_placement: api_pb2.SchedulerPlacement | None = None
        if region:
            regions = [region] if isinstance(region, str) else (list(region) if region else None)
            scheduler_placement = api_pb2.SchedulerPlacement(regions=regions)

        if pty:
            pty_info = _Sandbox._default_pty_info()

        async def _load(self: _Sandbox, resolver: Resolver, load_context: LoadContext, _existing_object_id: str | None):
            load_start = time.monotonic()
            # An already-hydrated image (e.g. one returned by
            # `Sandbox.snapshot_directory`) is skipped — there's nothing to load.
            dep_tasks: list = []
            if not image._is_hydrated:
                dep_tasks.append(resolver.load(image, load_context))
            for dep in list(mounts) + list(secrets):
                dep_tasks.append(resolver.load(dep, load_context))
            for _, vol in validated_network_file_systems:
                dep_tasks.append(resolver.load(vol, load_context))
            for _, vol in validated_volumes:
                dep_tasks.append(resolver.load(vol, load_context))
            for _, cloud_bucket_mount in cloud_bucket_mounts:
                if cloud_bucket_mount.secret:
                    dep_tasks.append(resolver.load(cloud_bucket_mount.secret, load_context))
            if proxy:
                dep_tasks.append(resolver.load(proxy, load_context))
            dep_timings = await _gather_load_with_timings(dep_tasks) if dep_tasks else []

            # Validate that the same volume (by object_id) isn't mounted at multiple paths
            validate_volumes_by_object_id(validated_volumes)

            # Relies on dicts being ordered (true as of Python 3.6).
            volume_mounts = [_volume_to_mount_proto(path, volume) for path, volume in validated_volumes]

            open_ports = [api_pb2.PortSpec(port=port, unencrypted=False) for port in encrypted_ports]
            open_ports.extend([api_pb2.PortSpec(port=port, unencrypted=True) for port in unencrypted_ports])
            open_ports.extend(
                [
                    api_pb2.PortSpec(port=port, unencrypted=False, tunnel_type=api_pb2.TUNNEL_TYPE_H2)
                    for port in h2_ports
                ]
            )

            if block_network:
                if outbound_cidr_allowlist is not None:
                    raise InvalidError("`outbound_cidr_allowlist` cannot be used when `block_network` is enabled")
                if outbound_domain_allowlist is not None:
                    raise InvalidError("`outbound_domain_allowlist` cannot be used when `block_network` is enabled")
                if inbound_cidr_allowlist is not None:
                    raise InvalidError("`inbound_cidr_allowlist` cannot be used when `block_network` is enabled")
                network_access = api_pb2.NetworkAccess(
                    network_access_type=api_pb2.NetworkAccess.NetworkAccessType.BLOCKED,
                )
            else:
                if outbound_domain_allowlist is None and outbound_cidr_allowlist is None:
                    network_access = api_pb2.NetworkAccess(
                        network_access_type=api_pb2.NetworkAccess.NetworkAccessType.OPEN,
                    )
                else:
                    network_access = api_pb2.NetworkAccess(
                        network_access_type=api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST,
                        allowed_cidrs=list(outbound_cidr_allowlist or []),
                        allowed_domains=list(outbound_domain_allowlist or []),
                    )

            ephemeral_disk = None  # Ephemeral disk requests not supported on Sandboxes.
            definition = api_pb2.Sandbox(
                entrypoint_args=args,
                image_id=image.object_id,
                mount_ids=[mount.object_id for mount in mounts] + [mount.object_id for mount in image._mount_layers],
                secret_ids=[secret.object_id for secret in secrets],
                timeout_secs=timeout,
                idle_timeout_secs=idle_timeout,
                workdir=workdir,
                resources=convert_fn_config_to_resources_config(
                    cpu=cpu, memory=memory, gpu=gpu, ephemeral_disk=ephemeral_disk
                ),
                cloud_provider_str=cloud if cloud else None,  # Supersedes cloud_provider
                nfs_mounts=network_file_system_mount_protos(validated_network_file_systems),
                runtime=config.get("function_runtime"),
                runtime_debug=config.get("function_runtime_debug"),
                cloud_bucket_mounts=cloud_bucket_mounts_to_proto(cloud_bucket_mounts),
                volume_mounts=volume_mounts,
                pty_info=pty_info,
                scheduler_placement=scheduler_placement,
                worker_id=config.get("worker_id"),
                open_ports=api_pb2.PortSpecs(ports=open_ports),
                network_access=network_access,
                proxy_id=(proxy.object_id if proxy else None),
                readiness_probe=(readiness_probe._to_proto() if readiness_probe else None),
                enable_snapshot=enable_snapshot,
                verbose=verbose,
                name=name,
                experimental_options_v2=(
                    {k: str(v) for k, v in experimental_options.items()} if experimental_options else None
                ),
                custom_domain=custom_domain,
                include_oidc_identity_token=include_oidc_identity_token,
                inbound_cidr_allowlist=list(inbound_cidr_allowlist) if inbound_cidr_allowlist is not None else [],
            )

            tag_protos = [api_pb2.SandboxTag(tag_name=k, tag_value=v) for k, v in tags.items()] if tags else []

            create_req = api_pb2.SandboxCreateRequest(
                app_id=load_context.app_id, definition=definition, tags=tag_protos
            )
            rpc_start = time.monotonic()
            create_resp = await load_context.client.stub.SandboxCreate(create_req)
            rpc_elapsed = time.monotonic() - rpc_start
            sandbox_id = create_resp.sandbox_id
            self._hydrate(sandbox_id, load_context.client, None)

            if logger.isEnabledFor(logging.DEBUG):
                total_elapsed = time.monotonic() - load_start
                logger.debug(_format_sandbox_create_timing_log(sandbox_id, total_elapsed, rpc_elapsed, dep_timings))

        return _Sandbox._from_loader(_load, "Sandbox()", load_context_overrides=LoadContext.empty())

    @staticmethod
    async def create(
        *args: str,
        app: "modal.app._App | None" = None,
        name: str | None = None,
        tags: dict[str, str] | None = None,
        image: _Image | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        network_file_systems: dict[str | os.PathLike, _NetworkFileSystem] = {},
        timeout: int = 300,
        idle_timeout: int | None = None,
        workdir: str | None = None,
        gpu: str | None = None,
        cloud: str | None = None,
        region: str | Sequence[str] | None = None,
        cpu: float | tuple[float, float] | None = None,
        memory: int | tuple[int, int] | None = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        inbound_cidr_allowlist: Sequence[str] | None = None,
        volumes: dict[str | os.PathLike, _Volume | _CloudBucketMount] = {},
        pty: bool = False,
        encrypted_ports: Sequence[int] = [],
        h2_ports: Sequence[int] = [],
        unencrypted_ports: Sequence[int] = [],
        custom_domain: str | None = None,
        proxy: _Proxy | None = None,
        include_oidc_identity_token: bool = False,
        readiness_probe: Probe | None = None,
        verbose: bool = False,
        experimental_options: dict[str, Any] | None = None,
        _experimental_enable_snapshot: bool = False,
        client: _Client | None = None,
        environment_name: str | None = None,  # *DEPRECATED*
        pty_info: api_pb2.PTYInfo | None = None,  # *DEPRECATED*
        cidr_allowlist: Sequence[str] | None = None,  # *DEPRECATED*
    ) -> "_Sandbox":
        """
        Create a new Sandbox to run untrusted, arbitrary code.

        The Sandbox's corresponding container will be created asynchronously.

        Args:
            *args: Set the CMD of the Sandbox, overriding any CMD of the container image.
            app: Associate the sandbox with an app. Required unless creating from a container.
            name: Optionally give the sandbox a name. Unique within an app.
            tags: Tags to assign to the Sandbox.
            image: The image to run as the container for the sandbox.
            env: Environment variables to set in the Sandbox.
            secrets: Secrets to inject into the Sandbox as environment variables.
            network_file_systems: Network file systems to mount into the sandbox.
            timeout: Maximum lifetime of the sandbox in seconds.
            idle_timeout: The amount of time in seconds that a sandbox can be idle before being terminated.
            workdir: Working directory of the sandbox.
            gpu: GPU reservation for the sandbox.
            cloud: Cloud provider for the sandbox.
            region: Region or regions to run the sandbox on.
            cpu:
                Specify, in fractional CPU cores, how many CPU cores to request. Or, pass (request, limit) to
                additionally specify a hard limit in fractional CPU cores. CPU throttling will prevent a container
                from exceeding its specified limit.
            memory:
                Specify, in MiB, a memory request which is the minimum memory required. Or, pass (request, limit) to
                additionally specify a hard limit in MiB.
            block_network: Whether to block network access.
            outbound_cidr_allowlist: List of CIDRs the sandbox is allowed to access. If None, all CIDRs are allowed.
            outbound_domain_allowlist: List of domain names the sandbox is allowed to access. Supports
                wildcard prefixes (``*.``); a bare ``"*"`` allows all domains. The outbound policy
                can be replaced later via `Sandbox._experimental_set_outbound_network_policy`.
            inbound_cidr_allowlist:
                List of CIDRs allowed to connect inbound to the sandbox (tunnels and connection tokens). If None,
                all CIDRs are allowed.
            volumes: Mount points for Modal Volumes and CloudBucketMounts.
            pty:
                Enable a PTY for the Sandbox entrypoint command. When enabled, all output (stdout and stderr from the
                process) is multiplexed into stdout, and the stderr stream is effectively empty.
            encrypted_ports: List of ports to tunnel into the sandbox. Encrypted ports are tunneled with TLS.
            h2_ports: List of encrypted ports to tunnel into the sandbox, using HTTP/2.
            unencrypted_ports: List of ports to tunnel into the sandbox without encryption.
            custom_domain:
                Allow connections to the Sandbox via a subdomain of this parent rather than a default Modal domain.
            proxy: Reference to a Modal Proxy to use in front of this Sandbox.
            include_oidc_identity_token:
                If True, the sandbox will receive a MODAL_IDENTITY_TOKEN env var for OIDC-based auth.
            readiness_probe: Probe used to determine when the sandbox has become ready.
            verbose: Enable verbose logging for sandbox operations.
            experimental_options: Experimental options to pass to the sandbox.
            _experimental_enable_snapshot: Enable memory snapshots.
            client: Modal Client to use for the sandbox.
            environment_name: *DEPRECATED* Optionally override the default environment
            pty_info: *DEPRECATED* Use `pty` instead. `pty` will override `pty_info`.
            cidr_allowlist: *DEPRECATED* Use outbound_cidr_allowlist instead.

        Returns:
            A `Sandbox` object representing the created sandbox which can be used to interact with the sandbox.

        Raises:
            AlreadyExistsError: If a sandbox with the same name already exists.

        Examples:
            ```python
            app = modal.App.lookup('sandbox-hello-world', create_if_missing=True)
            sandbox = modal.Sandbox.create("echo", "hello world", app=app)
            print(sandbox.stdout.read())
            sandbox.wait()
            ```
        """
        if environment_name is not None:
            deprecation_warning(
                (2025, 7, 16),
                "Passing `environment_name` to `Sandbox.create` is deprecated and will be removed in a future release. "
                "A sandbox's environment is determined by the app it is associated with.",
            )

        if pty_info is not None:
            deprecation_warning(
                (2025, 9, 12),
                "The `pty_info` parameter is deprecated and will be removed in a future release. "
                "Set the `pty` parameter to `True` instead.",
            )

        if cidr_allowlist is not None:
            if outbound_cidr_allowlist is not None:
                raise InvalidError("Cannot specify both `cidr_allowlist` and `outbound_cidr_allowlist`.")
            deprecation_warning(
                (2026, 5, 11),
                "The `cidr_allowlist` parameter has been renamed to `outbound_cidr_allowlist`. "
                "`cidr_allowlist` will be removed in a future release.",
            )
            outbound_cidr_allowlist = cidr_allowlist

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        return await _Sandbox._create(
            *args,
            app=app,
            name=name,
            image=image,
            secrets=secrets,
            network_file_systems=network_file_systems,
            timeout=timeout,
            idle_timeout=idle_timeout,
            workdir=workdir,
            gpu=gpu,
            cloud=cloud,
            region=region,
            cpu=cpu,
            memory=memory,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            inbound_cidr_allowlist=inbound_cidr_allowlist,
            volumes=volumes,
            pty=pty,
            encrypted_ports=encrypted_ports,
            h2_ports=h2_ports,
            unencrypted_ports=unencrypted_ports,
            proxy=proxy,
            readiness_probe=readiness_probe,
            experimental_options=experimental_options,
            tags=tags,
            _experimental_enable_snapshot=_experimental_enable_snapshot,
            include_oidc_identity_token=include_oidc_identity_token,
            client=client,
            verbose=verbose,
            pty_info=pty_info,
            custom_domain=custom_domain,
        )

    @staticmethod
    async def _create(
        *args: str,
        app: "modal.app._App | None" = None,
        name: str | None = None,
        tags: dict[str, str] | None = None,
        image: _Image | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        mounts: Sequence[_Mount] = (),
        network_file_systems: dict[str | os.PathLike, _NetworkFileSystem] = {},
        timeout: int = 300,
        idle_timeout: int | None = None,
        workdir: str | None = None,
        gpu: str | None = None,
        cloud: str | None = None,
        region: str | Sequence[str] | None = None,
        cpu: float | tuple[float, float] | None = None,
        memory: int | tuple[int, int] | None = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        inbound_cidr_allowlist: Sequence[str] | None = None,
        volumes: dict[str | os.PathLike, _Volume | _CloudBucketMount] = {},
        pty: bool = False,
        encrypted_ports: Sequence[int] = [],
        h2_ports: Sequence[int] = [],
        unencrypted_ports: Sequence[int] = [],
        proxy: _Proxy | None = None,
        include_oidc_identity_token: bool = False,
        readiness_probe: Probe | None = None,
        experimental_options: dict[str, Any] | None = None,
        _experimental_enable_snapshot: bool = False,
        client: _Client | None = None,
        verbose: bool = False,
        pty_info: api_pb2.PTYInfo | None = None,
        custom_domain: str | None = None,
    ):
        """Private method used internally.

        This method exposes some internal arguments (currently `mounts`) which are not in the public API.
        `mounts` is currently only used by modal shell (cli) to provide a function's mounts to the
        sandbox that runs the shell session.
        """
        from .app import _App

        _validate_exec_args(args)
        if name is not None:
            check_object_name(name, "Sandbox")

        if block_network and (encrypted_ports or h2_ports or unencrypted_ports):
            raise InvalidError("Cannot specify open ports when `block_network` is enabled")

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        # TODO(erikbern): Get rid of the `_new` method and create an already-hydrated object
        obj = _Sandbox._new(
            args,
            image=image or _default_image,
            secrets=secrets,
            name=name,
            timeout=timeout,
            idle_timeout=idle_timeout,
            workdir=workdir,
            gpu=gpu,
            cloud=cloud,
            region=region,
            cpu=cpu,
            memory=memory,
            mounts=mounts,
            network_file_systems=network_file_systems,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            inbound_cidr_allowlist=inbound_cidr_allowlist,
            volumes=volumes,
            pty=pty,
            pty_info=pty_info,
            encrypted_ports=encrypted_ports,
            h2_ports=h2_ports,
            unencrypted_ports=unencrypted_ports,
            proxy=proxy,
            readiness_probe=readiness_probe,
            experimental_options=experimental_options,
            tags=tags,
            enable_snapshot=_experimental_enable_snapshot,
            verbose=verbose,
            custom_domain=custom_domain,
            include_oidc_identity_token=include_oidc_identity_token,
        )
        obj._enable_snapshot = _experimental_enable_snapshot

        app_id: str | None = None
        app_client: _Client | None = None

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
            # implicit app/client provided by running in a modal Function
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

        client = client or app_client

        resolver = Resolver()
        async with TaskContext() as tc:
            load_context = LoadContext(client=client, app_id=app_id, task_context=tc)
            await resolver.load(obj, load_context)
        return obj

    @staticmethod
    async def _experimental_create(
        *args: str,
        app: "modal.app._App | None" = None,
        name: str | None = None,
        tags: dict[str, str] | None = None,
        image: _Image | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        timeout: int = 300,
        idle_timeout: int | None = None,
        workdir: str | None = None,
        cpu: float | None = None,
        memory: int | None = None,
        cloud: str | None = None,
        region: str | Sequence[str] | None = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        inbound_cidr_allowlist: Sequence[str] | None = None,
        i6pn: bool = False,
        volumes: dict[str | os.PathLike, _Volume | _CloudBucketMount] = {},
        pty: bool = False,
        encrypted_ports: Sequence[int] = [],
        h2_ports: Sequence[int] = [],
        unencrypted_ports: Sequence[int] = [],
        proxy: _Proxy | None = None,
        readiness_probe: Probe | None = None,
        experimental_options: dict[str, Any] | None = None,
        include_oidc_identity_token: bool = False,
        verbose: bool = False,
        client: _Client | None = None,
    ) -> "_Sandbox":
        """Create a sandbox using the V2 backend.

        Supported features include exec, encrypted tunnels, wait/poll/terminate,
        CPU and memory configuration, region placement, private IPv6 networking
        (i6pn), volumes, cloud bucket mounts (with static credentials via
        `secret=...` or `oidc_auth_role_arn`), OIDC identity tokens, proxies, and
        filesystem snapshots.

        Features like memory snapshots, network file systems, GPUs, and custom
        domains are not supported.

        Set `i6pn=True` to enable private IPv6 networking so sandboxes in the same
        workspace can address each other directly at their `i6pn.modal.local`
        address. i6pn only connects sandboxes co-located on the same routable
        network, so pin every sandbox in the group to the same specific region
        (e.g. `region="us-east-1"`).

        V2 sandboxes created with this method are not currently returned by
        `Sandbox.list()`. A named sandbox can be looked up with
        `Sandbox._experimental_from_name(app_name, name)`; otherwise store
        `sandbox.object_id` and use `Sandbox.from_id(sandbox.object_id)` to
        reattach.
        """
        from .app import _App

        _validate_exec_args(args)
        if name is not None:
            check_object_name(name, "Sandbox")

        if workdir is not None and not workdir.startswith("/"):
            raise InvalidError(f"workdir must be an absolute path, got: {workdir}")

        if block_network and (encrypted_ports or h2_ports or unencrypted_ports):
            raise InvalidError("Cannot specify open ports when `block_network` is enabled")

        if block_network and i6pn:
            raise InvalidError(
                "`block_network` disables all networking, including i6pn. To keep i6pn while blocking "
                "public egress, use an empty outbound allowlist (`outbound_cidr_allowlist=[]`) instead."
            )

        validated_volumes = validate_volumes(volumes)
        cloud_bucket_mounts = [(k, v) for k, v in validated_volumes if isinstance(v, _CloudBucketMount)]
        validated_volumes = [(k, v) for k, v in validated_volumes if isinstance(v, _Volume)]

        secrets = secrets or []

        env_dict, resolvable_secrets = _split_env_dict_and_resolvable_secrets(secrets)
        if env:
            env_type_err = "the env argument to Sandbox must be a dict[str, str | None]"
            if not isinstance(env, dict):
                raise InvalidError(env_type_err)
            ephemeral_env = {k: v for k, v in env.items() if v is not None}
            if not all(isinstance(k, str) for k in ephemeral_env) or not all(
                isinstance(v, str) for v in ephemeral_env.values()
            ):
                raise InvalidError(env_type_err)
            _validate_sandbox_env(ephemeral_env)
            # `env` has a higher precedience over environment variables from secrets
            env_dict |= ephemeral_env

        image = image or _default_image

        scheduler_placement: api_pb2.SchedulerPlacement | None = None
        if region:
            regions = [region] if isinstance(region, str) else list(region)
            scheduler_placement = api_pb2.SchedulerPlacement(regions=regions)

        pty_info: api_pb2.PTYInfo | None = None
        if pty:
            pty_info = _Sandbox._default_pty_info()

        open_ports = [api_pb2.PortSpec(port=port, unencrypted=False) for port in encrypted_ports]
        open_ports.extend([api_pb2.PortSpec(port=port, unencrypted=True) for port in unencrypted_ports])
        open_ports.extend(
            [api_pb2.PortSpec(port=port, unencrypted=False, tunnel_type=api_pb2.TUNNEL_TYPE_H2) for port in h2_ports]
        )

        if block_network:
            if outbound_cidr_allowlist is not None:
                raise InvalidError("`outbound_cidr_allowlist` cannot be used when `block_network` is enabled")
            if outbound_domain_allowlist is not None:
                raise InvalidError("`outbound_domain_allowlist` cannot be used when `block_network` is enabled")
            if inbound_cidr_allowlist is not None:
                raise InvalidError("`inbound_cidr_allowlist` cannot be used when `block_network` is enabled")
            network_access = api_pb2.NetworkAccess(
                network_access_type=api_pb2.NetworkAccess.NetworkAccessType.BLOCKED,
            )
        elif outbound_domain_allowlist is None and outbound_cidr_allowlist is None:
            network_access = api_pb2.NetworkAccess(
                network_access_type=api_pb2.NetworkAccess.NetworkAccessType.OPEN,
            )
        else:
            network_access = api_pb2.NetworkAccess(
                network_access_type=api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST,
                allowed_cidrs=list(outbound_cidr_allowlist or []),
                allowed_domains=list(outbound_domain_allowlist or []),
            )

        async def _load(self: _Sandbox, resolver: Resolver, load_context: LoadContext, _existing_object_id: str | None):
            load_start = time.monotonic()
            dep_tasks: list = []
            if not image._is_hydrated:
                dep_tasks.append(resolver.load(image, load_context))
            for secret in resolvable_secrets:
                dep_tasks.append(resolver.load(secret, load_context))
            for _, vol in validated_volumes:
                dep_tasks.append(resolver.load(vol, load_context))
            for _, cloud_bucket_mount in cloud_bucket_mounts:
                if cloud_bucket_mount.secret:
                    dep_tasks.append(resolver.load(cloud_bucket_mount.secret, load_context))
            if proxy:
                dep_tasks.append(resolver.load(proxy, load_context))
            dep_timings = await _gather_load_with_timings(dep_tasks) if dep_tasks else []

            validate_volumes_by_object_id(validated_volumes)

            volume_mounts = [_volume_to_mount_proto(path, volume) for path, volume in validated_volumes]

            definition = api_pb2.Sandbox(
                entrypoint_args=args,
                image_id=image.object_id,
                mount_ids=[mount.object_id for mount in image._mount_layers],
                secret_ids=[secret.object_id for secret in resolvable_secrets],
                timeout_secs=timeout,
                idle_timeout_secs=idle_timeout,
                workdir=workdir,
                resources=convert_fn_config_to_resources_config(cpu=cpu, memory=memory, gpu=None, ephemeral_disk=None),
                cloud_provider_str=cloud if cloud else None,
                runtime=config.get("function_runtime"),
                runtime_debug=config.get("function_runtime_debug"),
                pty_info=pty_info,
                scheduler_placement=scheduler_placement,
                worker_id=config.get("worker_id"),
                open_ports=api_pb2.PortSpecs(ports=open_ports),
                network_access=network_access,
                proxy_id=(proxy.object_id if proxy else None),
                verbose=verbose,
                name=name,
                include_oidc_identity_token=include_oidc_identity_token,
                inbound_cidr_allowlist=list(inbound_cidr_allowlist) if inbound_cidr_allowlist is not None else [],
                i6pn_enabled=i6pn,
                volume_mounts=volume_mounts,
                cloud_bucket_mounts=cloud_bucket_mounts_to_proto(cloud_bucket_mounts),
                readiness_probe=(readiness_probe._to_proto() if readiness_probe else None),
                experimental_options_v2=(
                    {k: str(v) for k, v in experimental_options.items()} if experimental_options else None
                ),
            )

            tag_protos = [api_pb2.SandboxTag(tag_name=k, tag_value=v) for k, v in tags.items()] if tags else []
            create_req = api_pb2.SandboxCreateV2Request(
                app_id=load_context.app_id,
                definition=definition,
                ephemeral_secrets=api_pb2.StringMap(contents=env_dict) if env_dict else None,
                tags=tag_protos,
            )
            assert load_context.client._auth_token_manager
            auth_token = await load_context.client._auth_token_manager.get_token()
            rpc_start = time.monotonic()
            create_resp = await load_context.client.stub.SandboxCreateV2(
                create_req, metadata=[("x-modal-auth-token", auth_token)]
            )
            rpc_elapsed = time.monotonic() - rpc_start
            sandbox_id = create_resp.sandbox_id
            self._hydrate(sandbox_id, load_context.client, None)
            self._is_v2 = True
            self._task_id = create_resp.task_id
            self._hydrate_metadata_v2()
            self._tunnels = {
                t.container_port: Tunnel(t.host, t.port, t.unencrypted_host, t.unencrypted_port)
                for t in create_resp.tunnels
            }

            if logger.isEnabledFor(logging.DEBUG):
                total_elapsed = time.monotonic() - load_start
                logger.debug(_format_sandbox_create_timing_log(sandbox_id, total_elapsed, rpc_elapsed, dep_timings))

        obj = _Sandbox._from_loader(_load, "Sandbox()", load_context_overrides=LoadContext.empty())

        app_id: str | None = None
        app_client: _Client | None = None

        if app is not None:
            if app.app_id is None:
                raise ValueError(
                    "App has not been initialized yet. To create an App lazily, use `App.lookup`: \n"
                    "app = modal.App.lookup('my-app', create_if_missing=True)\n"
                    "modal.Sandbox._experimental_create('echo', 'hi', app=app)\n"
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
                "sb = modal.Sandbox._experimental_create(..., app=app)\n"
                "```",
            )

        client = client or app_client

        resolver = Resolver()
        async with TaskContext() as tc:
            load_context = LoadContext(client=client, app_id=app_id, task_context=tc)
            await resolver.load(obj, load_context)
        return obj

    def _hydrate_metadata(self, handle_metadata: Message | None):
        self._stdout = StreamReader(
            _StreamReaderThroughServerParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                object_id=self.object_id,
                client=self._client,
            ),
            by_line=True,
        )
        self._stderr = StreamReader(
            _StreamReaderThroughServerParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDERR,
                object_id=self.object_id,
                client=self._client,
            ),
            by_line=True,
        )
        self._stdin = StreamWriter(_StreamWriterThroughServerParams(object_id=self.object_id, client=self._client))
        self._result = None
        self._task_id = None
        self._tunnels = None
        self._enable_snapshot = False
        self._command_router_client = None
        self._filesystem = None
        self._is_v2 = False

    def _hydrate_metadata_v2(self) -> None:
        """Wire up V2 stdio readers that read directly from the worker. Cheap
        to call eagerly: the router connection is opened lazily on first read.
        """
        # TODO: combine with `_hydrate_metadata`, so we can set `stdout` & `stderr` differently for V1 and V2 sandboxes.
        weak_self = weakref.ref(self)

        async def resolve_router() -> tuple[str, TaskCommandRouterClient]:
            this = weak_self()
            if this is None:
                raise RuntimeError("Sandbox was garbage collected before its stdio reader connected")
            task_id = await this._get_task_id()
            router = await this._get_command_router_client(task_id)
            return task_id, router

        self._stdout = StreamReader(
            _StreamReaderThroughSandboxCommandRouterParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                sandbox_id=self.object_id,
                resolve_router=resolve_router,
            ),
            by_line=True,
        )
        self._stderr = StreamReader(
            _StreamReaderThroughSandboxCommandRouterParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDERR,
                sandbox_id=self.object_id,
                resolve_router=resolve_router,
            ),
            by_line=True,
        )
        self._stdin = StreamWriter(_StreamWriterThroughCommandRouterSandboxParams(resolve_router=resolve_router))

    def _initialize_from_other(self, other):
        super()._initialize_from_other(other)
        self._attached = other._attached
        self._is_v2 = other._is_v2

    def _initialize_from_empty(self):
        super()._initialize_from_empty()
        self._attached = True
        self._is_v2 = False

    async def detach(self):
        """Disconnects your client from the sandbox and cleans up resources assoicated with the connection.

        Be sure to only call `detach` when you are done interacting with the sandbox. After calling `detach`,
        any operation using the Sandbox object is not guaranteed to work anymore. If you want to continue interacting
        with a running sandbox, use `Sandbox.from_id` to get a new Sandbox object.

        """
        if not self._attached:
            return
        if self._command_router_client is not None:
            await self._command_router_client.close()
        self._attached = False

    @property
    def _client(self) -> _Client:
        self._ensure_attached()
        return self.__client

    @_client.setter
    def _client(self, value):
        self.__client = value

    def _ensure_attached(self):
        if not self._attached:
            raise ClientClosed("Unable to perform operation on a detached sandbox")

    def _ensure_v1(self, method_name: str):
        if self._is_v2:
            raise InvalidError(f"Sandbox.{method_name}() is not supported for V2 sandboxes")

    @staticmethod
    async def from_name(
        app_name: str,
        name: str,
        *,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> "_Sandbox":
        """Get a running Sandbox by name from a deployed App.

        A Sandbox's name is the `name` argument passed to `Sandbox.create`.

        Args:
            app_name: Name of the deployed app to look up the sandbox under.
            name: Sandbox name to resolve.
            environment_name: Optional environment name for the lookup; defaults to the configured environment.
            client: Modal client to use for the RPC; defaults to `Client.from_env()` when omitted.

        Returns:
            A `Sandbox` handle for the running sandbox.

        Raises:
            NotFoundError: If no running sandbox exists with the given name.
        """
        if client is None:
            client = await _Client.from_env()
        env_name = _get_environment_name(environment_name)

        req = api_pb2.SandboxGetFromNameRequest(sandbox_name=name, app_name=app_name, environment_name=env_name)
        resp = await client.stub.SandboxGetFromName(req)
        return _Sandbox._new_hydrated(resp.sandbox_id, client, None)

    @staticmethod
    async def _experimental_from_name(
        app_name: str,
        name: str,
        *,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> "_Sandbox":
        """Get a running V2 Sandbox by name from a deployed App.

        This looks up V2 sandboxes, ie sandboxes created via `modal.Sandbox._experimental_create`.

        Args:
            app_name: Name of the deployed app to look up the sandbox under.
            name: Sandbox name to resolve.
            environment_name: Optional environment name for the lookup; defaults to the configured environment.
            client: Modal client to use for the RPC; defaults to `Client.from_env()` when omitted.

        Returns:
            A `Sandbox` handle for the running sandbox.

        Raises:
            NotFoundError: If no running sandbox exists with the given name.
        """
        if client is None:
            client = await _Client.from_env()
        env_name = _get_environment_name(environment_name)

        req = api_pb2.SandboxGetFromNameRequest(sandbox_name=name, app_name=app_name, environment_name=env_name)
        assert client._auth_token_manager
        auth_token = await client._auth_token_manager.get_token()
        resp = await client.stub.SandboxGetFromNameV2(req, metadata=[("x-modal-auth-token", auth_token)])

        obj = _Sandbox._new_hydrated(resp.sandbox_id, client, None)
        obj._is_v2 = True
        obj._hydrate_metadata_v2()
        return obj

    @staticmethod
    async def from_id(sandbox_id: str, client: _Client | None = None) -> "_Sandbox":
        """Construct a Sandbox from an id and look up the Sandbox result.

        The ID of a Sandbox object can be accessed using `.object_id`.

        Args:
            sandbox_id: Sandbox object ID to attach to.
            client: Modal client to use for the lookup; defaults to the environment client when omitted.

        Returns:
            A `Sandbox` handle with any available result metadata populated from the server.
        """
        if client is None:
            client = await _Client.from_env()

        sandbox_version = _get_sandbox_version(sandbox_id)
        is_v2 = sandbox_version == SandboxVersion.V2
        req = api_pb2.SandboxWaitRequest(sandbox_id=sandbox_id, timeout=0)
        if is_v2:
            assert client._auth_token_manager
            auth_token = await client._auth_token_manager.get_token()
            resp = await client.stub.SandboxWaitV2(req, metadata=[("x-modal-auth-token", auth_token)])
        else:
            resp = await client.stub.SandboxWait(req)

        obj = _Sandbox._new_hydrated(sandbox_id, client, None)
        obj._is_v2 = is_v2
        if is_v2:
            obj._hydrate_metadata_v2()

        if resp.result.status:
            obj._result = resp.result

        return obj

    async def get_tags(self) -> dict[str, str]:
        """Fetches any tags (key-value pairs) currently attached to this Sandbox from the server.

        Returns:
            Tags as a map from tag name to tag value.
        """
        req = api_pb2.SandboxTagsGetRequest(sandbox_id=self.object_id)
        stub = self._client.stub
        if self._is_v2:
            assert self._client._auth_token_manager
            auth_token = await self._client._auth_token_manager.get_token()
            resp = await stub.SandboxTagsGetV2(req, metadata=[("x-modal-auth-token", auth_token)])
        else:
            resp = await stub.SandboxTagsGet(req)

        return {tag.tag_name: tag.tag_value for tag in resp.tags}

    async def set_tags(self, tags: dict[str, str], *, client: _Client | None = None) -> None:
        """Set tags (key-value pairs) on the Sandbox. Tags can be used to filter results in `Sandbox.list`.

        Setting tags replaces the Sandbox's entire tag set; passing an empty dict clears all tags.

        Args:
            tags: Tag names and values to set on this sandbox.
            client: Deprecated. Prefer setting the client when creating or re-attaching to the sandbox.

        """
        if client is not None:
            deprecation_warning(
                (2025, 9, 18),
                "The `client` parameter is deprecated. Set `client` when creating the Sandbox instead "
                "(in e.g. `Sandbox.create()`/`.from_id()`/`.from_name()`).",
            )

        tags_list = [api_pb2.SandboxTag(tag_name=name, tag_value=value) for name, value in tags.items()]
        stub = self._client.stub
        if self._is_v2:
            assert self._client._auth_token_manager
            auth_token = await self._client._auth_token_manager.get_token()
            req = api_pb2.SandboxTagsSetRequest(sandbox_id=self.object_id, tags=tags_list)
            await stub.SandboxTagsSetV2(req, metadata=[("x-modal-auth-token", auth_token)])
        else:
            req = api_pb2.SandboxTagsSetRequest(
                environment_name=_get_environment_name(),
                sandbox_id=self.object_id,
                tags=tags_list,
            )
            await stub.SandboxTagsSet(req)

    async def _experimental_set_outbound_network_policy(
        self,
        *,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
    ) -> None:
        """Replace the outbound network policy of a running Sandbox.

        Established connections that the new policy no longer permits are
        terminated.

        Args:
            outbound_cidr_allowlist: List of CIDRs the Sandbox is allowed to access. If None, all CIDRs are allowed.
            outbound_domain_allowlist: List of domain names the Sandbox is allowed to access. Supports
                wildcard prefixes (``*.``); a bare ``"*"`` allows all domains.
        """
        task_id = await self._get_task_id()
        command_router_client = await self._get_command_router_client(task_id)

        if outbound_cidr_allowlist is not None or outbound_domain_allowlist is not None:
            network_access = api_pb2.NetworkAccess(
                network_access_type=api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST,
                allowed_cidrs=list(outbound_cidr_allowlist or []),
                allowed_domains=list(outbound_domain_allowlist or []),
            )
        else:
            network_access = api_pb2.NetworkAccess(
                network_access_type=api_pb2.NetworkAccess.NetworkAccessType.OPEN,
            )
        req = sr_pb2.TaskSetNetworkAccessRequest(task_id=task_id, network_access=network_access)
        await command_router_client.set_network_access(req)

    async def snapshot_filesystem(
        self,
        timeout: int = 55,
        *,
        ttl: int | None = 30 * 24 * 3600,
    ) -> _Image:
        """Snapshot the filesystem of the Sandbox.

        Args:
            timeout:
                Maximum time in seconds to wait for the snapshot operation. If the snapshot does not return within
                that window, the call is cancelled and `modal.exception.TimeoutError` is raised.
            ttl:
                The resulting Image is retained for `ttl` seconds (default: 30 days). Pass `ttl=None` to retain
                the image indefinitely.

        Returns:
            An [`Image`](https://modal.com/docs/sdk/py/latest/modal.Image) object which can be used to spawn a new
            Sandbox with the same filesystem.
        """
        if os.environ.get("MODAL_USE_LEGACY_FILESYSTEM_SNAPSHOT") == "1" and not self._is_v2:
            if ttl != 30 * 24 * 3600:
                raise InvalidError("ttl is not supported with MODAL_USE_LEGACY_FILESYSTEM_SNAPSHOT")
            return await self._legacy_snapshot_filesystem(timeout)

        wire_ttl_seconds = _ttl_to_wire_ttl(ttl)

        task_id = await self._get_task_id()
        command_router_client = await self._get_command_router_client(task_id)

        req = sr_pb2.TaskSnapshotFilesystemRequest(
            task_id=task_id,
            snapshot_id=str(uuid.uuid4()),
            ttl_seconds=wire_ttl_seconds,
        )
        res = await command_router_client.snapshot_filesystem(req, timeout=float(timeout))
        return _Image._new_hydrated(res.image_id, self._client, None)

    async def _legacy_snapshot_filesystem(self, timeout: int = 55) -> _Image:
        self._ensure_v1("snapshot_filesystem")
        await self._get_task_id()
        req = api_pb2.SandboxSnapshotFsRequest(sandbox_id=self.object_id, timeout=timeout)
        resp = await self._client.stub.SandboxSnapshotFs(req)

        if resp.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
            raise ExecutionError(resp.result.exception)

        return _Image._new_hydrated(resp.image_id, self._client, resp.image_metadata)

    async def mount_image(
        self,
        path: PurePosixPath | str,
        image: _Image,
        *,
        _experimental_encryption_key: bytes | None = None,
    ):
        """Mount an Image at a specified path in a running Sandbox.

        `path` should be a directory that is **not** the root path (`/`). If the path doesn't exist
        it will be created. If it exists and contains data, the previous directory will be replaced
        by the mount.

        The `image` argument supports any Image that has an object ID, including:
        - Images built using `image.build()`
        - Images referenced by ID, e.g. `Image.from_id(...)`
        - Filesystem/directory snapshots, e.g. created by `.snapshot_directory()` or `.snapshot_filesystem()`
        - Empty images created with `Image.from_scratch()`

        Args:
            path: Absolute mount point directory inside the sandbox (not `/`).
            image: Image to mount at `path` (must be built, referenced by ID, or snapshot-based as described above).


        Examples:
            ```py notest
            user_project_snapshot: Image = sandbox_session_1.snapshot_directory("/user_project")

            # You can later mount this snapshot to another Sandbox:
            sandbox_session_2 = modal.Sandbox.create(...)
            sandbox_session_2.mount_image("/user_project", user_project_snapshot)
            sandbox_session_2.filesystem.list_files("/user_project")
            ```
        """
        if not isinstance(image, _Image):
            raise TypeError(f"Sandbox.mount_image(image=...) expects an Image object, got {image!r}")

        if image._mount_layers:
            raise InvalidError(
                "Sandbox.mount_image() only supports pre-built images. When using `add_local*` methods, "
                "specify `copy=True` and call `.build()` before passing the image to `mount_image()`:\n\nE.g.\n"
                'img = modal.Image.debian_slim().add_local_file("foo", "/foo", copy=True).build(app)\n'
                "sandbox.mount_image(path, img)"
            )
        if image._is_empty:
            image_id = ""
        elif image._object_id:
            image_id = image._object_id
        else:
            raise InvalidError(
                "Sandbox.mount_image() currently only supports Images that are either:\n"
                "- prebuilt using `image.build()`\n"
                "- referenced by id, e.g. `Image.from_id()`\n"
                "- filesystem/directory snapshots e.g. created by `.snapshot_directory()` "
                "or `.snapshot_filesystem()`\n"
            )

        task_id = await self._get_task_id()
        command_router_client = await self._get_command_router_client(task_id)

        posix_path = PurePosixPath(path)
        if not posix_path.is_absolute():
            raise InvalidError(f"Mount path must be absolute; got: {posix_path}")
        path_bytes = posix_path.as_posix().encode("utf8")

        req = sr_pb2.TaskMountDirectoryRequest(
            task_id=task_id,
            path=path_bytes,
            image_id=image_id,
            customer_supplied_encryption_key=_validate_experimental_encryption_key(_experimental_encryption_key),
        )
        await command_router_client.mount_image(req)

    async def unmount_image(self, path: PurePosixPath | str):
        """Unmount a previously mounted Image from a running Sandbox.

        `path` must be the exact mount point that was passed to `.mount_image()`.
        After unmounting, the underlying Sandbox filesystem at that path becomes
        visible again.

        Args:
            path: Absolute mount point directory to unmount.

        """
        task_id = await self._get_task_id()
        command_router_client = await self._get_command_router_client(task_id)

        posix_path = PurePosixPath(path)
        if not posix_path.is_absolute():
            raise InvalidError(f"Unmount path must be absolute; got: {posix_path}")
        path_bytes = posix_path.as_posix().encode("utf8")

        req = sr_pb2.TaskUnmountDirectoryRequest(task_id=task_id, path=path_bytes)
        await command_router_client.unmount_image(req)

    async def snapshot_directory(
        self,
        path: PurePosixPath | str,
        *,
        timeout: int = 55,
        ttl: int | None = 30 * 24 * 3600,
        _experimental_encryption_key: bytes | None = None,
    ) -> _Image:
        """Snapshot a directory in a running Sandbox, creating a new Image with its content.

        `timeout` If the snapshot does not return within that window, the call is cancelled
        and `modal.exception.TimeoutError` is raised.

        `ttl` The resulting Image is retained for `ttl` seconds (default: 30 days)
        Pass `ttl=None` to retain the image indefinitely.

        Args:
            path: Absolute path of the directory inside the sandbox to snapshot.

        Returns:
            An `Image` containing the directory contents.

        Examples:
            ```py notest
            user_project_snapshot: Image = sandbox_session_1.snapshot_directory("/user_project")

            # You can later mount this snapshot to another Sandbox:
            sandbox_session_2 = modal.Sandbox.create(...)
            sandbox_session_2.mount_image("/user_project", user_project_snapshot)
            sandbox_session_2.filesystem.list_files("/user_project")
            ```
        """
        wire_ttl_seconds = _ttl_to_wire_ttl(ttl)

        task_id = await self._get_task_id()
        command_router_client = await self._get_command_router_client(task_id)

        posix_path = PurePosixPath(path)
        if not posix_path.is_absolute():
            raise InvalidError(f"Snapshot path must be absolute; got: {posix_path}")
        path_bytes = posix_path.as_posix().encode("utf8")

        snapshot_id = str(uuid.uuid4())
        req = sr_pb2.TaskSnapshotDirectoryRequest(
            task_id=task_id,
            path=path_bytes,
            snapshot_id=snapshot_id,
            ttl_seconds=wire_ttl_seconds,
            customer_supplied_encryption_key=_validate_experimental_encryption_key(_experimental_encryption_key),
        )
        res = await command_router_client.snapshot_directory(req, timeout=float(timeout))
        return _Image._new_hydrated(res.image_id, self._client, None)

    # Live handle methods

    async def wait(self, raise_on_termination: bool = True):
        """Wait for the Sandbox to finish running.

        Args:
            raise_on_termination: If True, raise when the sandbox is terminated externally.

        """

        while True:
            req = api_pb2.SandboxWaitRequest(sandbox_id=self.object_id, timeout=10)
            # Use the private __client to allow `wait` to work with a detached sandbox
            stub = self.__client.stub
            if self._is_v2:
                assert self.__client._auth_token_manager
                auth_token = await self.__client._auth_token_manager.get_token()
                resp = await stub.SandboxWaitV2(req, metadata=[("x-modal-auth-token", auth_token)])
            else:
                resp = await stub.SandboxWait(req)
            if resp.result.status:
                logger.debug(f"Sandbox {self.object_id} wait completed with status {resp.result.status}")
                self._result = resp.result

                if resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
                    raise SandboxTimeoutError()
                elif resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED and raise_on_termination:
                    raise SandboxTerminatedError()
                break

    async def wait_until_ready(self, *, timeout: int = 300) -> None:
        """Wait for the Sandbox readiness probe to report that the Sandbox is ready.

        The Sandbox must be configured with a `readiness_probe` in order to use this method.

        Args:
            timeout: Maximum time in seconds to wait for readiness.


        Examples:
            ```py notest
            app = modal.App.lookup('sandbox-wait-until-ready', create_if_missing=True)
            sandbox = modal.Sandbox.create(
                "python3", "-m", "http.server", "8080",
                readiness_probe=modal.Probe.with_tcp(8080),
                app=app,
            )
            sandbox.wait_until_ready()
            ```
        """
        if timeout <= 0:
            raise InvalidError(f"`timeout` must be positive, got: {timeout}")

        # Route to the task command router for both V1 and V2 sandboxes.
        task_id = await self._get_task_id(raise_if_task_complete=True)
        try:
            command_router_client = await self._get_command_router_client(task_id)
        except NotFoundError as e:
            # We do this to maintain backwards compatibility within wait_until_ready.
            # The V1 implementation would raise ConflictError instead of NotFoundError
            # if the sandbox was terminated, so we do the same for V2.
            raise ConflictError(str(e)) from e
        await command_router_client.sandbox_wait_until_ready(task_id, timeout=timeout)

    async def tunnels(self, timeout: int = 50) -> dict[int, Tunnel]:
        """Get Tunnel metadata for the sandbox.

        NOTE: Previous to client [v0.64.153](https://modal.com/docs/sdk/py/changelog#064153-2024-09-30), this
        returned a list of `TunnelData` objects.

        Args:
            timeout: Maximum time in seconds to wait for tunnel metadata when not already cached.

        Returns:
            A dictionary mapping container port to `Tunnel` metadata.

        Raises:
            SandboxTimeoutError: If the tunnels are not available after the timeout.
        """

        if self._tunnels:
            return self._tunnels

        req = api_pb2.SandboxGetTunnelsRequest(sandbox_id=self.object_id, timeout=timeout)
        stub = self._client.stub
        if self._is_v2:
            assert self._client._auth_token_manager
            auth_token = await self._client._auth_token_manager.get_token()
            resp = await stub.SandboxGetTunnelsV2(req, metadata=[("x-modal-auth-token", auth_token)])
        else:
            resp = await stub.SandboxGetTunnels(req)

        # If we couldn't get the tunnels in time, report the timeout.
        if resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
            raise SandboxTimeoutError()

        # Otherwise, we got the tunnels and can report the result.
        self._tunnels = {
            t.container_port: Tunnel(t.host, t.port, t.unencrypted_host, t.unencrypted_port) for t in resp.tunnels
        }

        return self._tunnels

    async def create_connect_token(
        self, user_metadata: str | dict[str, Any] | None = None, port: int = 8080
    ) -> SandboxConnectCredentials:
        """Create a token for making HTTP connections to the Sandbox.

        Accepts an optional user_metadata string or dict to associate with the token. This metadata
        will be added to the headers by the proxy when forwarding requests to the Sandbox.
        Also accepts a port that requests will be routed to.

        Args:
            user_metadata: Optional JSON-serializable metadata or string stored with the connect token.
            port: Optional container port that requests are routed to when using this token.

        Returns:
            URL and token credentials for connecting to the sandbox over HTTP.
        """
        self._ensure_v1("create_connect_token")
        if user_metadata is not None and isinstance(user_metadata, dict):
            try:
                user_metadata = json.dumps(user_metadata)
            except Exception as e:
                raise InvalidError(f"Failed to serialize user_metadata: {e}")

        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise InvalidError("port must be between 1 and 65535")

        req = api_pb2.SandboxCreateConnectTokenRequest(
            sandbox_id=self.object_id, user_metadata=user_metadata, port=port
        )
        resp = await self._client.stub.SandboxCreateConnectToken(req)
        return SandboxConnectCredentials(resp.url, resp.token)

    async def reload_volumes(self, *, timeout: int = 55) -> None:
        """Reload all Volumes mounted in the Sandbox.

        Added in v1.1.0.

        Blocks until the Volumes have been reloaded, bounded by `timeout` (55 seconds by default). If the reload
        does not complete within that window, `modal.exception.TimeoutError` is raised; note that the reload may
        still complete in the background.

        Args:
            timeout: Maximum time in seconds to wait for the reload. Must be positive.
        """
        if timeout <= 0:
            raise InvalidError("The `timeout` argument to `Sandbox.reload_volumes` must be positive.")
        task_id = await self._get_task_id()
        command_router_client = await self._get_command_router_client(task_id)
        await command_router_client.reload_volumes(task_id, timeout=float(timeout))

    @overload
    async def terminate(
        self,
        *,
        wait: Literal[True],
    ) -> int: ...

    @overload
    async def terminate(
        self,
        *,
        wait: Literal[False] = False,
    ) -> None: ...

    async def terminate(
        self,
        *,
        wait: bool = False,
    ) -> int | None:
        """Terminate Sandbox execution.

        This is a no-op if the Sandbox has already finished running.

        Args:
            wait: If True, block until termination completes and return the exit code.

        Returns:
            The sandbox exit code when `wait` is True; otherwise None.
        """
        req = api_pb2.SandboxTerminateRequest(sandbox_id=self.object_id)
        stub = self._client.stub
        if self._is_v2:
            assert self._client._auth_token_manager
            auth_token = await self._client._auth_token_manager.get_token()
            await stub.SandboxTerminateV2(req, metadata=[("x-modal-auth-token", auth_token)])
        else:
            await stub.SandboxTerminate(req)
        if wait:
            await self.wait(raise_on_termination=False)
            return self.returncode

    async def poll(self) -> int | None:
        """Check if the Sandbox has finished running.

        Returns:
            `None` if the Sandbox is still running, otherwise the exit code.
        """

        req = api_pb2.SandboxWaitRequest(sandbox_id=self.object_id, timeout=0)
        stub = self._client.stub
        if self._is_v2:
            assert self._client._auth_token_manager
            auth_token = await self._client._auth_token_manager.get_token()
            resp = await stub.SandboxWaitV2(req, metadata=[("x-modal-auth-token", auth_token)])
        else:
            resp = await stub.SandboxWait(req)

        if resp.result.status:
            self._result = resp.result

        return self.returncode

    async def _get_task_id(self, raise_if_task_complete=False) -> str:
        while not self._task_id:
            req = api_pb2.SandboxGetTaskIdRequest(sandbox_id=self.object_id)
            stub = self._client.stub
            if self._is_v2:
                assert self._client._auth_token_manager
                auth_token = await self._client._auth_token_manager.get_token()
                resp = await stub.SandboxGetTaskIdV2(req, metadata=[("x-modal-auth-token", auth_token)])
            else:
                resp = await stub.SandboxGetTaskId(req)
            if not resp.task_id and raise_if_task_complete and resp.HasField("task_result"):
                msg = resp.task_result.exception or "Sandbox already finished"
                raise ConflictError(msg)
            self._task_id = resp.task_id
            if not self._task_id:
                await asyncio.sleep(0.5)
        return self._task_id

    async def _get_command_router_client(self, task_id: str) -> TaskCommandRouterClient:
        if self._command_router_client is None:
            try:
                if self._is_v2:
                    self._command_router_client = await TaskCommandRouterClient.init_v2(
                        self._client, self.object_id, task_id
                    )
                else:
                    self._command_router_client = await TaskCommandRouterClient.init(self._client, task_id)
            except ConflictError as e:
                raise NotFoundError(str(e)) from e
        return self._command_router_client

    @property
    def _experimental_sidecars(self) -> "_SidecarManager":
        """Manage sidecar containers running in this Sandbox."""
        self._ensure_attached()
        return _SidecarManager(self)

    @overload
    async def exec(
        self,
        *args: str,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        text: Literal[True] = True,
        bufsize: Literal[-1, 1] = -1,
        pty: bool = False,
        pty_info: api_pb2.PTYInfo | None = None,
        _pty_info: api_pb2.PTYInfo | None = None,
    ) -> _ContainerProcess[str]: ...

    @overload
    async def exec(
        self,
        *args: str,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        text: Literal[False] = False,
        bufsize: Literal[-1, 1] = -1,
        pty: bool = False,
        pty_info: api_pb2.PTYInfo | None = None,
        _pty_info: api_pb2.PTYInfo | None = None,
    ) -> _ContainerProcess[bytes]: ...

    async def exec(
        self,
        *args: str,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        text: bool = True,
        bufsize: Literal[-1, 1] = -1,
        pty: bool = False,
        _pty_info: api_pb2.PTYInfo | None = None,  # *DEPRECATED*
        pty_info: api_pb2.PTYInfo | None = None,  # *DEPRECATED*
    ):
        """Execute a command in the Sandbox and return a ContainerProcess handle.

        See the [`ContainerProcess`](https://modal.com/docs/sdk/py/latest/modal.container_process#modalcontainer_processcontainerprocess)
        docs for more information.

        Args:
            *args: Command and arguments to run inside the sandbox.
            stdout: Where to connect the process stdout stream.
            stderr: Where to connect the process stderr stream.
            timeout: Optional timeout in seconds for the exec session.
            workdir: Working directory for the command; must be absolute if set.
            env: Environment variables to set during command execution.
            secrets: Secrets to inject as environment variables during command execution.
            text: If True, decode streams as text; if False, yield bytes.
            bufsize:
                Control line-buffered output. ``-1`` means unbuffered; ``1`` means line-buffered (only when ``text`` is
                True).
            pty:
                Enable a PTY for the command. When enabled, all output (stdout and stderr from the process) is
                multiplexed into stdout, and the stderr stream is effectively empty.
            _pty_info: *DEPRECATED* Use `pty` instead. `pty` will override `_pty_info`.
            pty_info: *DEPRECATED* Use `pty` instead. `pty` will override `pty_info`.

        Returns:
            A `ContainerProcess` handle for the running command (text or bytes depending on `text`).

        Examples:
            ```python fixture:sandbox
            process = sandbox.exec("bash", "-c", "for i in $(seq 1 3); do echo foo $i; sleep 0.1; done")
            for line in process.stdout:
                print(line)
            ```
        """
        if pty_info is not None or _pty_info is not None:
            deprecation_warning(
                (2025, 9, 12),
                "The `_pty_info` and `pty_info` parameters are deprecated and will be removed in a future release. "
                "Set the `pty` parameter to `True` instead.",
            )
        pty_info = _pty_info or pty_info
        if pty:
            pty_info = self._default_pty_info()

        return await self._exec(
            *args,
            pty_info=pty_info,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            workdir=workdir,
            env=env,
            secrets=secrets,
            text=text,
            bufsize=bufsize,
        )

    async def _exec(
        self,
        *args: str,
        pty_info: api_pb2.PTYInfo | None = None,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        text: bool = True,
        bufsize: Literal[-1, 1] = -1,
        container_id: str | None = None,
    ) -> _ContainerProcess[bytes] | _ContainerProcess[str]:
        """Private method used internally.

        This method exposes some internal arguments (currently `pty_info`) which are not in the public API.
        """
        if workdir is not None and not workdir.startswith("/"):
            raise InvalidError(f"workdir must be an absolute path, got: {workdir}")
        _validate_exec_args(args)

        secrets = list(secrets or [])
        env_dict, resolvable_secrets = _split_env_dict_and_resolvable_secrets(secrets)
        env_dict |= {k: v for k, v in (env or {}).items() if v is not None}

        # Force explicit secret resolution so we can pass the secret IDs to the backend.
        secret_coros = [secret.hydrate(client=self._client) for secret in resolvable_secrets]
        await TaskContext.gather(*secret_coros)

        task_id = await self._get_task_id(raise_if_task_complete=True)

        # NB: This must come after the task ID is set, since the sandbox must be
        # scheduled before we can create a router client.
        command_router_client = await self._get_command_router_client(task_id)

        return await self._exec_through_command_router(
            *args,
            task_id=task_id,
            command_router_client=command_router_client,
            pty_info=pty_info,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            workdir=workdir,
            secret_ids=[secret.object_id for secret in resolvable_secrets],
            env=env_dict,
            text=text,
            bufsize=bufsize,
            runtime_debug=config.get("function_runtime_debug"),
            container_id=container_id,
        )

    async def _exec_through_command_router(
        self,
        *args: str,
        task_id: str,
        command_router_client: TaskCommandRouterClient,
        pty_info: api_pb2.PTYInfo | None = None,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: int | None = None,
        workdir: str | None = None,
        secret_ids: Collection[str] | None = None,
        env: dict[str, str] | None = None,
        text: bool = True,
        bufsize: Literal[-1, 1] = -1,
        runtime_debug: bool = False,
        container_id: str | None = None,
    ) -> _ContainerProcess[bytes] | _ContainerProcess[str]:
        """Execute a command through a task command router running on the Modal worker."""

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
            secret_ids=secret_ids,
            pty_info=pty_info,
            runtime_debug=runtime_debug,
            container_id=container_id or "",
            env=env or {},
        )
        _ = await command_router_client.exec_start(start_req)

        return _ContainerProcess(
            process_id,
            task_id,
            self._client,
            command_router_client=command_router_client,
            stdout=stdout,
            stderr=stderr,
            text=text,
            by_line=bufsize == 1,
            exec_deadline=time.monotonic() + int(timeout) if timeout else None,
        )

    async def _experimental_snapshot(self) -> _SandboxSnapshot:
        self._ensure_v1("_experimental_snapshot")
        await self._get_task_id()
        snap_req = api_pb2.SandboxSnapshotRequest(sandbox_id=self.object_id)
        snap_resp = await self._client.stub.SandboxSnapshot(snap_req)

        snapshot_id = snap_resp.snapshot_id

        # wait for the snapshot to succeed. this is implemented as a second idempotent rpc
        # because the snapshot itself may take a while to complete.
        wait_req = api_pb2.SandboxSnapshotWaitRequest(snapshot_id=snapshot_id, timeout=55.0)
        wait_resp = await self._client.stub.SandboxSnapshotWait(wait_req)
        if wait_resp.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
            raise ExecutionError(wait_resp.result.exception)

        async def _load(
            self: _SandboxSnapshot, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None
        ):
            # we eagerly hydrate the sandbox snapshot below
            pass

        rep = "SandboxSnapshot()"
        # TODO: use ._new_hydrated instead
        obj = _SandboxSnapshot._from_loader(_load, rep, hydrate_lazily=True, load_context_overrides=LoadContext.empty())
        obj._hydrate(snapshot_id, self._client, None)

        return obj

    @staticmethod
    async def _experimental_from_snapshot(
        snapshot: _SandboxSnapshot,
        client: _Client | None = None,
        *,
        name: str | None = _DEFAULT_SANDBOX_NAME_OVERRIDE,
    ):
        client = client or await _Client.from_env()

        if name is not None and name != _DEFAULT_SANDBOX_NAME_OVERRIDE:
            check_object_name(name, "Sandbox")

        if name is _DEFAULT_SANDBOX_NAME_OVERRIDE:
            restore_req = api_pb2.SandboxRestoreRequest(
                snapshot_id=snapshot.object_id,
                sandbox_name_override_type=api_pb2.SandboxRestoreRequest.SANDBOX_NAME_OVERRIDE_TYPE_UNSPECIFIED,
            )
        elif name is None:
            restore_req = api_pb2.SandboxRestoreRequest(
                snapshot_id=snapshot.object_id,
                sandbox_name_override_type=api_pb2.SandboxRestoreRequest.SANDBOX_NAME_OVERRIDE_TYPE_NONE,
            )
        else:
            restore_req = api_pb2.SandboxRestoreRequest(
                snapshot_id=snapshot.object_id,
                sandbox_name_override=name,
                sandbox_name_override_type=api_pb2.SandboxRestoreRequest.SANDBOX_NAME_OVERRIDE_TYPE_STRING,
            )
        # Pin the restored Sandbox to a specific worker when MODAL_WORKER_ID is
        # set.
        if worker_id := config.get("worker_id"):
            restore_req.worker_id = worker_id
        restore_resp: api_pb2.SandboxRestoreResponse = await client.stub.SandboxRestore(restore_req)

        sandbox = await _Sandbox.from_id(restore_resp.sandbox_id, client)

        task_id_req = api_pb2.SandboxGetTaskIdRequest(
            sandbox_id=restore_resp.sandbox_id, wait_until_ready=True, timeout=55.0
        )
        resp = await client.stub.SandboxGetTaskId(task_id_req)
        if resp.task_result.status not in [
            api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED,
            api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
        ]:
            raise ExecutionError(resp.task_result.exception)
        return sandbox

    @property
    def filesystem(self) -> _SandboxFilesystem:
        """Namespace for filesystem APIs.

        Returns:
            A `SandboxFilesystem` helper bound to this sandbox.
        """
        self._ensure_attached()
        if self._filesystem is None:
            self._filesystem = _SandboxFilesystem(self)
        return self._filesystem

    @overload
    async def open(
        self,
        path: str,
    ) -> _FileIO[str]: ...

    @overload
    async def open(
        self,
        path: str,
        mode: "_typeshed.OpenTextMode",
    ) -> _FileIO[str]: ...

    @overload
    async def open(
        self,
        path: str,
        mode: "_typeshed.OpenBinaryMode",
    ) -> _FileIO[bytes]: ...

    async def open(
        self,
        path: str,
        mode: Union["_typeshed.OpenTextMode", "_typeshed.OpenBinaryMode"] = "r",
    ):
        """[Alpha] Open a file in the Sandbox and return a FileIO handle.

        **Deprecated (2026-03-09):** Use the `Sandbox.filesystem` APIs instead for improved reliability.

        See the [`FileIO`](https://modal.com/docs/sdk/py/latest/modal.file_io#modalfile_iofileio)
        docs for more information.

        Args:
            path: Absolute path of the file inside the sandbox.
            mode: File open mode (text or binary), following built-in ``open`` conventions.

        Returns:
            A `FileIO` handle for reading or writing the remote file.

        Examples:
            ```python notest
            sb = modal.Sandbox.create(app=sb_app)
            f = sb.open("/test.txt", "w")
            f.write("hello")
            f.close()
            ```
        """
        self._ensure_v1("open")
        deprecation_warning(
            (2026, 3, 9),
            "`Sandbox.open()` is deprecated. Use the `Sandbox.filesystem` APIs instead for improved reliability.",
        )
        task_id = await self._get_task_id()
        return await _FileIO.create(path, mode, self._client, task_id)

    async def ls(self, path: str) -> builtins.list[str]:
        """[Alpha] List the contents of a directory in the Sandbox.

        **Deprecated (2026-04-15):** Use `Sandbox.filesystem.list_files()` instead for improved reliability.

        Args:
            path: Absolute directory path inside the sandbox.

        Returns:
            Entry names in the directory as a list of strings.
        """
        self._ensure_v1("ls")
        deprecation_warning(
            (2026, 4, 15),
            "`Sandbox.ls()` is deprecated. Use `Sandbox.filesystem.list_files()` instead for improved reliability.",
        )
        task_id = await self._get_task_id()
        return await ls(path, self._client, task_id)

    async def mkdir(self, path: str, parents: bool = False) -> None:
        """[Alpha] Create a new directory in the Sandbox.

        **Deprecated (2026-04-15):** Use `Sandbox.filesystem.make_directory()` instead for improved reliability.
        """
        self._ensure_v1("mkdir")
        deprecation_warning(
            (2026, 4, 15),
            "`Sandbox.mkdir()` is deprecated. Use `Sandbox.filesystem.make_directory()` instead for improved "
            "reliability.",
        )
        task_id = await self._get_task_id()
        return await mkdir(path, self._client, task_id, parents)

    async def rm(self, path: str, recursive: bool = False) -> None:
        """[Alpha] Remove a file or directory in the Sandbox.

        **Deprecated (2026-04-15):** Use `Sandbox.filesystem.remove()` instead for improved reliability.
        """
        self._ensure_v1("rm")
        deprecation_warning(
            (2026, 4, 15),
            "`Sandbox.rm()` is deprecated. Use `Sandbox.filesystem.remove()` instead for improved reliability.",
        )
        task_id = await self._get_task_id()
        return await rm(path, self._client, task_id, recursive)

    async def watch(
        self,
        path: str,
        filter: builtins.list[FileWatchEventType] | None = None,
        recursive: bool | None = None,
        timeout: int | None = None,
    ) -> AsyncIterator[FileWatchEvent]:
        """[Alpha] Watch a file or directory in the Sandbox for changes.

        **Deprecated (2026-05-08):** Use `Sandbox.filesystem.watch()` instead for improved reliability.

        Args:
            path: Absolute path to watch.
            filter: Optional list of event types to include.
            recursive: Whether to watch subdirectories; None uses server defaults.
            timeout: Optional timeout for the watch stream.

        Returns:
            An async iterator of `FileWatchEvent` values.
        """
        self._ensure_v1("watch")
        deprecation_warning(
            (2026, 5, 8),
            "`Sandbox.watch()` is deprecated. Use `Sandbox.filesystem.watch()` instead for improved reliability.",
        )
        task_id = await self._get_task_id()
        async for event in watch(path, self._client, task_id, filter, recursive, timeout):
            yield event

    @property
    def stdout(self) -> _StreamReader[str]:
        """
        [`StreamReader`](https://modal.com/docs/sdk/py/latest/modal.io_streams#modalio_streamsstreamreader)
        for the sandbox's stdout stream.

        Returns:
            Stream reader for sandbox stdout.
        """
        self._ensure_attached()
        return self._stdout

    @property
    def stderr(self) -> _StreamReader[str]:
        """
        [`StreamReader`](https://modal.com/docs/sdk/py/latest/modal.io_streams#modalio_streamsstreamreader)
        for the Sandbox's stderr stream.

        Returns:
            Stream reader for sandbox stderr.
        """
        self._ensure_attached()
        return self._stderr

    @property
    def stdin(self) -> _StreamWriter:
        """
        [`StreamWriter`](https://modal.com/docs/sdk/py/latest/modal.io_streams#modalio_streamsstreamwriter)
        for the Sandbox's stdin stream.

        Returns:
            Stream writer for sandbox stdin.
        """
        self._ensure_attached()
        return self._stdin

    @property
    def returncode(self) -> int | None:
        """Return code of the Sandbox process if it has finished running, else `None`.

        Returns:
            Exit code when the sandbox process has completed, otherwise None.
        """
        return _result_returncode(self._result)

    @staticmethod
    async def list(
        *, app_id: str | None = None, tags: dict[str, str] | None = None, client: _Client | None = None
    ) -> AsyncGenerator["_Sandbox", None]:
        """List all Sandboxes for the current Environment or App ID (if specified). If tags are specified, only
        Sandboxes that have at least those tags are returned.

        Args:
            app_id: If set, restrict results to sandboxes under this app ID.
            tags: If set, only sandboxes containing at least these tags are returned.
            client: Modal client to use for listing; defaults to `Client.from_env()` when omitted.

        Returns:
            An async generator yielding `Sandbox` objects.
        """
        before_timestamp = None
        environment_name = _get_environment_name()
        if client is None:
            client = await _Client.from_env()

        tags_list = [api_pb2.SandboxTag(tag_name=name, tag_value=value) for name, value in tags.items()] if tags else []

        while True:
            req = api_pb2.SandboxListRequest(
                app_id=app_id,
                before_timestamp=before_timestamp,
                environment_name=environment_name,
                include_finished=False,
                tags=tags_list,
            )

            # Fetches a batch of sandboxes.
            resp = await client.stub.SandboxList(req)

            if not resp.sandboxes:
                return

            for sandbox_info in resp.sandboxes:
                sandbox_info: api_pb2.SandboxInfo
                obj = _Sandbox._new_hydrated(sandbox_info.id, client, None)
                obj._result = sandbox_info.task_info.result  # TODO: send SandboxInfo as metadata to _new_hydrated?
                yield obj

            # Fetch the next batch starting from the end of the current one.
            before_timestamp = resp.sandboxes[-1].created_at

    @staticmethod
    async def _experimental_list(
        *, app_id: str | None = None, tags: dict[str, str] | None = None, client: _Client | None = None
    ) -> AsyncGenerator["_Sandbox", None]:
        """List v2 Sandboxes in an App.

        This function lists v2 sandboxes, ie sandboxes created via modal.Sandbox._experimental_create.

        Args:
            app_id: The App to list Sandboxes under.
            tags: If set, only sandboxes containing at least these tags are returned.
            client: Optional client to use for the session.

        Yields:
            `Sandbox` objects that are currently running in the App.
        """
        if not app_id:
            raise InvalidError(
                "Sandbox._experimental_list requires an `app_id`:\n\n"
                'app = modal.App.lookup("my-app")\n'
                "Sandbox._experimental_list(app_id=app.app_id)"
            )

        before_timestamp = None
        if client is None:
            client = await _Client.from_env()

        tags_list = [api_pb2.SandboxTag(tag_name=name, tag_value=value) for name, value in tags.items()] if tags else []

        assert client._auth_token_manager
        while True:
            req = api_pb2.SandboxListRequest(
                app_id=app_id,
                before_timestamp=before_timestamp,
                include_finished=False,
                tags=tags_list,
            )

            # Fetches a batch of sandboxes. SandboxListV2 authenticates via the
            # auth-token metadata, like the other V2 sandbox RPCs.
            auth_token = await client._auth_token_manager.get_token()
            resp = await client.stub.SandboxListV2(req, metadata=[("x-modal-auth-token", auth_token)])

            if not resp.sandboxes:
                return

            for sandbox_info in resp.sandboxes:
                sandbox_info: api_pb2.SandboxInfo
                obj = _Sandbox._new_hydrated(sandbox_info.id, client, None)
                # SandboxListV2 only returns V2 sandboxes; mark them as such so
                # operations like wait/terminate/exec use the V2 RPCs and stdio.
                obj._is_v2 = True
                obj._hydrate_metadata_v2()
                obj._result = sandbox_info.task_info.result
                yield obj

            # Fetch the next batch starting from the end of the current one.
            before_timestamp = resp.sandboxes[-1].created_at


class _SidecarContainer:
    """Handle to an additional container running in a Sandbox."""

    _result: api_pb2.GenericResult | None
    _filesystem: _SandboxFilesystem | None

    def __init__(
        self,
        sandbox: _Sandbox,
        container_id: str,
        container_name: str,
        result: api_pb2.GenericResult | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._container_id = container_id
        self._container_name = container_name
        self._result = result
        self._filesystem = None

    @property
    def object_id(self) -> str:
        return self._container_id

    @property
    def name(self) -> str:
        return self._container_name

    @staticmethod
    def _from_container_info(sandbox: "_Sandbox", container_info: sr_pb2.TaskContainerInfo) -> "_SidecarContainer":
        result = container_info.result if container_info.HasField("result") else None
        return _SidecarContainer(sandbox, container_info.container_id, container_info.container_name, result)

    async def _get_command_router(self) -> tuple[str, "TaskCommandRouterClient"]:
        """Get task ID and command router client."""
        task_id = await self._sandbox._get_task_id()
        command_router_client = await self._sandbox._get_command_router_client(task_id)
        return task_id, command_router_client

    @typing.overload
    async def exec(
        self,
        *args: str,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        text: Literal[True] = True,
        bufsize: Literal[-1, 1] = -1,
        # Enable a PTY for the command. When enabled, all output (stdout and stderr from the
        # process) is multiplexed into stdout, and the stderr stream is effectively empty.
        pty: bool = False,
    ) -> _ContainerProcess[str]: ...

    @typing.overload
    async def exec(
        self,
        *args: str,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        text: Literal[False],
        bufsize: Literal[-1, 1] = -1,
        # Enable a PTY for the command. When enabled, all output (stdout and stderr from the
        # process) is multiplexed into stdout, and the stderr stream is effectively empty.
        pty: bool = False,
    ) -> _ContainerProcess[bytes]: ...

    async def exec(
        self,
        *args: str,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        text: bool = True,
        bufsize: Literal[-1, 1] = -1,
        # Enable a PTY for the command. When enabled, all output (stdout and stderr from the
        # process) is multiplexed into stdout, and the stderr stream is effectively empty.
        pty: bool = False,
    ) -> _ContainerProcess[bytes] | _ContainerProcess[str]:
        pty_info = self._sandbox._default_pty_info() if pty else None
        return await self._sandbox._exec(
            *args,
            pty_info=pty_info,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            workdir=workdir,
            env=env,
            secrets=secrets,
            text=text,
            bufsize=bufsize,
            container_id=self._container_id,
        )

    @property
    def filesystem(self) -> _SandboxFilesystem:
        """Namespace for filesystem APIs."""
        if self._filesystem is None:
            self._filesystem = _SandboxFilesystem(self)
        return self._filesystem

    async def wait(self, raise_on_termination: bool = True) -> None:
        if self._result is not None and self._result.status != api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED:
            if self._result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED and raise_on_termination:
                raise SandboxTerminatedError()
            return

        task_id, command_router_client = await self._get_command_router()
        while True:
            resp = await command_router_client.container_wait(
                sr_pb2.TaskContainerWaitRequest(
                    task_id=task_id,
                    container_id=self._container_id,
                    timeout=10,
                )
            )
            if resp.result.status:
                self._result = resp.result
                if resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED and raise_on_termination:
                    raise SandboxTerminatedError()
                return

    async def poll(self) -> int | None:
        if self._result is not None and self._result.status != api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED:
            return _result_returncode(self._result)

        task_id, command_router_client = await self._get_command_router()
        resp = await command_router_client.container_wait(
            sr_pb2.TaskContainerWaitRequest(
                task_id=task_id,
                container_id=self._container_id,
                timeout=0,
            )
        )
        if resp.result.status:
            self._result = resp.result
        return _result_returncode(self._result)

    @overload
    async def terminate(
        self,
        *,
        wait: Literal[True],
    ) -> int: ...

    @overload
    async def terminate(
        self,
        *,
        wait: Literal[False] = False,
    ) -> None: ...

    async def terminate(
        self,
        *,
        wait: bool = False,
    ) -> int | None:
        task_id, command_router_client = await self._get_command_router()
        await command_router_client.container_terminate(
            sr_pb2.TaskContainerTerminateRequest(
                task_id=task_id,
                container_id=self._container_id,
            )
        )
        if wait:
            await self.wait(raise_on_termination=False)
            return _result_returncode(self._result)


_MAIN_CONTAINER_NAME: str = "main"


class _SidecarManager:
    """Creates and manages sidecar containers in a Sandbox."""

    def __init__(self, sandbox: _Sandbox) -> None:
        self._sandbox = sandbox

    async def _get_command_router(self) -> tuple[str, "TaskCommandRouterClient"]:
        """Get task ID and command router client."""
        task_id = await self._sandbox._get_task_id()
        command_router_client = await self._sandbox._get_command_router_client(task_id)
        return task_id, command_router_client

    async def create(
        self,
        *args: str,
        name: str,
        image: _Image,
        env: dict[str, str] | None = None,
        secrets: Collection[_Secret] | None = None,
        workdir: str | None = None,
        volumes: dict[str | os.PathLike, _Volume] | None = None,
    ) -> _SidecarContainer:
        """Create a sidecar container running alongside the Sandbox's main container.

        Sidecar containers share the Sandbox's lifecycle but run their own Image and command. They
        can be used to run auxiliary processes, such as a database or a service the main container
        depends on.

        Args:
            *args: Command and arguments to run inside the sidecar container.
            name: Unique name for the sidecar container. The name ``"main"`` is reserved.
            image: Image to run the sidecar container with. Must be a pre-built or referenced Image.
            env: Environment variables to set in the sidecar container.
            secrets: Secrets to inject as environment variables in the sidecar container.
            workdir: Working directory for the command; must be absolute if set.
            volumes: Mapping of mount paths to `Volume` objects to mount in the sidecar container.

        Returns:
            A `SidecarContainer` handle for the running container.
        """
        if name == _MAIN_CONTAINER_NAME:
            raise InvalidError(f"The name {_MAIN_CONTAINER_NAME!r} is reserved for the sandbox's main container.")
        if workdir is not None and not workdir.startswith("/"):
            raise InvalidError(f"workdir must be an absolute path, got: {workdir}")
        _validate_exec_args(args)

        validated_volumes = validate_only_modal_volumes(volumes, "Sandbox._experimental_sidecars.create(volumes=...)")

        if image._mount_layers:
            raise InvalidError(
                "Sandbox._experimental_sidecars.create(image=...) only supports pre-built images. "
                "When using `add_local*` methods, specify `copy=True` and call `.build()` before passing "
                "the image to `._experimental_sidecars.create()`:\n\nE.g.\n"
                'img = modal.Image.debian_slim().add_local_file("foo", "/foo", copy=True).build(app)\n'
                'sandbox._experimental_sidecars.create(name="worker", image=img)'
            )
        if not image._object_id:
            raise InvalidError(
                "Sandbox._experimental_sidecars.create(image=...) currently only supports Images that are "
                "either:\n"
                "- prebuilt using `image.build()`\n"
                "- referenced by id, e.g. `Image.from_id()`\n"
                "- filesystem/directory snapshots e.g. created by `.snapshot_directory()` "
                "or `.snapshot_filesystem()`\n"
            )

        secrets = secrets or []
        hydrate_coros = [secret.hydrate(client=self._sandbox._client) for secret in secrets] + [
            volume.hydrate(client=self._sandbox._client) for _, volume in validated_volumes
        ]
        await TaskContext.gather(*hydrate_coros)

        # Validate that the same volume (by object_id) isn't mounted at multiple paths. This relies on
        # the volumes being hydrated above, since it compares object_ids.
        validate_volumes_by_object_id(validated_volumes)

        # Relies on dicts being ordered (true as of Python 3.6).
        volume_mounts = [_volume_to_mount_proto(path, volume) for path, volume in validated_volumes]

        task_id, command_router_client = await self._get_command_router()

        create_req = sr_pb2.TaskContainerCreateRequest(
            task_id=task_id,
            container_name=name,
            image_id=image.object_id,
            args=list(args),
            env=env or {},
            workdir=workdir or "",
            secret_ids=[secret.object_id for secret in secrets],
            volume_mounts=volume_mounts,
        )
        create_resp = await command_router_client.container_create(create_req)
        container_id = create_resp.container_id
        container_name = create_resp.container_name or name
        return _SidecarContainer(self._sandbox, container_id, container_name)

    async def get(self, *, name: str, include_terminated: bool = False) -> "_SidecarContainer":
        if name == _MAIN_CONTAINER_NAME:
            raise InvalidError(
                "Cannot get the main sandbox container through the sidecars API. "
                "Use Sandbox methods directly to interact with the main container."
            )
        task_id, command_router_client = await self._get_command_router()
        resp = await command_router_client.container_get(
            sr_pb2.TaskContainerGetRequest(
                task_id=task_id,
                container_name=name,
                include_terminated=include_terminated,
            )
        )
        return _SidecarContainer._from_container_info(self._sandbox, resp.container)

    async def list(self, include_terminated: bool = False) -> builtins.list[_SidecarContainer]:
        task_id, command_router_client = await self._get_command_router()
        resp = await command_router_client.container_list(
            sr_pb2.TaskContainerListRequest(task_id=task_id, include_terminated=include_terminated)
        )
        return [
            _SidecarContainer._from_container_info(self._sandbox, container)
            for container in resp.containers
            if container.container_name != _MAIN_CONTAINER_NAME
        ]


SidecarContainer = synchronize_api(_SidecarContainer)
SidecarManager = synchronize_api(_SidecarManager)
Sandbox = synchronize_api(_Sandbox)
