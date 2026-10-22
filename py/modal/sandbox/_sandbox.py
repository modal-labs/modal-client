# Copyright Modal Labs 2022
import json
import os
import uuid
from collections.abc import AsyncGenerator, Collection, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, overload

from google.protobuf.message import Message

from modal._logs_manager import _SandboxLogsManager
from modal._supports_logs import LogsFilters, _LogQueryData
from modal._tunnel import Tunnel
from modal.mount import _Mount
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._image import _Image
from .._object import _get_environment_name, _Object
from .._outbound_policy import _OutboundPolicy
from .._output.pty import get_pty_info
from .._utils.deprecation import deprecation_warning
from .._utils.name_utils import check_object_name
from ..client import _Client
from ..cloud_bucket_mount import _CloudBucketMount
from ..config import config, logger as logger
from ..container_process import _ContainerProcess
from ..exception import ClientClosed, ExecutionError, InvalidError, NotFoundError
from ..io_streams import _StreamReader, _StreamWriter
from ..network_file_system import _NetworkFileSystem
from ..proxy import _Proxy
from ..secret import _Secret
from ..stream_type import StreamType
from ..types import SandboxConnectCredentials, SandboxRuntime
from ..volume import _Volume
from ._common import (
    SandboxCreateArgs,
    SandboxImpl,
    SandboxVersion as SandboxVersion,
    _default_image as _default_image,
    _format_sandbox_create_timing_log as _format_sandbox_create_timing_log,
    _get_sandbox_version as _get_sandbox_version,
    _image_id_for_mount,
    _is_v1_sandbox_id as _is_v1_sandbox_id,
    _result_returncode,
    default_pty_info,
)
from ._filesystem import _SandboxFilesystem
from ._impl_persistent import SandboxImplPersistent
from ._impl_v1 import SandboxImplV1, command_router_access_fetcher as v1_command_router_access_fetcher
from ._impl_v2 import SandboxImplV2, command_router_access_fetcher as v2_command_router_access_fetcher
from ._sidecar import _SidecarManager
from ._snapshot import _SandboxSnapshot
from ._task_command_router_client import TaskCommandRouterClient, _is_v2_task_id

if TYPE_CHECKING:
    import modal.app


# How long `Sandbox.create` waits for capacity before giving up.
_SANDBOX_SCHEDULING_TIMEOUT: float = 21 * 60


def impl_class_for_sandbox_id(sandbox_id: str) -> type[SandboxImpl]:
    """Select the implementation backing the Sandbox with the given ID."""
    if _get_sandbox_version(sandbox_id) == SandboxVersion.V1:
        return SandboxImplV1
    return SandboxImplV2


def impl_class_for_create_args(create_args: SandboxCreateArgs) -> type[SandboxImpl]:
    """Select the implementation that creates the Sandbox described by `create_args`."""
    if create_args.persistent:
        return SandboxImplPersistent
    if create_args.gpu is not None or create_args.network_file_systems or create_args.pty_info is not None:
        # GPUs, network file systems, and PTYs are not supported on V2, so those calls use the V1
        # backend regardless of the setting.
        return SandboxImplV1
    return SandboxImplV2 if config.get("sandbox_v2") else SandboxImplV1


class DefaultSandboxNameOverride(str):
    """A singleton class that represents the default sandbox name override.

    It is used to indicate that the sandbox name should not be overridden.
    """

    def __repr__(self) -> str:
        # NOTE: this must match the qualified instance var name below in order for type stubs to work 😬
        return "modal.sandbox._sandbox._DEFAULT_SANDBOX_NAME_OVERRIDE"


_DEFAULT_SANDBOX_NAME_OVERRIDE: DefaultSandboxNameOverride = DefaultSandboxNameOverride()


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

    # Backend-specific behavior and connection state, selected from the Sandbox type.
    _impl: SandboxImpl
    _stdout: _StreamReader[str]
    _stderr: _StreamReader[str]
    _stdin: _StreamWriter
    _attached: bool
    _filesystem: _SandboxFilesystem | None
    _app_id: str | None

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
        timeout: int | None = 300,
        idle_timeout: int | None = None,
        workdir: str | None = None,
        gpu: str | None = None,
        cloud: str | None = None,
        region: str | Sequence[str] | None = None,
        cpu: float | tuple[float, float] | None = None,
        memory: int | tuple[int, int] | None = None,
        runtime: SandboxRuntime | None = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        _experimental_outbound_policy: _OutboundPolicy | None = None,
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
        _experimental_persistent: bool = False,
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
            runtime:
                Runtime under which the Sandbox executes, or None to let Modal pick.
            block_network: Whether to block network access.
            outbound_cidr_allowlist: List of CIDRs the sandbox is allowed to access. If None, all CIDRs are allowed.
            outbound_domain_allowlist: List of domain names the sandbox is allowed to access. Supports
                wildcard prefixes (``*.``); a bare ``"*"`` allows all domains. The outbound policy
                can be replaced later via `Sandbox._experimental_set_outbound_network_policy`.
            inbound_cidr_allowlist:
                List of CIDRs allowed to connect inbound to the sandbox (encrypted tunnels and connection tokens).
                If None, all CIDRs are allowed.
            _experimental_outbound_policy: Configuration for replacing headers in outbound HTTPS requests from
                the Sandbox. Secrets referenced by the policy are resolved outside the Sandbox and are never
                visible to the workload. See `modal.experimental.OutboundPolicy`. This API is experimental and
                may change in the future.
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

        create_args = SandboxCreateArgs(
            args=args,
            image=image,
            secrets=secrets,
            env=env,
            name=name,
            timeout=timeout,
            idle_timeout=idle_timeout,
            workdir=workdir,
            gpu=gpu,
            cloud=cloud,
            region=region,
            cpu=cpu,
            memory=memory,
            network_file_systems=network_file_systems,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            outbound_policy=_experimental_outbound_policy,
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
            runtime=runtime,
            persistent=_experimental_persistent,
        )
        impl, metadata = await impl_class_for_create_args(create_args).create(create_args, app=app, client=client)
        return _Sandbox._from_impl(impl, metadata)

    @staticmethod
    async def _experimental_create(
        *args: str,
        app: "modal.app._App | None" = None,
        name: str | None = None,
        tags: dict[str, str] | None = None,
        image: _Image | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        mounts: Sequence[_Mount] = (),
        timeout: int = 300,
        idle_timeout: int | None = None,
        workdir: str | None = None,
        cpu: float | tuple[float, float] | None = None,
        memory: int | tuple[int, int] | None = None,
        runtime: SandboxRuntime | None = None,
        cloud: str | None = None,
        region: str | Sequence[str] | None = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        _experimental_outbound_policy: _OutboundPolicy | None = None,
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
        custom_domain: str | None = None,
        client: _Client | None = None,
        _experimental_enable_snapshot: bool = False,
    ) -> "_Sandbox":
        """Create a sandbox using the V2 backend.

        Supported features include exec, encrypted tunnels, wait/poll/terminate,
        CPU and memory configuration, region placement, private IPv6 networking
        (i6pn), volumes, cloud bucket mounts (with static credentials via
        `secret=...` or `oidc_auth_role_arn`), OIDC identity tokens, proxies,
        filesystem snapshots, and custom domains (`custom_domain=...` allows
        connections to the sandbox via a subdomain of that parent domain rather
        than a default Modal domain; requires prior setup by Modal).

        `cpu` and `memory` accept either a scalar request or a `(request, limit)`
        tuple that additionally sets a hard limit (fractional CPU cores; memory
        in MiB), matching `Sandbox.create()`.

        Pass `_experimental_enable_snapshot=True` to create a sandbox that can be
        snapshotted with `._experimental_snapshot()`. Features like network
        file systems and GPUs are not supported.

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
        create_args = SandboxCreateArgs(
            args=args,
            image=image,
            secrets=secrets,
            env=env,
            name=name,
            timeout=timeout,
            idle_timeout=idle_timeout,
            workdir=workdir,
            cloud=cloud,
            region=region,
            cpu=cpu,
            memory=memory,
            mounts=mounts,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            outbound_policy=_experimental_outbound_policy,
            inbound_cidr_allowlist=inbound_cidr_allowlist,
            i6pn=i6pn,
            volumes=volumes,
            pty=pty,
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
            runtime=runtime,
        )
        impl, metadata = await SandboxImplV2.create(create_args, app=app, client=client)
        return _Sandbox._from_impl(impl, metadata)

    def _get_metadata(self) -> api_pb2.SandboxHandleMetadata:
        metadata = api_pb2.SandboxHandleMetadata(app_id=self._app_id or "")
        if self._impl.result is not None:
            metadata.result.CopyFrom(self._impl.result)
        return metadata

    def _hydrate_metadata(self, handle_metadata: Message | None) -> None:
        self._app_id = None
        result = None
        if handle_metadata is not None:
            assert isinstance(handle_metadata, api_pb2.SandboxHandleMetadata), (
                f"{type(handle_metadata)} is not SandboxHandleMetadata"
            )
            self._app_id = handle_metadata.app_id
            if handle_metadata.HasField("result"):
                result = handle_metadata.result
        self._filesystem = None
        impl_cls = impl_class_for_sandbox_id(self.object_id)
        self._adopt_impl(impl_cls(self.client, self.object_id, result=result))

    def _adopt_impl(self, impl: SandboxImpl) -> None:
        """Back this Sandbox with `impl`, which owns all of its backend state."""
        self._impl = impl
        self._stdout, self._stderr, self._stdin = impl.make_stdio(self._ensure_attached)

    @staticmethod
    def _from_impl(impl: SandboxImpl, metadata: api_pb2.SandboxHandleMetadata | None) -> "_Sandbox":
        """Create a hydrated handle for the Sandbox that `impl` was constructed for."""
        obj = _Sandbox._new_hydrated(impl.sandbox_id, impl.client, metadata)
        obj._adopt_impl(impl)
        return obj

    def _initialize_from_other(self, other):
        super()._initialize_from_other(other)
        self._attached = other._attached
        self._app_id = other._app_id
        self._impl = other._impl.clone()

    def _initialize_from_empty(self):
        super()._initialize_from_empty()
        self._attached = True
        self._app_id = None

    async def detach(self):
        """Disconnects your client from the sandbox and cleans up resources associated with the connection.

        Be sure to only call `detach` when you are done interacting with the sandbox. After calling `detach`,
        any operation using the Sandbox object is not guaranteed to work anymore. If you want to continue interacting
        with a running sandbox, use `Sandbox.from_id` to get a new Sandbox object.

        This method does not interrupt or wait for running concurrent operations on the sandbox. Resources are
        promptly closed once those operations complete.

        """
        if not self._attached:
            return
        # Publish the detached state before requesting release of any resources.
        self._attached = False
        await self._impl.close()

    def _ensure_attached(self):
        """Raise `ClientClosed` if the Sandbox is detached.

        Every public operation calls this on entry, except those that remain available after
        `detach`: `wait`, `returncode`, `logs` and `_experimental_get_exit_snapshot`.
        """
        if not self._attached:
            raise ClientClosed("Unable to perform operation on a detached sandbox")

    @staticmethod
    async def from_name(
        app_name: str,
        name: str,
        *,
        environment_name: str | None = None,
        include_terminated: bool = False,
        client: _Client | None = None,
    ) -> "_Sandbox":
        """Get a running Sandbox by name from a deployed App.

        A Sandbox's name is the `name` argument passed to `Sandbox.create`.

        Args:
            app_name: Name of the deployed app to look up the sandbox under.
            name: Sandbox name to resolve.
            environment_name: Optional environment name for the lookup; defaults to the configured environment.
            include_terminated: If set to `True` and no Sandbox with the name is currently running, the most
                recent terminated Sandbox will be returned instead.
            client: Modal client to use for the RPC; defaults to `Client.from_env()` when omitted.

        Returns:
            A `Sandbox` handle for the running Sandbox, or the most recent terminated one when
            `include_terminated=True`.

        Raises:
            NotFoundError: If no matching sandbox exists with the given name.
        """
        if client is None:
            client = await _Client.from_env()

        if include_terminated or config.get("sandbox_v2") is True:
            try:
                return await _Sandbox._experimental_from_name(
                    app_name,
                    name,
                    environment_name=environment_name,
                    include_terminated=include_terminated,
                    client=client,
                )
            except NotFoundError:
                if include_terminated:
                    raise

        return await _Sandbox._from_name(SandboxImplV1, app_name, name, environment_name, client)

    @staticmethod
    async def _experimental_from_name(
        app_name: str,
        name: str,
        *,
        environment_name: str | None = None,
        include_terminated: bool = False,
        client: _Client | None = None,
    ) -> "_Sandbox":
        """Get a running V2 Sandbox by name from a deployed App.

        This looks up V2 sandboxes, ie sandboxes created via `modal.Sandbox._experimental_create`.

        Args:
            app_name: Name of the deployed app to look up the sandbox under.
            name: Sandbox name to resolve.
            environment_name: Optional environment name for the lookup; defaults to the configured environment.
            include_terminated: If set to `True` and no Sandbox with the name is currently running, the most
                recent terminated Sandbox will be returned instead.
            client: Modal client to use for the RPC; defaults to `Client.from_env()` when omitted.

        Returns:
            A `Sandbox` handle for the running Sandbox, or the most recent terminated one when
            `include_terminated=True`.

        Raises:
            NotFoundError: If no matching sandbox exists with the given name.
        """
        if client is None:
            client = await _Client.from_env()
        return await _Sandbox._from_name(
            SandboxImplV2, app_name, name, environment_name, client, include_terminated=include_terminated
        )

    @staticmethod
    async def _from_name(
        impl_cls: type[SandboxImpl],
        app_name: str,
        name: str,
        environment_name: str | None,
        client: _Client,
        *,
        include_terminated: bool = False,
    ) -> "_Sandbox":
        env_name = _get_environment_name(environment_name)
        req = api_pb2.SandboxGetFromNameRequest(
            sandbox_name=name,
            app_name=app_name,
            environment_name=env_name,
            include_terminated=include_terminated,
        )
        resp = await impl_cls.lookup_by_name(client, req)
        return _Sandbox._new_hydrated(resp.sandbox_id, client, resp.metadata)

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

        resp = await impl_class_for_sandbox_id(sandbox_id).lookup_by_id(client, sandbox_id)
        obj = _Sandbox._new_hydrated(sandbox_id, client, resp.metadata)
        if resp.result.status:
            obj._impl.result = resp.result

        return obj

    async def get_tags(self) -> dict[str, str]:
        """Fetches any tags (key-value pairs) currently attached to this Sandbox from the server.

        Returns:
            Tags as a map from tag name to tag value.
        """
        self._ensure_attached()
        return await self._impl.get_tags()

    async def set_tags(self, tags: dict[str, str], *, client: _Client | None = None) -> None:
        """Set tags (key-value pairs) on the Sandbox. Tags can be used to filter results in `Sandbox.list`.

        Setting tags replaces the Sandbox's entire tag set; passing an empty dict clears all tags.

        Args:
            tags: Tag names and values to set on this sandbox.
            client: Deprecated. Prefer setting the client when creating or re-attaching to the sandbox.

        """
        self._ensure_attached()
        if client is not None:
            deprecation_warning(
                (2025, 9, 18),
                "The `client` parameter is deprecated. Set `client` when creating the Sandbox instead "
                "(in e.g. `Sandbox.create()`/`.from_id()`/`.from_name()`).",
            )

        await self._impl.set_tags(tags)

    async def _experimental_set_name(self, name: str) -> None:
        """Assign a name to a running V2 Sandbox that was created without one.

        This is only supported for V2 sandboxes, ie sandboxes created via
        `modal.Sandbox._experimental_create`. A name may only be set once and
        only on a Sandbox that has never had one; afterwards the Sandbox can
        be looked up with `Sandbox._experimental_from_name(app_name, name)`.
        Repeating the Sandbox's current name is idempotent and succeeds, so
        the call is safe to retry.

        Args:
            name: Name to assign to the Sandbox. Must be unique within the App.

        Raises:
            AlreadyExistsError: If another running Sandbox in the App already holds the name.
            ConflictError: If the Sandbox already has a different name or is no longer running.
        """
        self._ensure_attached()
        await self._impl.set_name(name)

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
        self._ensure_attached()
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
        await self._impl.set_network_access(network_access)

    async def _experimental_update_outbound_policy(self, outbound_policy: _OutboundPolicy) -> None:
        """Replace the outbound policy of a running Sandbox.

        This API is experimental and may change in the future.

        The new policy replaces all existing policy configuration on the
        Sandbox; build a policy including any existing rules you want to keep.

        Only Sandboxes created with an `_experimental_outbound_policy` can be
        updated this way; for Sandboxes created without one this fails, since
        header replacement is only set up at creation time.

        Args:
            outbound_policy: The new policy to apply.
        """
        self._ensure_attached()
        outbound_policy._validate()
        await self._impl.update_outbound_policy(outbound_policy)

    async def _experimental_get_exit_snapshot(self, timeout: float | None = None) -> _Image:
        """Get the exit filesystem snapshot image.

        An exit snapshot captures the Sandbox filesystem when the Sandbox exits,
        whether its entrypoint finishes gracefully, abruptly, or it is stopped with
        `terminate()`. Exit snapshots are opt-in: the Sandbox must have been
        created with `experimental_options={"enable_exit_snapshot": True}`.

        Args:
            timeout: Total time to wait in seconds, spread across repeated long
                polls of at most `_EXIT_SNAPSHOT_LONG_POLL_TIMEOUT` seconds each.
                `None` (the default) waits until the snapshot reaches a terminal
                state; `0` performs an immediate check without waiting.

        Returns:
            The exit snapshot Image.

        Raises:
            InvalidError: If `timeout` is negative, or if exit snapshot is not
                enabled for the sandbox.
            TimeoutError: If `timeout` elapses before the snapshot reaches a
                terminal state. This includes `timeout=0` when the snapshot is
                still pending.
            SnapshotCreationError: Snapshot operation is done and failed.
                Polling again will not produce an Image; filesystem state is gone.
            NotFoundError: If the sandbox does not exist.
            PermissionDeniedError: If the caller cannot access the sandbox.
        """
        if timeout is not None and timeout < 0:
            raise InvalidError("timeout must be non-negative or None")
        return await self._impl.get_exit_snapshot(timeout)

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
            An [`Image`](https://modal.com/docs/sdk/py/latest/Image) object which can be used to spawn a new
            Sandbox with the same filesystem.
        """
        self._ensure_attached()
        return await self._impl.snapshot_filesystem(timeout, ttl=ttl)

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
        self._ensure_attached()
        image_id = _image_id_for_mount(image, "Sandbox.mount_image")
        await self._impl.mount_image(path, image_id, encryption_key=_experimental_encryption_key)

    async def unmount_image(self, path: PurePosixPath | str):
        """Unmount a previously mounted Image from a running Sandbox.

        `path` must be the exact mount point that was passed to `.mount_image()`.
        After unmounting, the underlying Sandbox filesystem at that path becomes
        visible again.

        Args:
            path: Absolute mount point directory to unmount.

        """
        self._ensure_attached()
        await self._impl.unmount_image(path)

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
        self._ensure_attached()
        return await self._impl.snapshot_directory(
            path, timeout=timeout, ttl=ttl, encryption_key=_experimental_encryption_key
        )

    # Live handle methods

    async def wait(self, raise_on_termination: bool = True):
        """Wait for the Sandbox to finish running.

        Args:
            raise_on_termination: If True, raise when the sandbox is terminated externally.

        """
        await self._impl.wait(raise_on_termination)

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
        self._ensure_attached()
        if timeout <= 0:
            raise InvalidError(f"`timeout` must be positive, got: {timeout}")
        await self._impl.wait_until_ready(timeout)

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
        self._ensure_attached()
        return await self._impl.tunnels(timeout)

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
        self._ensure_attached()
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
        resp = await self._impl.create_connect_token(req)
        return SandboxConnectCredentials(resp.url, resp.token)

    async def reload_volumes(self, *, timeout: int = 55) -> None:
        """Reload all Volumes mounted in the Sandbox.

        Added in v1.1.0.

        Blocks until the reload completes, or raises `modal.exception.TimeoutError` on timeout (the reload
        may still complete in the background).

        Args:
            timeout: Defaults to 55 seconds.
        """
        self._ensure_attached()
        await self._impl.reload_volumes(timeout=timeout)

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
        self._ensure_attached()
        await self._impl.terminate()
        if wait:
            await self.wait(raise_on_termination=False)
            return self.returncode
        return None

    async def _experimental_pause(self) -> None:
        """Stop the Sandbox's current container, keeping its state so it can be resumed.

        Only supported for experimental persistent Sandboxes.
        """
        await self._impl.pause()

    async def _experimental_resume(self) -> None:
        """Resume a paused Sandbox from its saved state.

        Only supported for experimental persistent Sandboxes.
        """
        await self._impl.resume()

    async def poll(self) -> int | None:
        """Check if the Sandbox has finished running.

        Returns:
            `None` if the Sandbox is still running, otherwise the exit code.
        """
        self._ensure_attached()
        await self._impl.poll()
        return self.returncode

    async def _get_task_id(
        self, raise_if_task_complete=False, timeout: float | None = None, retry_transient: bool = False
    ) -> str:
        self._ensure_attached()
        return await self._impl.get_task_id(
            raise_if_task_complete=raise_if_task_complete, timeout=timeout, retry_transient=retry_transient
        )

    async def _resolve_task_id_for_logs(self) -> str:
        return await self._impl.resolve_task_id_for_logs()

    async def _get_command_router_client(self, task_id: str) -> TaskCommandRouterClient:
        self._ensure_attached()
        return await self._impl.get_command_router_client(task_id)

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

        See the [`ContainerProcess`](https://modal.com/docs/sdk/py/latest/container_process#containerprocess)
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
            pty_info = default_pty_info()

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
        self._ensure_attached()
        return await self._impl.exec(
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
            container_id=container_id,
        )

    async def _experimental_snapshot(self) -> _SandboxSnapshot:
        self._ensure_attached()
        return await self._impl.snapshot_memory()

    @staticmethod
    async def _experimental_from_snapshot(
        snapshot: _SandboxSnapshot,
        client: _Client | None = None,
        *,
        name: str | None = _DEFAULT_SANDBOX_NAME_OVERRIDE,
    ):
        """Restore a Sandbox from a memory snapshot.

        The restore targets the same backend the snapshot was taken from. A V1
        snapshot restores as a V1 sandbox, a V2 snapshot as a V2 sandbox.
        """
        client = client or await _Client.from_env()

        if snapshot._is_v2 is None:
            await snapshot.hydrate(client=client)

        if name is not None and name != _DEFAULT_SANDBOX_NAME_OVERRIDE:
            check_object_name(name, "Sandbox")

        if name is _DEFAULT_SANDBOX_NAME_OVERRIDE:
            override_type = api_pb2.SandboxRestoreRequest.SANDBOX_NAME_OVERRIDE_TYPE_UNSPECIFIED
            name_override = None
        elif name is None:
            override_type = api_pb2.SandboxRestoreRequest.SANDBOX_NAME_OVERRIDE_TYPE_NONE
            name_override = None
        else:
            override_type = api_pb2.SandboxRestoreRequest.SANDBOX_NAME_OVERRIDE_TYPE_STRING
            name_override = name

        impl_cls = SandboxImplV2 if snapshot._is_v2 else SandboxImplV1
        impl, metadata = await impl_cls.restore_from_memory_snapshot(
            snapshot, client, override_type=override_type, name_override=name_override
        )
        return _Sandbox._from_impl(impl, metadata)

    @property
    def filesystem(self) -> _SandboxFilesystem:
        """Namespace for Sandbox filesystem APIs."""
        self._ensure_attached()
        if self._filesystem is None:
            self._filesystem = _SandboxFilesystem(self)
        return self._filesystem

    @property
    def stdout(self) -> _StreamReader[str]:
        """
        [`StreamReader`](https://modal.com/docs/sdk/py/latest/io_streams#streamreader)
        for the sandbox's stdout stream.

        Returns:
            Stream reader for sandbox stdout.
        """
        self._ensure_attached()
        return self._stdout

    @property
    def stderr(self) -> _StreamReader[str]:
        """
        [`StreamReader`](https://modal.com/docs/sdk/py/latest/io_streams#streamreader)
        for the Sandbox's stderr stream.

        Returns:
            Stream reader for sandbox stderr.
        """
        self._ensure_attached()
        return self._stderr

    @property
    def stdin(self) -> _StreamWriter:
        """
        [`StreamWriter`](https://modal.com/docs/sdk/py/latest/io_streams#streamwriter)
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
        return _result_returncode(self._impl.result)

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
        if config.get("sandbox_v2") is True:
            async for sandbox in _Sandbox._experimental_list(app_id=app_id, tags=tags, client=client):
                yield sandbox
            return

        environment_name = _get_environment_name()
        async for sandbox in _Sandbox._list(SandboxImplV1, app_id, environment_name, tags, client):
            yield sandbox

    @staticmethod
    async def _experimental_list(
        *, app_id: str | None = None, tags: dict[str, str] | None = None, client: _Client | None = None
    ) -> AsyncGenerator["_Sandbox", None]:
        """List all sandboxes (both v1 and v2).

        Pass `app_id` to scope to an App; without it, Sandboxes across the current
        Environment are listed (deprecated — prefer scoping by `app_id`).

        Args:
            app_id: If set, restrict results to Sandboxes under this App.
            tags: If set, only sandboxes containing at least these tags are returned.
            client: Optional client to use for the session.

        Yields:
            `Sandbox` objects that are currently running.
        """
        environment_name = ""
        if not app_id:
            deprecation_warning(
                (2026, 8, 10),
                "Listing Sandboxes without an `app_id` scans the entire Environment, which is deprecated. "
                "Pass an `app_id` to scope the query:\n\n"
                'app = modal.App.lookup("my-app")\n'
                "modal.Sandbox.list(app_id=app.app_id)",
            )
            environment_name = _get_environment_name()

        async for sandbox in _Sandbox._list(SandboxImplV2, app_id or "", environment_name, tags, client):
            yield sandbox

    @staticmethod
    async def _list(
        impl_cls: type[SandboxImpl],
        app_id: str | None,
        environment_name: str,
        tags: dict[str, str] | None,
        client: _Client | None,
    ) -> AsyncGenerator["_Sandbox", None]:
        before_timestamp = None
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
            resp = await impl_cls.list_page(client, req)

            if not resp.sandboxes:
                return

            for sandbox_info in resp.sandboxes:
                obj = _Sandbox._new_hydrated(sandbox_info.id, client, sandbox_info.metadata)
                obj._impl.result = sandbox_info.task_info.result
                yield obj

            # Fetch the next batch starting from the end of the current one.
            before_timestamp = resp.sandboxes[-1].created_at

    async def _get_log_query_data(self) -> _LogQueryData:
        if not self._app_id:
            if not self._attached:
                raise ExecutionError("Cannot fetch logs for a detached Sandbox without hydrated app_id")
            await self.hydrate()

        if not self._app_id:
            raise ExecutionError("app_id should have been set during sandbox hydration")

        task_id = await self._resolve_task_id_for_logs()
        return _LogQueryData(self._client, self._app_id, LogsFilters(task_id=task_id), self.object_id)

    @property
    def logs(self) -> _SandboxLogsManager:
        """Access logs for a `Sandbox` entrypoint.

        Useful for inspecting logs after a Sandbox terminates.
        Use [`fetch()`](#logsfetch)
        to read logs from a UTC time range, [`tail()`](#logstail)
        to read the most recent logs.

        Note that the logs from executed commands in the sandbox (via `exec()`) are not included in the
        entrypoint logs.

        See also:
            - [`modal app logs`](https://modal.com/docs/cli/latest/app#modal-app-logs):
              CLI access to logs for an App.
        """
        return _SandboxLogsManager(self)


async def _container_exec(
    pty: bool,
    container_id: str = "",
    command: tuple[str, ...] = (),
    sandbox_id_v2: str | None = None,
):
    """Execute a command in a container.

    For tasks belonging to V2 sandboxes, `sandbox_id_v2` may be set to the
    sandbox ID; otherwise the V2 backend is used whenever the task ID itself
    marks a V2 sandbox task.
    """
    client = await _Client.from_env()

    if sandbox_id_v2 is not None:
        fetch_access = v2_command_router_access_fetcher(client, sandbox_id=sandbox_id_v2)
    elif _is_v2_task_id(container_id):
        fetch_access = v2_command_router_access_fetcher(client, task_id=container_id)
    else:
        fetch_access = v1_command_router_access_fetcher(client, container_id)
    command_router_client = await TaskCommandRouterClient.connect(client, container_id, fetch_access)

    process_id = str(uuid.uuid4())

    start_req = sr_pb2.TaskExecStartRequest(
        task_id=container_id,
        exec_id=process_id,
        command_args=command,
        stdout_config=sr_pb2.TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_PIPE,
        stderr_config=sr_pb2.TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_PIPE,
        pty_info=get_pty_info(shell=True) if pty else None,
        runtime_debug=config.get("function_runtime_debug"),
    )
    await command_router_client.exec_start(start_req)

    if pty:
        # PTY output is raw terminal bytes (control sequences, mouse events,
        # glyphs in non-UTF-8 locales, etc.) — not text. Strict UTF-8 decode on
        # this stream crashes the shell as soon as anything emits a byte that
        # isn't valid UTF-8, e.g. vim drawing a Latin-1 file under LC_CTYPE=C.
        # Pass bytes through unmodified; `attach()` writes them straight to the
        # local fd.
        await _ContainerProcess(
            process_id,
            container_id,
            client,
            command_router_client=command_router_client,
            text=False,
        ).attach()
    else:
        await _ContainerProcess(
            process_id,
            container_id,
            client,
            command_router_client=command_router_client,
            stdout=StreamType.STDOUT,
            stderr=StreamType.STDOUT,
        ).wait()
