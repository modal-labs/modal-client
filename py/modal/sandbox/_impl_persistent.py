# Copyright Modal Labs 2026
"""Persistent Sandboxes, managed by the `PersistentSandbox*` control-plane RPCs.

A persistent Sandbox runs as a sequence of regular V2 Sandboxes, its sessions. The control plane starts each
session from the exit snapshot of the previous one.
"""

import asyncio
import copy
import dataclasses
import time
import weakref
from collections.abc import Awaitable, Callable, Collection, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._experimental_outbound_policy import _OutboundPolicy as _ExperimentalOutboundPolicy
from .._image import _Image
from .._outbound_policy import _OutboundPolicy
from .._resolver import Resolver
from .._tunnel import Tunnel
from ..client import _Client
from ..cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from ..config import logger
from ..container_process import _ContainerProcess
from ..exception import ClientClosed, InvalidError, NotFoundError, SandboxTerminatedError
from ..io_streams import (
    StreamReader,
    StreamWriter,
    _StreamReaderThroughSandboxCommandRouterParams,
    _StreamWriterThroughCommandRouterSandboxParams,
)
from ..secret import _resolvable_secrets, _Secret, _secret_sources
from ..stream_type import StreamType
from ._common import (
    SandboxCreateArgs,
    SandboxCreateCommon,
    SandboxImpl,
    command_router,
    sandbox_create_context,
    tunnels_from_protos,
)
from ._impl_v2 import SandboxImplV2, auth_metadata
from ._snapshot import _SandboxSnapshot
from ._task_command_router_client import TaskCommandRouterClient

if TYPE_CHECKING:
    import modal.app

_T = TypeVar("_T")

# Errors caused by the caller rather than by the session, which a new session can't fix.
_NON_RECOVERABLE_ERRORS = (InvalidError, ClientClosed)

# Lifetime limits of each session. A session that hits one is paused, and resumed on next use.
PERSISTENT_SANDBOX_SESSION_TIMEOUT = 60 * 60
PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT = 5 * 60
# How long a session lets running commands finish before it stops, by default and at most.
PERSISTENT_SANDBOX_DRAIN_TIMEOUT = 60
PERSISTENT_SANDBOX_MAX_DRAIN_TIMEOUT = 10 * 60

# How long to retry resuming while the previous session is still saving its exit snapshot.
_RESUME_RETRY_TIMEOUT = 120.0
_RESUME_RETRY_INTERVAL = 1.0

# How often `wait()` checks whether the persistent Sandbox has been terminated.
_WAIT_POLL_INTERVAL = 1.0

_TERMINATED_RESULT = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)


def _rejected_by_shutting_down_session(exc: Exception) -> bool:
    """Whether the session was shutting down and never started the failed request."""
    details = getattr(exc, "_grpc_details", None) or []
    return any(isinstance(detail, sr_pb2.TaskShuttingDown) for detail in details)


def _may_succeed_on_new_session(exc: Exception) -> bool:
    return _rejected_by_shutting_down_session(exc) or not isinstance(exc, _NON_RECOVERABLE_ERRORS)


class _ExecNotSentError(Exception):
    """A failure to reach a session's command router, which comes before the command is sent to it."""

    def __init__(self, inner: Exception) -> None:
        super().__init__(str(inner))
        self.inner = inner


def _exec_never_ran(exc: Exception) -> bool:
    """Whether a failed exec certainly didn't run its command, so it can be started on another session."""
    if isinstance(exc, _ExecNotSentError):
        # Reaching the router takes no input from the caller, so only a closed client is their doing.
        return not isinstance(exc.inner, ClientClosed)
    if _rejected_by_shutting_down_session(exc):
        return True
    # A NOT_FOUND status means the session was gone before the command could
    # reach it. Any other failure, including a dropped connection, may come
    # after the command started.
    return isinstance(exc, NotFoundError)


class SandboxImplPersistent:
    """Backs a persistent Sandbox, whose object ID is the ID of its first session.

    Operations are forwarded to the current session. When one fails and the Sandbox hasn't been terminated
    through this handle, the control plane is asked for the running session, resuming the Sandbox from its
    latest exit snapshot if there is none. The operation is then retried once on the new session.
    """

    def __init__(
        self,
        client: _Client | None,
        sandbox_id: str,
        *,
        result: api_pb2.GenericResult | None = None,
        app_id: str = "",
        persistent_sandbox_id: str = "",
        session: SandboxImplV2 | None = None,
    ) -> None:
        self.client = client
        self.sandbox_id = sandbox_id
        self.result = result
        self.app_id = app_id
        self._persistent_sandbox_id = persistent_sandbox_id
        self._session = session if session is not None else SandboxImplV2(client, sandbox_id)
        self._replace_lock: asyncio.Lock | None = None
        self._terminated = result is not None
        self._closed = False

    @property
    def _delegate(self) -> SandboxImplV2:
        return self._session

    async def get_app_id(self) -> str:
        return self.app_id

    @property
    def is_hydrated(self) -> bool:
        # An impl always has an object id, while its client is optional, so having a client
        # is equivalent to being hydrated.
        return self.client is not None

    async def hydrate(self, client: _Client | None = None) -> None:
        if self.client is not None and self.client._snapshotted:
            self.client = client or await _Client.from_env()
        else:
            self.client = client or self.client or await _Client.from_env()
        await self._session.hydrate(self.client)

    async def _get_client(self) -> _Client:
        await self.hydrate()
        assert self.client is not None
        return self.client

    def _get_object_metadata(self) -> api_pb2.SandboxHandleMetadata:
        metadata = api_pb2.SandboxHandleMetadata(app_id=self.app_id)
        if self.result is not None:
            metadata.result.CopyFrom(self.result)
        return metadata

    @classmethod
    def _from_object_metadata(
        cls, client: _Client, sandbox_id: str, metadata: Message | None
    ) -> "SandboxImplPersistent":
        if metadata is None:
            return cls(client, sandbox_id)
        assert isinstance(metadata, api_pb2.SandboxHandleMetadata)
        return cls(
            client, sandbox_id, app_id=metadata.app_id, result=metadata.result if metadata.HasField("result") else None
        )

    @classmethod
    async def create(
        cls, create_args: SandboxCreateArgs, *, app: "modal.app._App | None", client: _Client | None
    ) -> "SandboxImpl":
        if create_args.timeout is not None or create_args.idle_timeout is not None:
            raise InvalidError("Persistent Sandboxes require `timeout=None` and don't support `idle_timeout`.")
        drain_timeout = create_args.drain_timeout
        if drain_timeout is None:
            drain_timeout = PERSISTENT_SANDBOX_DRAIN_TIMEOUT
        elif not 0 < drain_timeout <= PERSISTENT_SANDBOX_MAX_DRAIN_TIMEOUT:
            raise InvalidError(
                '`experimental_options["drain_timeout_secs"]` must be between 1 and '
                f"{PERSISTENT_SANDBOX_MAX_DRAIN_TIMEOUT} seconds."
            )
        create_args = dataclasses.replace(
            create_args,
            timeout=PERSISTENT_SANDBOX_SESSION_TIMEOUT,
            idle_timeout=PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT,
            drain_timeout=drain_timeout,
        )
        if create_args.name is not None:
            raise InvalidError("Named persistent Sandboxes are not supported yet.")
        if create_args.gpu is not None or create_args.pty_info is not None:
            raise InvalidError("GPUs and PTYs are not supported by persistent Sandboxes yet.")
        if create_args.enable_snapshot:
            raise InvalidError("Memory snapshots are not supported by persistent Sandboxes.")
        if create_args.env or create_args.tags:
            raise InvalidError("`env` and `tags` are not supported by persistent Sandboxes yet.")

        secrets = create_args.secrets or []
        common = SandboxCreateCommon(create_args, secrets=_resolvable_secrets(secrets))
        async with sandbox_create_context(app, client) as (client, load_context):
            await common.load_dependencies(Resolver(), load_context, load_ephemeral_bucket_secrets=False)
            cloud_bucket_mount_protos, cloud_bucket_credentials = cloud_bucket_mounts_to_proto(
                common.cloud_bucket_mounts, split_ephemeral_credentials=True
            )
            if cloud_bucket_credentials or any(source.HasField("env") for source in _secret_sources(secrets)):
                raise InvalidError("Ephemeral credentials are not supported by persistent Sandboxes yet.")
            req = api_pb2.PersistentSandboxCreateRequest(
                app_id=load_context.app_id or "",
                definition=common.definition(cloud_bucket_mounts=cloud_bucket_mount_protos),
            )
            resp = await client._stub.PersistentSandboxCreate(req, metadata=await auth_metadata(client))
            app_id = load_context.app_id or ""
        impl = cls(client, resp.sandbox_id, persistent_sandbox_id=resp.persistent_sandbox_id, app_id=app_id)
        return impl

    # Persistent Sandboxes are only created through `create`, and can't be looked up yet.

    @classmethod
    async def lookup_by_name(cls, *args: Any, **kwargs: Any) -> api_pb2.SandboxGetFromNameResponse:
        raise NotImplementedError

    @classmethod
    async def lookup_by_id(cls, *args: Any, **kwargs: Any) -> "SandboxImpl":
        raise NotImplementedError

    @classmethod
    async def list_page(cls, *args: Any, **kwargs: Any) -> api_pb2.SandboxListResponse:
        raise NotImplementedError

    @classmethod
    async def restore_from_memory_snapshot(cls, *args: Any, **kwargs: Any) -> "SandboxImpl":
        raise NotImplementedError

    def clone(self) -> "SandboxImplPersistent":
        return copy.copy(self)

    def make_stdio(self, check_open: Callable[[], None]) -> tuple[StreamReader[str], StreamReader[str], StreamWriter]:
        """Create stdio streams of the entrypoint of whichever session is current when they connect."""
        weak_self = weakref.ref(self)

        async def resolve_router() -> tuple[str, TaskCommandRouterClient]:
            this = weak_self()
            if this is None:
                raise RuntimeError("Sandbox was garbage collected before its stdio reader connected")
            this._check_open()
            return await command_router(this)

        def reader(file_descriptor: "api_pb2.FileDescriptor.ValueType") -> StreamReader[str]:
            return StreamReader(
                _StreamReaderThroughSandboxCommandRouterParams(
                    file_descriptor=file_descriptor, sandbox_id=self.sandbox_id, resolve_router=resolve_router
                ),
                by_line=True,
            )

        stdin = StreamWriter(
            _StreamWriterThroughCommandRouterSandboxParams(resolve_router=resolve_router, check_open=check_open)
        )
        return reader(api_pb2.FILE_DESCRIPTOR_STDOUT), reader(api_pb2.FILE_DESCRIPTOR_STDERR), stdin

    # Session replacement

    async def _run(
        self,
        op: Callable[[SandboxImplV2], Awaitable[_T]],
        *,
        can_retry: Callable[[Exception], bool] = _may_succeed_on_new_session,
    ) -> _T:
        """Run `op` on the session, switching to the running session and retrying once if it fails.

        Only failures accepted by `can_retry` are retried.
        """
        session = self._delegate
        try:
            return await op(session)
        except Exception as exc:
            if self._terminated:
                # A terminated persistent Sandbox is stopped for good, so it isn't resumed on use.
                raise
            if not can_retry(exc):
                raise
            replacement = await self._replace_session(session)
            if replacement is None:
                raise
            logger.debug(f"Retrying operation on persistent Sandbox {self._persistent_sandbox_id} after: {exc!r}")
            return await op(replacement)

    async def _get_current_session_id(self) -> str:
        client = await self._get_client()
        req = api_pb2.PersistentSandboxGetRequest(persistent_sandbox_id=self._persistent_sandbox_id)
        resp = await client._stub.PersistentSandboxGet(req, metadata=await auth_metadata(client))
        if resp.HasField("terminated_at"):
            self._terminated = True
            self._set_terminated_result()
        return "" if self._terminated else resp.current_sandbox_id

    async def _resume_session(self) -> str:
        """Resume the persistent Sandbox, returning the ID of its running session."""
        client = await self._get_client()
        req = api_pb2.PersistentSandboxResumeRequest(persistent_sandbox_id=self._persistent_sandbox_id)
        deadline = time.monotonic() + _RESUME_RETRY_TIMEOUT
        while True:
            try:
                resp = await client._stub.PersistentSandboxResume(req, metadata=await auth_metadata(client))
                return resp.sandbox_id
            except GRPCError as exc:
                # UNAVAILABLE means the previous session is still pausing or another resume is in progress.
                if exc.status != Status.UNAVAILABLE or time.monotonic() > deadline:
                    raise
            await asyncio.sleep(_RESUME_RETRY_INTERVAL)

    async def _replace_session(self, failed: SandboxImplV2 | None, *, resume: bool = True) -> SandboxImplV2 | None:
        """Switch from the `failed` session, or return None if it is still the running session."""
        client = await self._get_client()
        # Lazily create the lock on the event loop that first uses the Sandbox.
        if self._replace_lock is None:
            self._replace_lock = asyncio.Lock()
        async with self._replace_lock:
            if self._session is not failed:
                # A concurrent operation has already replaced the session.
                return self._session

            session_id = await self._get_current_session_id()
            self._check_open()
            self._check_not_terminated()
            # The current session may have finished already, so the lookup can't tell
            # whether `failed` is still running. Resuming returns the running session,
            # which is `failed` itself if it never stopped.
            if not session_id or (failed is not None and session_id == failed.sandbox_id):
                if not resume:
                    return None
                session_id = await self._resume_session()
                if failed is not None and session_id == failed.sandbox_id:
                    return None
            logger.debug(f"Switching persistent Sandbox {self._persistent_sandbox_id} to session {session_id}")
            replacement = SandboxImplV2(client, session_id)
            self._session = replacement
            if failed is not None:
                await failed.close()
            return replacement

    # Operations

    async def close(self) -> None:
        self._closed = True
        await self._session.close()

    def ensure_attached(self) -> None:
        self._check_open()

    def _set_terminated_result(self) -> None:
        self.result = _TERMINATED_RESULT

    async def wait(self, raise_on_termination: bool) -> None:
        await self.poll()
        while not self._terminated:
            await asyncio.sleep(_WAIT_POLL_INTERVAL)
            await self.poll()
        self._set_terminated_result()
        if raise_on_termination:
            raise SandboxTerminatedError()

    async def poll(self) -> None:
        if not self._terminated:
            await self._get_current_session_id()
        if self._terminated:
            self._set_terminated_result()
        else:
            self.result = None

    async def terminate(self) -> None:
        client = await self._get_client()
        req = api_pb2.PersistentSandboxTerminateRequest(persistent_sandbox_id=self._persistent_sandbox_id)
        await client._stub.PersistentSandboxTerminate(req, metadata=await auth_metadata(client))
        self._terminated = True
        self._set_terminated_result()

    async def pause(self) -> None:
        """Stop the current session, which saves its exit snapshot for the next `resume`."""
        self._check_not_terminated()
        session = self._delegate
        await session.terminate()
        await session.wait(raise_on_termination=False)

    async def resume(self) -> None:
        """Start a session from the latest exit snapshot, unless one is already running."""
        client = await self._get_client()
        self._check_not_terminated()
        session = self._delegate
        session_id = await self._resume_session()
        if session_id != session.sandbox_id:
            self._session = SandboxImplV2(client, session_id)
            await session.close()

    def _check_open(self) -> None:
        if self._closed:
            raise ClientClosed("Unable to perform operation on a detached sandbox")

    def _check_not_terminated(self) -> None:
        if self._terminated:
            raise InvalidError(f"Persistent Sandbox {self._persistent_sandbox_id} has been terminated.")

    async def tunnels(self, timeout: int) -> dict[int, Tunnel]:
        client = await self._get_client()
        req = api_pb2.PersistentSandboxTunnelsListRequest(persistent_sandbox_id=self._persistent_sandbox_id)
        resp = await client._stub.PersistentSandboxTunnelsList(req, metadata=await auth_metadata(client))
        return tunnels_from_protos(resp.tunnels)

    async def get_task_id(
        self, raise_if_task_complete: bool = False, timeout: float | None = None, retry_transient: bool = False
    ) -> str:
        return await self._run(
            lambda d: d.get_task_id(
                raise_if_task_complete=raise_if_task_complete, timeout=timeout, retry_transient=retry_transient
            )
        )

    async def resolve_task_id_for_logs(self) -> str:
        return await self._delegate.resolve_task_id_for_logs()

    async def get_exit_snapshot(self, timeout: float | None) -> _Image:
        return await self._delegate.get_exit_snapshot(timeout)

    async def snapshot_memory(self) -> _SandboxSnapshot:
        raise NotImplementedError("Memory snapshots aren't supported for persistent Sandboxes.")

    async def get_command_router_client(self, task_id: str) -> TaskCommandRouterClient:
        # The task ID belongs to the current delegate, so this can't be retried on a replacement.
        return await self._delegate.get_command_router_client(task_id)

    async def get_tags(self) -> dict[str, str]:
        raise NotImplementedError("Tags aren't supported for persistent Sandboxes yet.")

    async def set_tags(self, tags: dict[str, str]) -> None:
        raise NotImplementedError("Tags aren't supported for persistent Sandboxes yet.")

    async def set_name(self, name: str) -> None:
        raise InvalidError("The name of a persistent Sandbox can't be changed.")

    async def create_connect_token(
        self, req: api_pb2.SandboxCreateConnectTokenRequest
    ) -> api_pb2.SandboxCreateConnectTokenResponse:
        # Connect tokens are addressed by the persistent Sandbox, not a session, so they keep working across
        # sessions and are created without resuming it.
        persistent_req = api_pb2.PersistentSandboxCreateConnectTokenRequest(
            persistent_sandbox_id=self._persistent_sandbox_id, user_metadata=req.user_metadata
        )
        if req.HasField("port"):
            persistent_req.port = req.port
        client = await self._get_client()
        resp = await client._stub.PersistentSandboxCreateConnectToken(
            persistent_req, metadata=await auth_metadata(client)
        )
        return api_pb2.SandboxCreateConnectTokenResponse(url=resp.url, token=resp.token)

    async def wait_until_ready(self, timeout: int) -> None:
        await self._run(lambda d: d.wait_until_ready(timeout))

    async def set_network_access(self, network_access: api_pb2.NetworkAccess) -> None:
        raise NotImplementedError("Updating network access isn't supported for persistent Sandboxes yet.")

    async def set_outbound_policy(self, outbound_policy: _OutboundPolicy) -> None:
        raise NotImplementedError("Setting the outbound policy isn't supported for persistent Sandboxes yet.")

    async def update_experimental_outbound_policy(self, outbound_policy: _ExperimentalOutboundPolicy) -> None:
        raise NotImplementedError("Updating the outbound policy isn't supported for persistent Sandboxes yet.")

    async def snapshot_filesystem(self, timeout: int, *, ttl: int | None, container_id: str = "") -> _Image:
        return await self._run(lambda d: d.snapshot_filesystem(timeout, ttl=ttl, container_id=container_id))

    async def mount_image(
        self,
        path: PurePosixPath | str,
        image_id: str,
        *,
        encryption_key: bytes | None,
        container_id: str = "",
    ) -> None:
        raise NotImplementedError("Mounting images isn't supported for persistent Sandboxes yet.")

    async def unmount_image(self, path: PurePosixPath | str, *, container_id: str = "") -> None:
        raise NotImplementedError("Mounting images isn't supported for persistent Sandboxes yet.")

    async def snapshot_directory(
        self,
        path: PurePosixPath | str,
        *,
        timeout: int,
        ttl: int | None,
        encryption_key: bytes | None,
        container_id: str = "",
    ) -> _Image:
        return await self._run(
            lambda d: d.snapshot_directory(
                path, timeout=timeout, ttl=ttl, encryption_key=encryption_key, container_id=container_id
            )
        )

    async def reload_volumes(self, *, timeout: int, container_id: str = "") -> None:
        await self._run(lambda d: d.reload_volumes(timeout=timeout, container_id=container_id))

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
    ) -> _ContainerProcess[bytes] | _ContainerProcess[str]:
        async def start(d: SandboxImplV2) -> _ContainerProcess[bytes] | _ContainerProcess[str]:
            try:
                task_id = await d.get_task_id(raise_if_task_complete=True)
                await d.get_command_router_client(task_id)
            except Exception as exc:
                raise _ExecNotSentError(exc) from exc
            return await d.exec(
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

        try:
            return await self._run(start, can_retry=_exec_never_ran)
        except _ExecNotSentError as exc:
            raise exc.inner from None

    # Sidecar containers

    def validate_sidecar_create(
        self,
        *,
        cloud_bucket_mounts: Sequence[tuple[str, _CloudBucketMount]],
        has_proxy: bool,
        include_oidc_identity_token: bool,
    ) -> None:
        self._delegate.validate_sidecar_create(
            cloud_bucket_mounts=cloud_bucket_mounts,
            has_proxy=has_proxy,
            include_oidc_identity_token=include_oidc_identity_token,
        )

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
        return await self._run(
            lambda d: d.create_sidecar(
                *args,
                name=name,
                image_id=image_id,
                secret_ids=secret_ids,
                secret_sources=secret_sources,
                workdir=workdir,
                volume_mounts=volume_mounts,
                cloud_bucket_mounts=cloud_bucket_mounts,
                network_access=network_access,
                include_oidc_identity_token=include_oidc_identity_token,
                proxy_id=proxy_id,
                pty_info=pty_info,
                memory_reserve_consume_mib=memory_reserve_consume_mib,
            )
        )

    async def get_sidecar(self, name: str, include_terminated: bool) -> sr_pb2.TaskContainerInfo:
        return await self._run(lambda d: d.get_sidecar(name, include_terminated))

    async def list_sidecars(self, include_terminated: bool) -> list[sr_pb2.TaskContainerInfo]:
        return await self._run(lambda d: d.list_sidecars(include_terminated))

    async def wait_sidecar(self, container_id: str, timeout: float) -> api_pb2.GenericResult | None:
        # Sidecar containers don't carry over to a replacement delegate.
        return await self._delegate.wait_sidecar(container_id, timeout)

    async def terminate_sidecar(self, container_id: str) -> None:
        await self._delegate.terminate_sidecar(container_id)
