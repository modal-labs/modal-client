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

from grpclib import GRPCError, Status

from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._image import _Image
from .._outbound_policy import _OutboundPolicy
from .._resolver import Resolver
from .._tunnel import Tunnel
from ..client import _Client
from ..cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from ..config import logger
from ..container_process import _ContainerProcess
from ..exception import ClientClosed, InvalidError, SandboxTerminatedError
from ..io_streams import (
    StreamReader,
    StreamWriter,
    _StreamReaderThroughSandboxCommandRouterParams,
    _StreamWriterThroughCommandRouterSandboxParams,
)
from ..secret import _resolvable_secrets, _Secret, _secret_sources
from ..stream_type import StreamType
from ._common import SandboxCreateArgs, SandboxCreateCommon, SandboxImpl, command_router, sandbox_create_context
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

# How long to retry resuming while the previous session is still saving its exit snapshot.
_RESUME_RETRY_TIMEOUT = 120.0
_RESUME_RETRY_INTERVAL = 1.0

# How often `wait()` checks whether the persistent Sandbox has been terminated.
_WAIT_POLL_INTERVAL = 1.0

_TERMINATED_RESULT = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)


class SandboxImplPersistent:
    """Backs a persistent Sandbox, whose object ID is the ID of its first session.

    Operations are forwarded to the current session. When one fails and the Sandbox hasn't been terminated
    through this handle, the control plane is asked for the running session, resuming the Sandbox from its
    latest exit snapshot if there is none. The operation is then retried once on the new session.
    """

    def __init__(
        self,
        client: _Client,
        sandbox_id: str,
        *,
        result: api_pb2.GenericResult | None = None,
        persistent_sandbox_id: str = "",
        session: SandboxImplV2 | None = None,
    ) -> None:
        self.client = client
        self.sandbox_id = sandbox_id
        self.result = result
        self._persistent_sandbox_id = persistent_sandbox_id
        self._session = session if session is not None else SandboxImplV2(client, sandbox_id)
        self._replace_lock: asyncio.Lock | None = None
        self._terminated = result is not None

    @property
    def _delegate(self) -> SandboxImplV2:
        return self._session

    @property
    def _client(self) -> _Client:
        return self.client

    @classmethod
    async def create(
        cls, create_args: SandboxCreateArgs, *, app: "modal.app._App | None", client: _Client | None
    ) -> "tuple[SandboxImpl, api_pb2.SandboxHandleMetadata]":
        if create_args.timeout is not None or create_args.idle_timeout is not None:
            raise InvalidError("Persistent Sandboxes require `timeout=None` and don't support `idle_timeout`.")
        create_args = dataclasses.replace(
            create_args,
            timeout=PERSISTENT_SANDBOX_SESSION_TIMEOUT,
            idle_timeout=PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT,
        )
        if create_args.name is not None:
            raise InvalidError("Named persistent Sandboxes are not supported yet.")
        if create_args.gpu is not None or create_args.network_file_systems or create_args.pty_info is not None:
            raise InvalidError("GPUs, network file systems and PTYs are not supported by persistent Sandboxes yet.")
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
        impl = cls(client, resp.sandbox_id, persistent_sandbox_id=resp.persistent_sandbox_id)
        return impl, api_pb2.SandboxHandleMetadata(app_id=app_id)

    # Persistent Sandboxes are only created through `create`, and can't be looked up yet.

    @classmethod
    async def lookup_by_name(cls, *args: Any, **kwargs: Any) -> api_pb2.SandboxGetFromNameResponse:
        raise NotImplementedError

    @classmethod
    async def lookup_by_id(cls, *args: Any, **kwargs: Any) -> api_pb2.SandboxWaitResponse:
        raise NotImplementedError

    @classmethod
    async def list_page(cls, *args: Any, **kwargs: Any) -> api_pb2.SandboxListResponse:
        raise NotImplementedError

    @classmethod
    async def restore_from_memory_snapshot(
        cls, *args: Any, **kwargs: Any
    ) -> "tuple[SandboxImpl, api_pb2.SandboxHandleMetadata]":
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

    async def _run(self, op: Callable[[SandboxImplV2], Awaitable[_T]]) -> _T:
        """Run `op` on the session, switching to the running session and retrying once if it fails."""
        session = self._delegate
        try:
            return await op(session)
        except _NON_RECOVERABLE_ERRORS:
            raise
        except Exception as exc:
            if self._terminated:
                # A terminated persistent Sandbox is stopped for good, so it isn't resumed on use.
                raise
            replacement = await self._replace_session(session)
            if replacement is None:
                raise
            logger.debug(f"Retrying operation on persistent Sandbox {self._persistent_sandbox_id} after: {exc!r}")
            return await op(replacement)

    async def _get_running_session_id(self) -> str:
        req = api_pb2.PersistentSandboxGetRequest(persistent_sandbox_id=self._persistent_sandbox_id)
        resp = await self._client._stub.PersistentSandboxGet(req, metadata=await auth_metadata(self._client))
        return resp.sandbox_id

    async def _resume_session(self) -> str:
        """Resume the persistent Sandbox, returning the ID of its running session."""
        req = api_pb2.PersistentSandboxResumeRequest(persistent_sandbox_id=self._persistent_sandbox_id)
        deadline = time.monotonic() + _RESUME_RETRY_TIMEOUT
        while True:
            try:
                resp = await self._client._stub.PersistentSandboxResume(req, metadata=await auth_metadata(self._client))
                return resp.sandbox_id
            except GRPCError as exc:
                # UNAVAILABLE means the previous session is still pausing or another resume is in progress.
                if exc.status != Status.UNAVAILABLE or time.monotonic() > deadline:
                    raise
            await asyncio.sleep(_RESUME_RETRY_INTERVAL)

    async def _replace_session(self, failed: SandboxImplV2 | None, *, resume: bool = True) -> SandboxImplV2 | None:
        """Switch from the `failed` session, or return None if it is still the running session."""
        # Lazily create the lock on the event loop that first uses the Sandbox.
        if self._replace_lock is None:
            self._replace_lock = asyncio.Lock()
        async with self._replace_lock:
            if self._session is not failed:
                # A concurrent operation has already replaced the session.
                return self._session

            session_id = await self._get_running_session_id()
            if failed is not None and session_id == failed.sandbox_id:
                return None
            if not session_id:
                if not resume:
                    return None
                session_id = await self._resume_session()
            logger.debug(f"Switching persistent Sandbox {self._persistent_sandbox_id} to session {session_id}")
            replacement = SandboxImplV2(self.client, session_id)
            self._session = replacement
            if failed is not None:
                await failed.close()
            return replacement

    # Operations

    async def close(self) -> None:
        await self._session.close()

    def _set_terminated_result(self) -> None:
        self.result = _TERMINATED_RESULT

    async def wait(self, raise_on_termination: bool) -> None:
        # A persistent Sandbox keeps running across sessions, until it is terminated.
        while not self._terminated:
            await asyncio.sleep(_WAIT_POLL_INTERVAL)
        self._set_terminated_result()
        if raise_on_termination:
            raise SandboxTerminatedError()

    async def poll(self) -> None:
        if self._terminated:
            self._set_terminated_result()
        else:
            self.result = None

    async def terminate(self) -> None:
        req = api_pb2.PersistentSandboxTerminateRequest(persistent_sandbox_id=self._persistent_sandbox_id)
        await self._client._stub.PersistentSandboxTerminate(req, metadata=await auth_metadata(self._client))
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
        self._check_not_terminated()
        session = self._delegate
        session_id = await self._resume_session()
        if session_id != session.sandbox_id:
            self._session = SandboxImplV2(self.client, session_id)
            await session.close()

    def _check_not_terminated(self) -> None:
        if self._terminated:
            raise InvalidError(f"Persistent Sandbox {self._persistent_sandbox_id} has been terminated.")

    async def tunnels(self, timeout: int) -> dict[int, Tunnel]:
        return await self._run(lambda d: d.tunnels(timeout))

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
        def create(d: SandboxImplV2) -> Awaitable[api_pb2.SandboxCreateConnectTokenResponse]:
            delegate_req = api_pb2.SandboxCreateConnectTokenRequest()
            delegate_req.CopyFrom(req)
            delegate_req.sandbox_id = d.sandbox_id
            return d.create_connect_token(delegate_req)

        return await self._run(create)

    async def wait_until_ready(self, timeout: int) -> None:
        await self._run(lambda d: d.wait_until_ready(timeout))

    async def set_network_access(self, network_access: api_pb2.NetworkAccess) -> None:
        raise NotImplementedError("Updating network access isn't supported for persistent Sandboxes yet.")

    async def update_outbound_policy(self, outbound_policy: _OutboundPolicy) -> None:
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
        return await self._run(
            lambda d: d.exec(
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
        )

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
