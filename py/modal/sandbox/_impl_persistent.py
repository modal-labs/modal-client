# Copyright Modal Labs 2026
"""Persistent Sandbox lifecycle and session operations."""

import asyncio
import copy
import dataclasses
import time
import weakref
from collections.abc import Awaitable, Callable, Collection, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from google.protobuf.message import Message

from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._experimental_outbound_policy import _OutboundPolicy as _ExperimentalOutboundPolicy
from .._image import _Image
from .._outbound_policy import _OutboundPolicy
from .._resolver import Resolver
from .._tunnel import Tunnel
from .._utils.grpc_utils import Retry
from ..client import _Client
from ..cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from ..config import logger
from ..container_process import _ContainerProcess
from ..exception import ClientClosed, InvalidError, NotFoundError, SandboxPaused, SandboxTerminatedError
from ..io_streams import (
    StreamReader,
    StreamWriter,
    _StreamReaderThroughSandboxCommandRouterParams,
    _StreamWriterThroughCommandRouterSandboxParams,
)
from ..secret import _resolvable_secrets, _Secret, _secret_sources
from ..stream_type import StreamType
from ._common import (
    ExitSnapshotPoll,
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

# Lifetime limits of each session. A session that hits one is paused.
PERSISTENT_SANDBOX_SESSION_TIMEOUT = 60 * 60
PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT = 5 * 60
# How long a session lets running commands finish before it stops, by default and at most.
PERSISTENT_SANDBOX_DRAIN_TIMEOUT = 60
PERSISTENT_SANDBOX_MAX_DRAIN_TIMEOUT = 10 * 60

# Resuming is UNAVAILABLE while the previous session is still saving its exit snapshot or another resume is
# in progress, and pausing is while a session is starting, so both are retried this long at this interval.
_BUSY_RETRY = Retry(base_delay=1.0, max_delay=1.0, max_retries=None, total_timeout=120.0)

# How often `wait()` checks whether the persistent Sandbox has been terminated.
_WAIT_POLL_INTERVAL = 1.0

_TERMINATED_RESULT = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
_PAUSED_MESSAGE = "The Sandbox session has stopped or is shutting down. Call `_experimental_resume()` and retry."


def _rejected_by_shutting_down_session(exc: Exception) -> bool:
    """Whether the session was shutting down and never started the failed request."""
    details = getattr(exc, "_grpc_details", None) or []
    return any(isinstance(detail, sr_pb2.TaskShuttingDown) for detail in details)


def _can_report_paused(exc: Exception) -> bool:
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
    """Backs a persistent Sandbox, whose object ID is its persistent Sandbox ID (`sx-...`).

    A stopped session requires an explicit resume before running more work.
    """

    def __init__(
        self,
        client: _Client | None,
        sandbox_id: str,
        *,
        result: api_pb2.GenericResult | None = None,
        app_id: str = "",
        session: SandboxImplV2 | None = None,
    ) -> None:
        self.client = client
        self.sandbox_id = sandbox_id
        self.result = result
        self.app_id = app_id
        self._session = session
        self._session_lookup_lock: asyncio.Lock | None = None
        self._terminated = result is not None and bool(result.status)
        self._hydrated = session is not None or self._terminated
        self._closed = False

    async def _ensure_hydrated(self) -> None:
        await self._get_client()
        if self._hydrated:
            return
        if self._session_lookup_lock is None:
            self._session_lookup_lock = asyncio.Lock()
        async with self._session_lookup_lock:
            if not self._hydrated:
                session_id = await self._get_current_session_id()
                if self._session is None and session_id:
                    await self._set_session(session_id)

    async def get_app_id(self) -> str:
        if not self.app_id and (session := await self._session_for_reads()) is not None:
            self.app_id = await session.get_app_id()
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
        if self._session is not None:
            await self._session.hydrate(self.client)

    async def _get_client(self) -> _Client:
        await self.hydrate()
        assert self.client is not None
        return self.client

    def _get_object_metadata(self) -> None:
        return None

    @classmethod
    def _from_object_metadata(
        cls, client: _Client, sandbox_id: str, metadata: Message | None
    ) -> "SandboxImplPersistent":
        return cls(client, sandbox_id)

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
            req = api_pb2.SxSandboxCreateRequest(
                app_id=load_context.app_id or "",
                definition=common.definition(cloud_bucket_mounts=cloud_bucket_mount_protos),
            )
            resp = await client._stub.SxSandboxCreate(req, metadata=await auth_metadata(client))
            app_id = load_context.app_id or ""
        session = SandboxImplV2(client, resp.sandbox_id)
        impl = cls(client, resp.sx_id, session=session, app_id=app_id)
        return impl

    @classmethod
    async def lookup_by_id(cls, client: _Client, sandbox_id: str) -> "SandboxImplPersistent":
        impl = cls(client, sandbox_id)
        await impl._ensure_hydrated()
        return impl

    # Persistent Sandboxes can't be looked up by name or listed yet.

    @classmethod
    async def lookup_by_name(cls, *args: Any, **kwargs: Any) -> api_pb2.SandboxGetFromNameResponse:
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
            return await this._run(command_router)

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

    # Session lookup and resume

    async def _current_session(self) -> SandboxImplV2:
        """Return the session this handle is attached to."""
        self._check_open()
        await self._ensure_hydrated()
        self._check_not_terminated()
        if self._session is None:
            raise SandboxPaused(_PAUSED_MESSAGE)
        return self._session

    async def _run(
        self,
        op: Callable[[SandboxImplV2], Awaitable[_T]],
        *,
        can_report_paused: Callable[[Exception], bool] = _can_report_paused,
    ) -> _T:
        """Report a session that has stopped accepting work if the operation can safely be retried."""
        session = await self._current_session()
        try:
            return await op(session)
        except Exception as exc:
            if not self._terminated and can_report_paused(exc) and await self._session_has_stopped(session, exc):
                raise SandboxPaused(_PAUSED_MESSAGE) from exc
            raise

    async def _session_has_stopped(self, session: SandboxImplV2, exc: Exception) -> bool:
        try:
            session_id = await self._get_current_session_id()
            if self._terminated:
                return False
            if session_id != session.sandbox_id or _rejected_by_shutting_down_session(exc):
                return True
            try:
                await session.poll()
            except NotFoundError:
                return True
            return session.result is not None and bool(session.result.status)
        except Exception:
            # Preserve the operation's error if its session status could not be checked.
            return False

    async def _get_current_session_id(self) -> str:
        """Look up the most recent session, which may have finished already. Returns "" if there is none.

        Only termination is final: whether the session is still running is learned from the session itself.
        """
        client = await self._get_client()
        req = api_pb2.SxSandboxGetRequest(sx_id=self.sandbox_id)
        resp = await client._stub.SxSandboxGet(req, metadata=await auth_metadata(client))
        self._hydrated = True
        if resp.HasField("terminated_at"):
            self._terminated = True
            self._set_terminated_result()
        if self._terminated and resp.HasField("current_sandbox_id"):
            # The last session, which holds the exit snapshot and logs.
            await self._set_session(resp.current_sandbox_id)
        return resp.current_sandbox_id

    async def _resume_session(self) -> str:
        """Resume the persistent Sandbox, returning the ID of its running session."""
        client = await self._get_client()
        req = api_pb2.SxSandboxResumeRequest(sx_id=self.sandbox_id)
        resp = await client._stub.SxSandboxResume(req, metadata=await auth_metadata(client), retry=_BUSY_RETRY)
        return resp.sandbox_id

    async def _set_session(self, session_id: str) -> SandboxImplV2:
        """Make `session_id` the current session and close the session it replaces.

        Every session switch goes through here, so a replaced session never keeps a connection open, and a
        detached Sandbox only ever holds closed sessions.
        """
        previous = self._session
        if previous is not None and previous.sandbox_id == session_id:
            return previous
        logger.debug(f"Switching persistent Sandbox {self.sandbox_id} to session {session_id}")
        session = SandboxImplV2(await self._get_client(), session_id)
        self._session = session
        if previous is not None:
            await previous.close()
        if self._closed:
            await session.close()
        return session

    async def _session_for_reads(self) -> SandboxImplV2 | None:
        """Return the last known session for reading its logs or metadata, without resuming."""
        await self._ensure_hydrated()
        return self._session

    # Operations

    async def close(self) -> None:
        self._closed = True
        if self._session is not None:
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
        if not self._hydrated:
            await self._ensure_hydrated()
        elif not self._terminated:
            await self._get_current_session_id()
        if self._terminated:
            self._set_terminated_result()
        else:
            self.result = None

    async def terminate(self) -> None:
        client = await self._get_client()
        req = api_pb2.SxSandboxTerminateRequest(sx_id=self.sandbox_id)
        resp = await client._stub.SxSandboxTerminate(req, metadata=await auth_metadata(client))
        self._terminated = True
        self._hydrated = True
        self._set_terminated_result()
        if resp.HasField("last_session_sandbox_id"):
            # The last session, which holds the exit snapshot and logs.
            await self._set_session(resp.last_session_sandbox_id)

    async def pause(self) -> None:
        """Stop the current session, which saves its state for the next `resume`, and wait for it to stop.

        Does nothing if the Sandbox is already paused. A paused Sandbox is never resumed in order to pause it.
        """
        self._check_not_terminated()
        client = await self._get_client()
        req = api_pb2.SxSandboxPauseRequest(sx_id=self.sandbox_id)
        resp = await client._stub.SxSandboxPause(req, metadata=await auth_metadata(client), retry=_BUSY_RETRY)
        if not resp.HasField("sandbox_id"):
            return
        # Wait on the session the pause stops, even if another handle has resumed the Sandbox since.
        session = await self._set_session(resp.sandbox_id)
        await session.wait(raise_on_termination=False)

    async def resume(self) -> None:
        """Start a session from the latest saved state, unless one is already running."""
        await self._get_client()
        self._check_not_terminated()
        await self._set_session(await self._resume_session())
        self._hydrated = True

    def _check_open(self) -> None:
        if self._closed:
            raise ClientClosed("Unable to perform operation on a detached sandbox")

    def _check_not_terminated(self) -> None:
        if self._terminated:
            raise InvalidError(f"Persistent Sandbox {self.sandbox_id} has been terminated.")

    async def tunnels(self, timeout: int) -> dict[int, Tunnel]:
        client = await self._get_client()
        req = api_pb2.SxSandboxTunnelsListRequest(sx_id=self.sandbox_id)
        resp = await client._stub.SxSandboxTunnelsList(req, metadata=await auth_metadata(client))
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
        # Logs are read from the last known session, so reading them never resumes a paused Sandbox.
        session = await self._session_for_reads()
        if session is None:
            raise InvalidError(f"Persistent Sandbox {self.sandbox_id} has no session to read logs from.")
        return await session.resolve_task_id_for_logs()

    async def get_exit_snapshot(self, timeout: float | None) -> _Image:
        # Pausing a persistent Sandbox is not an exit: only termination produces its exit snapshot.
        deadline = None if timeout is None else time.monotonic() + timeout
        await self.poll()
        while not self._terminated:
            if deadline is not None and time.monotonic() + _WAIT_POLL_INTERVAL > deadline:
                raise ExitSnapshotPoll(self.sandbox_id, timeout).timeout_error()
            await asyncio.sleep(_WAIT_POLL_INTERVAL)
            await self.poll()
        if self._session is None:
            raise InvalidError(
                f"Persistent Sandbox {self.sandbox_id} has no exit snapshot: it was terminated without a session."
            )
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        return await self._session.get_exit_snapshot(remaining)

    async def snapshot_memory(self) -> _SandboxSnapshot:
        raise NotImplementedError("Memory snapshots aren't supported for persistent Sandboxes.")

    async def get_command_router_client(self, task_id: str) -> TaskCommandRouterClient:
        # The task ID belongs to the current session, so this can't be retried on a replacement.
        return await (await self._current_session()).get_command_router_client(task_id)

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
        persistent_req = api_pb2.SxSandboxCreateConnectTokenRequest(
            sx_id=self.sandbox_id, user_metadata=req.user_metadata
        )
        if req.HasField("port"):
            persistent_req.port = req.port
        client = await self._get_client()
        resp = await client._stub.SxSandboxCreateConnectToken(persistent_req, metadata=await auth_metadata(client))
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
            return await self._run(start, can_report_paused=_exec_never_ran)
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
        SandboxImplV2.validate_sidecar_create(
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
        # Sidecar containers don't carry over to a replacement session.
        return await (await self._current_session()).wait_sidecar(container_id, timeout)

    async def terminate_sidecar(self, container_id: str) -> None:
        await (await self._current_session()).terminate_sidecar(container_id)
