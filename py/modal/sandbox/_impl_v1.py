# Copyright Modal Labs 2026
"""Sandbox implementation for V1 Sandboxes, which are managed through the `Sandbox*` control-plane RPCs."""

import asyncio
import logging
import os
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

from google.protobuf.message import Message

from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._experimental_outbound_policy import _OutboundPolicy as _ExperimentalOutboundPolicy
from .._image import _Image
from .._object import _get_environment_name
from .._outbound_policy import _OutboundPolicy
from .._resolver import Resolver
from .._tunnel import Tunnel
from .._utils.grpc_utils import RetryTimeoutError
from ..client import _Client
from ..cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from ..config import config, logger
from ..container_process import _ContainerProcess
from ..exception import (
    ConflictError,
    ExecutionError,
    InternalError,
    InvalidError,
    NotFoundError,
    ResourceExhaustedError,
    ServiceError,
    TimeoutError,
)
from ..io_streams import (
    StreamReader,
    StreamWriter,
    _StreamReaderThroughServerParams,
    _StreamWriterThroughServerParams,
)
from ..secret import _Secret
from ..stream_type import StreamType
from ._common import (
    _PERSISTENT_ONLY_MESSAGE,
    _TASK_ID_POLL_INTERVAL,
    ExitSnapshotPoll,
    SandboxCreateArgs,
    SandboxCreateCommon,
    SandboxImpl,
    _format_sandbox_create_timing_log,
    apply_wait_response,
    create_sidecar_via_tcr,
    exec_via_tcr,
    get_sidecar_via_tcr,
    list_sidecars_via_tcr,
    mount_image_via_tcr,
    new_snapshot_handle,
    raise_for_wait_result,
    raise_unless_retryable_task_id_error,
    reload_volumes_via_tcr,
    retry_while_starting,
    sandbox_create_context,
    set_network_access_via_tcr,
    set_outbound_policy_request,
    snapshot_directory_via_tcr,
    snapshot_filesystem_via_tcr,
    tag_protos,
    tags_from_protos,
    task_id_for_logs,
    task_id_from_response,
    task_id_poll_deadline,
    terminate_sidecar_via_tcr,
    tunnels_from_response,
    unmount_image_via_tcr,
    update_experimental_outbound_policy_via_tcr,
    wait_poll_request,
    wait_sidecar_via_tcr,
    wait_until_ready_via_tcr,
)
from ._snapshot import _SandboxSnapshot
from ._task_command_router_client import (
    CommandRouterAccessFetcher,
    LazyCommandRouterClient,
    TaskCommandRouterClient,
)

if TYPE_CHECKING:
    import modal.app


# How long `Sandbox.create` waits for capacity before giving up.
_SANDBOX_SCHEDULING_TIMEOUT: float = 21 * 60

DEFAULT_SNAPSHOT_TTL = 30 * 24 * 3600


def command_router_access_fetcher(client: _Client, task_id: str) -> CommandRouterAccessFetcher:
    """Return a fetcher of command router access for a V1 Sandbox's task."""

    async def fetch() -> api_pb2.CommandRouterAccess:
        req = api_pb2.TaskGetCommandRouterAccessRequest(task_id=task_id)
        resp = await client._stub.TaskGetCommandRouterAccess(req)
        return api_pb2.CommandRouterAccess(url=resp.url, jwt=resp.jwt)

    return fetch


class SandboxImplV1:
    _tunnels: dict[int, Tunnel] | None
    _task_id: str | None
    _command_router: LazyCommandRouterClient

    def __init__(
        self,
        client: _Client | None,
        sandbox_id: str,
        *,
        result: api_pb2.GenericResult | None = None,
        app_id: str = "",
        task_id: str | None = None,
    ) -> None:
        self.client = client
        self.sandbox_id = sandbox_id
        self.result = result
        self.app_id = app_id
        self._task_id = task_id
        self._command_router = LazyCommandRouterClient(self._open_command_router_client)
        self._tunnels = None

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
    def _from_object_metadata(cls, client: _Client, sandbox_id: str, metadata: Message | None) -> "SandboxImplV1":
        if metadata is None:
            return cls(client, sandbox_id)
        assert isinstance(metadata, api_pb2.SandboxHandleMetadata), f"{type(metadata)} is not SandboxHandleMetadata"
        return cls(
            client, sandbox_id, app_id=metadata.app_id, result=metadata.result if metadata.HasField("result") else None
        )

    async def get_app_id(self) -> str:
        if not self.app_id:
            await self.poll()
        return self.app_id

    def clone(self) -> "SandboxImplV1":
        return type(self)(self.client, self.sandbox_id, result=self.result, app_id=self.app_id)

    @classmethod
    async def create(
        cls, create_args: SandboxCreateArgs, *, app: "modal.app._App | None", client: _Client | None
    ) -> "SandboxImpl":
        if create_args.i6pn:
            raise InvalidError("i6pn is not supported by V1 Sandboxes")
        # V1 references every Secret by id, including ephemeral ones and the `env` dict.
        secrets = list(create_args.secrets or [])
        if create_args.env:
            secrets.append(_Secret.from_dict(create_args.env))

        common = SandboxCreateCommon(create_args, secrets=secrets)
        load_start = time.monotonic()
        async with sandbox_create_context(app, client) as (client, load_context):
            dep_timings = await common.load_dependencies(Resolver(), load_context, load_ephemeral_bucket_secrets=True)
            definition = common.definition(
                cloud_bucket_mounts=cloud_bucket_mounts_to_proto(common.cloud_bucket_mounts)[0]
            )

            create_req = api_pb2.SandboxCreateRequest(
                app_id=load_context.app_id, definition=definition, tags=tag_protos(create_args.tags)
            )
            rpc_start = time.monotonic()
            create_resp = await client._stub.SandboxCreate(create_req)
            rpc_elapsed = time.monotonic() - rpc_start
            sandbox_id = create_resp.sandbox_id
            impl = cls(client, sandbox_id)
            try:
                await impl.get_task_id(
                    raise_if_task_complete=True, timeout=_SANDBOX_SCHEDULING_TIMEOUT, retry_transient=True
                )
            except ConflictError:
                raise
            except BaseException as exc:
                # The caller never receives the Sandbox, so terminate it so it can't start later without a caller.
                try:
                    await asyncio.shield(impl.terminate())
                except Exception as terminate_exc:
                    logger.debug(f"Failed to terminate unscheduled Sandbox {sandbox_id}: {terminate_exc}")
                if isinstance(exc, TimeoutError):
                    raise ResourceExhaustedError("Insufficient capacity to create sandbox.") from None
                raise

            if logger.isEnabledFor(logging.DEBUG):
                total_elapsed = time.monotonic() - load_start
                logger.debug(_format_sandbox_create_timing_log(sandbox_id, total_elapsed, rpc_elapsed, dep_timings))
        impl.app_id = create_resp.metadata.app_id
        return impl

    @classmethod
    async def lookup_by_name(
        cls, client: _Client, req: api_pb2.SandboxGetFromNameRequest
    ) -> api_pb2.SandboxGetFromNameResponse:
        return await client._stub.SandboxGetFromName(req)

    @classmethod
    async def lookup_by_id(cls, client: _Client, sandbox_id: str) -> "SandboxImpl":
        resp = await client._stub.SandboxWait(api_pb2.SandboxWaitRequest(sandbox_id=sandbox_id, timeout=0))
        impl = cls(client, sandbox_id)
        apply_wait_response(impl, resp)
        return impl

    @classmethod
    async def list_page(cls, client: _Client, req: api_pb2.SandboxListRequest) -> api_pb2.SandboxListResponse:
        return await client._stub.SandboxList(req)

    @classmethod
    async def restore_from_memory_snapshot(
        cls,
        snapshot: _SandboxSnapshot,
        client: _Client,
        *,
        override_type: "api_pb2.SandboxRestoreRequest.SandboxNameOverrideType.ValueType",
        name_override: str | None,
    ) -> "SandboxImpl":
        restore_req = api_pb2.SandboxRestoreRequest(
            snapshot_id=snapshot.object_id,
            sandbox_name_override_type=override_type,
        )
        if name_override is not None:
            restore_req.sandbox_name_override = name_override
        # Pin the restored Sandbox to a specific worker when MODAL_WORKER_ID is
        # set.
        if worker_id := config.get("worker_id"):
            restore_req.worker_id = worker_id
        restore_resp: api_pb2.SandboxRestoreResponse = await client._stub.SandboxRestore(restore_req)

        impl = await cls.lookup_by_id(client, restore_resp.sandbox_id)

        task_id_req = api_pb2.SandboxGetTaskIdRequest(
            sandbox_id=restore_resp.sandbox_id, wait_until_ready=True, timeout=55.0
        )
        resp = await client._stub.SandboxGetTaskId(task_id_req)
        if resp.task_result.status not in [
            api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED,
            api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
        ]:
            raise ExecutionError(resp.task_result.exception)
        return impl

    def make_stdio(self, check_open: Callable[[], None]) -> tuple[StreamReader[str], StreamReader[str], StreamWriter]:
        sandbox_id = self.sandbox_id
        stdout: StreamReader[str] = StreamReader(
            _StreamReaderThroughServerParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                object_id=sandbox_id,
                get_client=self._get_client,
            ),
            by_line=True,
        )
        stderr: StreamReader[str] = StreamReader(
            _StreamReaderThroughServerParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDERR,
                object_id=sandbox_id,
                get_client=self._get_client,
            ),
            by_line=True,
        )
        stdin = StreamWriter(_StreamWriterThroughServerParams(object_id=sandbox_id, get_client=self._get_client))
        return stdout, stderr, stdin

    async def _snapshot_memory(self) -> str:
        client = await self._get_client()
        await self.get_task_id()
        snap_req = api_pb2.SandboxSnapshotRequest(sandbox_id=self.sandbox_id)
        snap_resp = await client._stub.SandboxSnapshot(snap_req)

        snapshot_id = snap_resp.snapshot_id

        # wait for the snapshot to succeed. this is implemented as a second idempotent rpc
        # because the snapshot itself may take a while to complete.
        wait_req = api_pb2.SandboxSnapshotWaitRequest(snapshot_id=snapshot_id, timeout=55.0)
        wait_resp = await client._stub.SandboxSnapshotWait(wait_req)
        if wait_resp.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
            raise ExecutionError(wait_resp.result.exception)
        return snapshot_id

    async def snapshot_memory(self) -> _SandboxSnapshot:
        client = await self._get_client()
        snapshot_id = await self._snapshot_memory()
        return new_snapshot_handle(client, snapshot_id, api_pb2.SandboxSnapshotHandleMetadata(is_v2=False))

    async def get_tags(self) -> dict[str, str]:
        client = await self._get_client()
        req = api_pb2.SandboxTagsGetRequest(sandbox_id=self.sandbox_id)
        resp = await client._stub.SandboxTagsGet(req)
        return tags_from_protos(resp.tags)

    async def set_tags(self, tags: dict[str, str]) -> None:
        client = await self._get_client()
        tags_list = tag_protos(tags)
        stub = client._stub
        req = api_pb2.SandboxTagsSetRequest(
            environment_name=_get_environment_name(),
            sandbox_id=self.sandbox_id,
            tags=tags_list,
        )
        await stub.SandboxTagsSet(req)

    async def set_name(self, name: str) -> None:
        raise InvalidError("Sandbox._experimental_set_name() is only supported for V2 sandboxes")

    async def create_connect_token(
        self, req: api_pb2.SandboxCreateConnectTokenRequest
    ) -> api_pb2.SandboxCreateConnectTokenResponse:
        client = await self._get_client()
        return await client._stub.SandboxCreateConnectToken(req)

    async def terminate(self) -> None:
        client = await self._get_client()
        req = api_pb2.SandboxTerminateRequest(sandbox_id=self.sandbox_id)
        await client._stub.SandboxTerminate(req)

    async def pause(self) -> None:
        raise NotImplementedError(_PERSISTENT_ONLY_MESSAGE.format("_experimental_pause()"))

    async def resume(self) -> None:
        raise NotImplementedError(_PERSISTENT_ONLY_MESSAGE.format("_experimental_resume()"))

    async def snapshot_filesystem(self, timeout: int, *, ttl: int | None, container_id: str = "") -> _Image:
        client = await self._get_client()
        if not container_id and os.environ.get("MODAL_USE_LEGACY_FILESYSTEM_SNAPSHOT") == "1":
            if ttl != DEFAULT_SNAPSHOT_TTL:
                raise InvalidError("ttl is not supported with MODAL_USE_LEGACY_FILESYSTEM_SNAPSHOT")
            return await self._legacy_snapshot_filesystem(timeout)
        return await snapshot_filesystem_via_tcr(self, client, timeout, ttl=ttl, container_id=container_id)

    async def _legacy_snapshot_filesystem(self, timeout: int) -> _Image:
        client = await self._get_client()
        await self.get_task_id()
        req = api_pb2.SandboxSnapshotFsRequest(sandbox_id=self.sandbox_id, timeout=timeout)
        resp = await client._stub.SandboxSnapshotFs(req)

        if resp.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
            raise ExecutionError(resp.result.exception)

        return _Image._new_hydrated(resp.image_id, client, resp.image_metadata)

    def validate_sidecar_create(
        self,
        *,
        cloud_bucket_mounts: Sequence[tuple[str, _CloudBucketMount]],
        has_proxy: bool,
        include_oidc_identity_token: bool,
    ) -> None:
        if cloud_bucket_mounts:
            raise InvalidError(
                "CloudBucketMount is not supported in sidecars of V1 Sandboxes. A Sandbox is V1 when it has a GPU "
                "or a PTY, or when MODAL_SANDBOX_V2=0 is set; contact Modal support for more information."
            )
        if has_proxy:
            raise InvalidError("Sandbox._experimental_sidecars.create(proxy=...) is not supported for GPU Sandboxes.")
        if include_oidc_identity_token:
            raise InvalidError("include_oidc_identity_token is not supported for GPU Sandboxes.")

    async def close(self) -> None:
        await self._command_router.close()

    def ensure_attached(self) -> None:
        self._command_router.ensure_open()

    async def wait(self, raise_on_termination: bool) -> None:
        client = await self._get_client()
        req, poll_retry = wait_poll_request(self.sandbox_id)
        while True:
            resp = await client._stub.SandboxWait(req, retry=poll_retry)
            apply_wait_response(self, resp)
            if resp.result.status:
                raise_for_wait_result(self.sandbox_id, resp.result, raise_on_termination)
                return

    async def poll(self) -> None:
        client = await self._get_client()
        req = api_pb2.SandboxWaitRequest(sandbox_id=self.sandbox_id, timeout=0)
        resp = await client._stub.SandboxWait(req)
        apply_wait_response(self, resp)

    async def tunnels(self, timeout: int) -> dict[int, Tunnel]:
        client = await self._get_client()
        if self._tunnels:
            return self._tunnels
        req = api_pb2.SandboxGetTunnelsRequest(sandbox_id=self.sandbox_id, timeout=timeout)
        resp = await client._stub.SandboxGetTunnels(req)
        self._tunnels = tunnels_from_response(resp)
        return self._tunnels

    async def get_task_id(
        self, raise_if_task_complete: bool = False, timeout: float | None = None, retry_transient: bool = False
    ) -> str:
        client = await self._get_client()
        deadline = task_id_poll_deadline(timeout)
        req = api_pb2.SandboxGetTaskIdRequest(sandbox_id=self.sandbox_id)
        while not self._task_id:
            try:
                resp = await client._stub.SandboxGetTaskId(req)
            except (ServiceError, InternalError, ConnectionError) as exc:
                raise_unless_retryable_task_id_error(exc, retry_transient=retry_transient, deadline=deadline)
            else:
                self._task_id = task_id_from_response(
                    resp, raise_if_task_complete=raise_if_task_complete, deadline=deadline
                )
                if self._task_id:
                    continue
            await asyncio.sleep(_TASK_ID_POLL_INTERVAL)
        return self._task_id

    async def resolve_task_id_for_logs(self) -> str:
        client = await self._get_client()
        if self._task_id:
            return self._task_id
        req = api_pb2.SandboxGetTaskIdRequest(sandbox_id=self.sandbox_id)
        return task_id_for_logs(await client._stub.SandboxGetTaskId(req))

    async def get_exit_snapshot(self, timeout: float | None) -> _Image:
        client = await self._get_client()
        poll = ExitSnapshotPoll(self.sandbox_id, timeout)
        while True:
            req, retry = poll.next_request()
            try:
                resp = await client._stub.SandboxGetExitSnapshot(req, retry=retry)
            except RetryTimeoutError:
                raise poll.timeout_error() from None
            if (image := poll.handle_response(resp, client)) is not None:
                return image

    async def get_command_router_client(self, task_id: str) -> TaskCommandRouterClient:
        return await self._command_router.get(task_id)

    async def _open_command_router_client(self, task_id: str) -> TaskCommandRouterClient:
        client = await self._get_client()
        fetch_access = command_router_access_fetcher(client, task_id)
        try:
            return await TaskCommandRouterClient.connect(client, task_id, fetch_access)
        except ConflictError as e:
            raise NotFoundError(str(e)) from e

    async def wait_until_ready(self, timeout: int) -> None:
        await wait_until_ready_via_tcr(self, timeout)

    async def set_network_access(self, network_access: api_pb2.NetworkAccess) -> None:
        await set_network_access_via_tcr(self, network_access)

    async def set_outbound_policy(self, outbound_policy: _OutboundPolicy) -> None:
        client = await self._get_client()
        req = await set_outbound_policy_request(client, self.sandbox_id, outbound_policy)
        await retry_while_starting(lambda: client._stub.SandboxSetOutboundPolicy(req))

    async def update_experimental_outbound_policy(self, outbound_policy: _ExperimentalOutboundPolicy) -> None:
        client = await self._get_client()
        await update_experimental_outbound_policy_via_tcr(self, client, outbound_policy)

    async def mount_image(
        self,
        path: PurePosixPath | str,
        image_id: str,
        *,
        encryption_key: bytes | None,
        container_id: str = "",
    ) -> None:
        await mount_image_via_tcr(self, path, image_id, encryption_key=encryption_key, container_id=container_id)

    async def unmount_image(self, path: PurePosixPath | str, *, container_id: str = "") -> None:
        await unmount_image_via_tcr(self, path, container_id=container_id)

    async def snapshot_directory(
        self,
        path: PurePosixPath | str,
        *,
        timeout: int,
        ttl: int | None,
        encryption_key: bytes | None,
        container_id: str = "",
    ) -> _Image:
        client = await self._get_client()
        return await snapshot_directory_via_tcr(
            self, client, path, timeout=timeout, ttl=ttl, encryption_key=encryption_key, container_id=container_id
        )

    async def reload_volumes(self, *, timeout: int, container_id: str = "") -> None:
        # Reload one container's Volumes; empty `container_id` targets the main container.
        await reload_volumes_via_tcr(self, timeout=timeout, container_id=container_id)

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
        client = await self._get_client()
        return await exec_via_tcr(
            self,
            client,
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
        return await create_sidecar_via_tcr(
            self,
            *args,
            name=name,
            image_id=image_id,
            secret_sources=secret_sources,
            workdir=workdir,
            volume_mounts=volume_mounts,
            network_access=network_access,
            pty_info=pty_info,
            memory_reserve_consume_mib=memory_reserve_consume_mib,
        )

    async def get_sidecar(self, name: str, include_terminated: bool) -> sr_pb2.TaskContainerInfo:
        return await get_sidecar_via_tcr(self, name, include_terminated)

    async def list_sidecars(self, include_terminated: bool) -> list[sr_pb2.TaskContainerInfo]:
        return await list_sidecars_via_tcr(self, include_terminated)

    async def wait_sidecar(self, container_id: str, timeout: float) -> api_pb2.GenericResult | None:
        return await wait_sidecar_via_tcr(self, container_id, timeout)

    async def terminate_sidecar(self, container_id: str) -> None:
        await terminate_sidecar_via_tcr(self, container_id)


if TYPE_CHECKING:
    _impl_check: type[SandboxImpl] = SandboxImplV1
