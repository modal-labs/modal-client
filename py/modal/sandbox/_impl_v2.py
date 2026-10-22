# Copyright Modal Labs 2026
"""Sandbox implementation for V2 Sandboxes, which are managed through the `*V2` control-plane RPCs."""

import asyncio
import logging
import time
import uuid
import weakref
from collections.abc import Callable, Collection, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._image import _Image
from .._outbound_policy import _OutboundPolicy
from .._resolver import Resolver
from .._tunnel import Tunnel
from .._utils.grpc_utils import RetryTimeoutError
from .._utils.name_utils import check_object_name
from ..client import _Client
from ..cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from ..config import config, logger
from ..container_process import _ContainerProcess
from ..exception import ConflictError, InternalError, InvalidError, NotFoundError, ServiceError
from ..io_streams import (
    StreamReader,
    StreamWriter,
    _StreamReaderThroughSandboxCommandRouterParams,
    _StreamWriterThroughCommandRouterSandboxParams,
)
from ..secret import _resolvable_secrets, _Secret, _secret_sources
from ..stream_type import StreamType
from ._common import (
    _PERSISTENT_ONLY_MESSAGE,
    _TASK_ID_POLL_INTERVAL,
    ExitSnapshotPoll,
    SandboxCreateArgs,
    SandboxCreateCommon,
    SandboxImpl,
    _format_sandbox_create_timing_log,
    _validate_sandbox_env,
    command_router,
    create_sidecar_via_tcr,
    exec_via_tcr,
    get_sidecar_via_tcr,
    list_sidecars_via_tcr,
    mount_image_via_tcr,
    new_snapshot_handle,
    raise_for_wait_result,
    raise_unless_retryable_task_id_error,
    reload_volumes_via_tcr,
    sandbox_create_context,
    set_network_access_via_tcr,
    snapshot_directory_via_tcr,
    snapshot_filesystem_via_tcr,
    tag_protos,
    tags_from_protos,
    task_id_for_logs,
    task_id_from_response,
    task_id_poll_deadline,
    terminate_sidecar_via_tcr,
    tunnels_from_protos,
    tunnels_from_response,
    unmount_image_via_tcr,
    update_outbound_policy_via_tcr,
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


_CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR = "MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE"


async def auth_metadata(client: _Client) -> list[tuple[str, str]]:
    assert client._auth_token_manager
    auth_token = await client._auth_token_manager.get_token()
    return [("x-modal-auth-token", auth_token)]


def command_router_access_fetcher(
    client: _Client,
    *,
    sandbox_id: str | None = None,
    task_id: str | None = None,
    sandbox_token: str | None = None,
) -> CommandRouterAccessFetcher:
    """Return a fetcher of command router access for a V2 Sandbox, identified by its ID or its task's ID.

    The fetcher holds plain values rather than the Sandbox, so a router client doesn't keep it alive.
    """
    if sandbox_id is not None:
        req = api_pb2.SandboxGetCommandRouterAccessRequest(sandbox_id=sandbox_id)
    elif task_id is not None:
        req = api_pb2.SandboxGetCommandRouterAccessRequest(task_id=task_id)
    else:
        raise ValueError("Either sandbox_id or task_id must be provided")

    async def fetch() -> api_pb2.CommandRouterAccess:
        metadata = await auth_metadata(client)
        if sandbox_token:
            metadata.append(("x-modal-sandbox-token", sandbox_token))
        resp = await client._stub.SandboxGetCommandRouterAccess(req, metadata=metadata)
        return api_pb2.CommandRouterAccess(url=resp.url, jwt=resp.jwt)

    return fetch


class SandboxImplV2:
    _sandbox_token: str | None
    _init_command_router_access: api_pb2.CommandRouterAccess | None
    _tunnels: dict[int, Tunnel] | None
    _task_id: str | None
    _command_router: LazyCommandRouterClient

    def __init__(
        self,
        client: _Client,
        sandbox_id: str,
        *,
        result: api_pb2.GenericResult | None = None,
        task_id: str | None = None,
        sandbox_token: str | None = None,
        command_router_access: api_pb2.CommandRouterAccess | None = None,
        tunnels: dict[int, Tunnel] | None = None,
    ) -> None:
        """`command_router_access` is only good for the first router connection; later ones fetch fresh access."""
        self.client = client
        self.sandbox_id = sandbox_id
        self.result = result
        self._task_id = task_id
        self._command_router = LazyCommandRouterClient(self._open_command_router_client)
        self._sandbox_token = sandbox_token
        self._init_command_router_access = command_router_access
        self._tunnels = tunnels

    @property
    def _client(self) -> _Client:
        return self.client

    def clone(self) -> "SandboxImplV2":
        return type(self)(self.client, self.sandbox_id, result=self.result)

    @classmethod
    async def create(
        cls, create_args: SandboxCreateArgs, *, app: "modal.app._App | None", client: _Client | None
    ) -> "tuple[SandboxImpl, api_pb2.SandboxHandleMetadata]":
        if create_args.gpu is not None or create_args.network_file_systems or create_args.pty_info is not None:
            raise InvalidError("GPUs, network file systems and PTY info are not supported by V2 Sandboxes")
        if create_args.block_network and create_args.i6pn:
            raise InvalidError(
                "`block_network` disables all networking, including i6pn. To keep i6pn while blocking "
                "public egress, use an empty outbound allowlist (`outbound_cidr_allowlist=[]`) instead."
            )

        ephemeral_env: dict[str, str] = {}
        if create_args.env:
            env_type_err = "the env argument to Sandbox must be a dict[str, str | None]"
            if not isinstance(create_args.env, dict):
                raise InvalidError(env_type_err)
            ephemeral_env = {k: v for k, v in create_args.env.items() if v is not None}
            if not all(isinstance(k, str) for k in ephemeral_env) or not all(
                isinstance(v, str) for v in ephemeral_env.values()
            ):
                raise InvalidError(env_type_err)
            _validate_sandbox_env(ephemeral_env)

        # V2 references persisted Secrets by id and sends ephemeral ones and the `env` dict inline.
        secrets = create_args.secrets or []
        secret_ids_to_load = _resolvable_secrets(secrets)

        common = SandboxCreateCommon(create_args, secrets=secret_ids_to_load)
        load_start = time.monotonic()
        async with sandbox_create_context(app, client) as (client, load_context):
            dep_timings = await common.load_dependencies(Resolver(), load_context, load_ephemeral_bucket_secrets=False)
            cloud_bucket_mount_protos, cloud_bucket_credentials = cloud_bucket_mounts_to_proto(
                common.cloud_bucket_mounts, split_ephemeral_credentials=True
            )
            definition = common.definition(cloud_bucket_mounts=cloud_bucket_mount_protos)

            create_req = api_pb2.SandboxCreateV2Request(
                app_id=load_context.app_id,
                definition=definition,
                secret_sources=_secret_sources(secrets, ephemeral_env),
                tags=tag_protos(create_args.tags),
                cloud_bucket_mount_credentials=cloud_bucket_credentials,
            )
            assert client._auth_token_manager
            auth_token = await client._auth_token_manager.get_token()
            rpc_start = time.monotonic()
            create_resp = await client._stub.SandboxCreateV2(create_req, metadata=[("x-modal-auth-token", auth_token)])
            rpc_elapsed = time.monotonic() - rpc_start
            sandbox_id = create_resp.sandbox_id
            impl = cls(
                client,
                sandbox_id,
                task_id=create_resp.task_id,
                sandbox_token=create_resp.sandbox_token or None,
                # An unset proto message reads as an empty one rather than None, so the
                # presence check is what distinguishes "no access" from "empty access".
                command_router_access=(
                    create_resp.command_router_access if create_resp.HasField("command_router_access") else None
                ),
                # Unencrypted tunnel endpoints are not known at create time. Only
                # cache a complete set so tunnels() fetches the rest otherwise.
                tunnels=(
                    tunnels_from_protos(create_resp.tunnels)
                    if len(create_resp.tunnels) == len(common.open_ports)
                    else None
                ),
            )

            if logger.isEnabledFor(logging.DEBUG):
                total_elapsed = time.monotonic() - load_start
                logger.debug(_format_sandbox_create_timing_log(sandbox_id, total_elapsed, rpc_elapsed, dep_timings))
        return impl, create_resp.metadata

    async def _metadata(self, client: _Client | None = None) -> list[tuple[str, str]]:
        metadata = await auth_metadata(client if client is not None else self.client)
        if self._sandbox_token:
            metadata.append(("x-modal-sandbox-token", self._sandbox_token))
        return metadata

    @classmethod
    async def lookup_by_name(
        cls, client: _Client, req: api_pb2.SandboxGetFromNameRequest
    ) -> api_pb2.SandboxGetFromNameResponse:
        return await client._stub.SandboxGetFromNameV2(req, metadata=await auth_metadata(client))

    @classmethod
    async def lookup_by_id(cls, client: _Client, sandbox_id: str) -> api_pb2.SandboxWaitResponse:
        req = api_pb2.SandboxWaitRequest(sandbox_id=sandbox_id, timeout=0)
        return await client._stub.SandboxWaitV2(req, metadata=await auth_metadata(client))

    @classmethod
    async def list_page(cls, client: _Client, req: api_pb2.SandboxListRequest) -> api_pb2.SandboxListResponse:
        # Lists both V1 and V2 Sandboxes.
        return await client._stub.SandboxListV2(req, metadata=await auth_metadata(client))

    @classmethod
    async def restore_from_memory_snapshot(
        cls,
        snapshot: _SandboxSnapshot,
        client: _Client,
        *,
        override_type: "api_pb2.SandboxRestoreRequest.SandboxNameOverrideType.ValueType",
        name_override: str | None,
    ) -> "tuple[SandboxImpl, api_pb2.SandboxHandleMetadata]":
        restore_req = api_pb2.SandboxRestoreV2Request(
            snapshot_id=snapshot.object_id,
            sandbox_name_override_type=override_type,
        )
        if name_override is not None:
            restore_req.sandbox_name_override = name_override
        if worker_id := config.get("worker_id"):
            restore_req.worker_id = worker_id

        restore_resp: api_pb2.SandboxRestoreV2Response = await client._stub.SandboxRestoreV2(
            restore_req, metadata=await auth_metadata(client)
        )

        lookup_resp = await cls.lookup_by_id(client, restore_resp.sandbox_id)
        impl = cls(
            client,
            restore_resp.sandbox_id,
            result=lookup_resp.result if lookup_resp.result.status else None,
            task_id=restore_resp.task_id,
            command_router_access=(
                restore_resp.command_router_access if restore_resp.HasField("command_router_access") else None
            ),
        )
        return impl, lookup_resp.metadata

    def make_stdio(self, check_open: Callable[[], None]) -> tuple[StreamReader[str], StreamReader[str], StreamWriter]:
        """Create stdio streams that read directly from the worker.

        Cheap to call eagerly: the router connection is opened lazily on first use.
        """
        weak_self = weakref.ref(self)

        async def resolve_router() -> tuple[str, TaskCommandRouterClient]:
            this = weak_self()
            if this is None:
                raise RuntimeError("Sandbox was garbage collected before its stdio reader connected")
            return await command_router(this)

        sandbox_id = self.sandbox_id
        stdout: StreamReader[str] = StreamReader(
            _StreamReaderThroughSandboxCommandRouterParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
                sandbox_id=sandbox_id,
                resolve_router=resolve_router,
            ),
            by_line=True,
        )
        stderr: StreamReader[str] = StreamReader(
            _StreamReaderThroughSandboxCommandRouterParams(
                file_descriptor=api_pb2.FILE_DESCRIPTOR_STDERR,
                sandbox_id=sandbox_id,
                resolve_router=resolve_router,
            ),
            by_line=True,
        )
        stdin = StreamWriter(
            _StreamWriterThroughCommandRouterSandboxParams(
                resolve_router=resolve_router,
                check_open=check_open,
            )
        )
        return stdout, stderr, stdin

    async def _snapshot_memory(self) -> str:
        task_id, command_router_client = await command_router(self)
        resp = await command_router_client.snapshot_memory(
            sr_pb2.TaskSnapshotMemoryRequest(task_id=task_id, idempotency_key=str(uuid.uuid4())),
            timeout=300.0,
        )
        return resp.snapshot_id

    async def snapshot_memory(self) -> _SandboxSnapshot:
        snapshot_id = await self._snapshot_memory()
        return new_snapshot_handle(self.client, snapshot_id, api_pb2.SandboxSnapshotHandleMetadata(is_v2=True))

    async def get_tags(self) -> dict[str, str]:
        req = api_pb2.SandboxTagsGetRequest(sandbox_id=self.sandbox_id)
        resp = await self.client._stub.SandboxTagsGetV2(req, metadata=await self._metadata())
        return tags_from_protos(resp.tags)

    async def set_tags(self, tags: dict[str, str]) -> None:
        tags_list = tag_protos(tags)
        req = api_pb2.SandboxTagsSetRequest(sandbox_id=self.sandbox_id, tags=tags_list)
        await self.client._stub.SandboxTagsSetV2(req, metadata=await self._metadata())

    async def set_name(self, name: str) -> None:
        check_object_name(name, "Sandbox")
        req = api_pb2.SandboxSetNameRequest(sandbox_id=self.sandbox_id, name=name)
        await self.client._stub.SandboxSetName(req, metadata=await self._metadata())

    async def create_connect_token(
        self, req: api_pb2.SandboxCreateConnectTokenRequest
    ) -> api_pb2.SandboxCreateConnectTokenResponse:
        return await self.client._stub.SandboxCreateConnectTokenV2(req, metadata=await self._metadata())

    async def terminate(self) -> None:
        req = api_pb2.SandboxTerminateRequest(sandbox_id=self.sandbox_id)
        await self.client._stub.SandboxTerminateV2(req, metadata=await self._metadata())

    async def pause(self) -> None:
        raise NotImplementedError(_PERSISTENT_ONLY_MESSAGE.format("_experimental_pause()"))

    async def resume(self) -> None:
        raise NotImplementedError(_PERSISTENT_ONLY_MESSAGE.format("_experimental_resume()"))

    def _create_sidecar_via_control_plane(self) -> bool:
        # Opt out via the `use_control_plane_sidecar_create` config setting to create
        # sidecars over the task command router instead.
        return config.get("use_control_plane_sidecar_create")

    def validate_sidecar_create(
        self,
        *,
        cloud_bucket_mounts: Sequence[tuple[str, _CloudBucketMount]],
        has_proxy: bool,
        include_oidc_identity_token: bool,
    ) -> None:
        if self._create_sidecar_via_control_plane():
            return
        if cloud_bucket_mounts:
            raise InvalidError(
                "CloudBucketMount is not supported in sidecars when MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE=0 is set; "
                "unset it to use cloud bucket mounts."
            )
        if has_proxy:
            raise InvalidError(
                "Sandbox._experimental_sidecars.create(proxy=...) is not supported when "
                f"{_CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR}=0 is set; unset it to use a proxy."
            )
        if include_oidc_identity_token:
            raise InvalidError(
                f"include_oidc_identity_token is not supported when {_CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR}=0 is set; "
                "unset it to use it."
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
        if not self._create_sidecar_via_control_plane():
            # TODO: Remove this path in favor of the control-plane creation below. Once it is gone,
            # `create_sidecar_via_tcr` is only used by V1 Sandboxes and can move into `SandboxImplV1`.
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

        cloud_bucket_mount_protos, cloud_bucket_credentials = cloud_bucket_mounts_to_proto(
            cloud_bucket_mounts, split_ephemeral_credentials=True
        )
        definition = api_pb2.Sandbox(
            entrypoint_args=list(args),
            image_id=image_id,
            secret_ids=list(secret_ids),
            workdir=workdir,
            volume_mounts=volume_mounts,
            cloud_bucket_mounts=cloud_bucket_mount_protos,
            network_access=network_access,
            include_oidc_identity_token=include_oidc_identity_token,
            proxy_id=proxy_id,
            pty_info=pty_info,
            resources=(
                api_pb2.Resources(memory_mb=memory_reserve_consume_mib)
                if memory_reserve_consume_mib is not None
                else None
            ),
        )
        create_req = api_pb2.SandboxContainerCreateV2Request(
            sandbox_id=self.sandbox_id,
            container_name=name,
            definition=definition,
            secret_sources=secret_sources,
            cloud_bucket_mount_credentials=cloud_bucket_credentials,
        )
        client = self.client
        create_resp = await client._stub.SandboxContainerCreateV2(create_req, metadata=await self._metadata(client))
        return create_resp.container_id, create_resp.container_name or name

    async def close(self) -> None:
        await self._command_router.close()

    async def wait(self, raise_on_termination: bool) -> None:
        client = self.client
        req, poll_retry = wait_poll_request(self.sandbox_id)
        while True:
            resp = await client._stub.SandboxWaitV2(req, retry=poll_retry, metadata=await self._metadata(client))
            if resp.result.status:
                self.result = resp.result
                raise_for_wait_result(self.sandbox_id, resp.result, raise_on_termination)
                return

    async def poll(self) -> None:
        req = api_pb2.SandboxWaitRequest(sandbox_id=self.sandbox_id, timeout=0)
        resp = await self.client._stub.SandboxWaitV2(req, metadata=await self._metadata())
        if resp.result.status:
            self.result = resp.result

    async def tunnels(self, timeout: int) -> dict[int, Tunnel]:
        if self._tunnels:
            return self._tunnels
        req = api_pb2.SandboxGetTunnelsRequest(sandbox_id=self.sandbox_id, timeout=timeout)
        resp = await self.client._stub.SandboxGetTunnelsV2(req, metadata=await self._metadata())
        self._tunnels = tunnels_from_response(resp)
        return self._tunnels

    async def get_task_id(
        self, raise_if_task_complete: bool = False, timeout: float | None = None, retry_transient: bool = False
    ) -> str:
        deadline = task_id_poll_deadline(timeout)
        req = api_pb2.SandboxGetTaskIdRequest(sandbox_id=self.sandbox_id)
        while not self._task_id:
            try:
                resp = await self.client._stub.SandboxGetTaskIdV2(req, metadata=await self._metadata())
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
        if self._task_id:
            return self._task_id
        req = api_pb2.SandboxGetTaskIdRequest(sandbox_id=self.sandbox_id)
        return task_id_for_logs(await self.client._stub.SandboxGetTaskIdV2(req, metadata=await self._metadata()))

    async def get_exit_snapshot(self, timeout: float | None) -> _Image:
        client = self.client
        poll = ExitSnapshotPoll(self.sandbox_id, timeout)
        while True:
            req, retry = poll.next_request()
            try:
                resp = await client._stub.SandboxGetExitSnapshotV2(
                    req, retry=retry, metadata=await self._metadata(client)
                )
            except RetryTimeoutError:
                raise poll.timeout_error() from None
            if (image := poll.handle_response(resp, client)) is not None:
                return image

    async def get_command_router_client(self, task_id: str) -> TaskCommandRouterClient:
        return await self._command_router.get(task_id)

    async def _open_command_router_client(self, task_id: str) -> TaskCommandRouterClient:
        # Access returned at create or restore is only good for the first connection.
        access, self._init_command_router_access = self._init_command_router_access, None
        fetch_access = command_router_access_fetcher(
            self.client, sandbox_id=self.sandbox_id, sandbox_token=self._sandbox_token
        )
        try:
            return await TaskCommandRouterClient.connect(self.client, task_id, fetch_access, access=access)
        except ConflictError as e:
            raise NotFoundError(str(e)) from e

    async def wait_until_ready(self, timeout: int) -> None:
        await wait_until_ready_via_tcr(self, timeout)

    async def set_network_access(self, network_access: api_pb2.NetworkAccess) -> None:
        await set_network_access_via_tcr(self, network_access)

    async def update_outbound_policy(self, outbound_policy: _OutboundPolicy) -> None:
        await update_outbound_policy_via_tcr(self, self.client, outbound_policy)

    async def snapshot_filesystem(self, timeout: int, *, ttl: int | None, container_id: str = "") -> _Image:
        return await snapshot_filesystem_via_tcr(self, self.client, timeout, ttl=ttl, container_id=container_id)

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
        return await snapshot_directory_via_tcr(
            self, self.client, path, timeout=timeout, ttl=ttl, encryption_key=encryption_key, container_id=container_id
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
        return await exec_via_tcr(
            self,
            self.client,
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

    async def get_sidecar(self, name: str, include_terminated: bool) -> sr_pb2.TaskContainerInfo:
        return await get_sidecar_via_tcr(self, name, include_terminated)

    async def list_sidecars(self, include_terminated: bool) -> list[sr_pb2.TaskContainerInfo]:
        return await list_sidecars_via_tcr(self, include_terminated)

    async def wait_sidecar(self, container_id: str, timeout: float) -> api_pb2.GenericResult | None:
        return await wait_sidecar_via_tcr(self, container_id, timeout)

    async def terminate_sidecar(self, container_id: str) -> None:
        await terminate_sidecar_via_tcr(self, container_id)


if TYPE_CHECKING:
    _impl_check: type[SandboxImpl] = SandboxImplV2
