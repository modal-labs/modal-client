# Copyright Modal Labs 2026
import asyncio
import builtins
import os
import typing
import uuid
from collections.abc import Collection, Sequence
from pathlib import PurePosixPath
from typing import Literal, overload

import modal.sandbox._sandbox
from modal.cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from modal.secret import _local_secret_env, _resolvable_secrets, _secret_sources
from modal.volume import _Volume, _volume_to_mount_proto
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._image import _Image
from .._load_context import LoadContext
from .._resolver import Resolver
from .._utils.async_utils import TaskContext
from .._utils.mount_utils import (
    validate_volumes,
    validate_volumes_by_object_id,
)
from ..config import config
from ..container_process import _ContainerProcess
from ..exception import (
    InvalidError,
    SandboxTerminatedError,
)
from ..proxy import _Proxy
from ..secret import _Secret
from ..stream_type import StreamType
from ._common import (
    _build_outbound_network_access,
    _image_id_for_mount,
    _result_returncode,
    _ttl_to_wire_ttl,
    _validate_exec_args,
    _validate_experimental_encryption_key,
    _validate_sandbox_env,
)
from ._filesystem import _SandboxFilesystem
from ._task_command_router_client import TaskCommandRouterClient


class _SidecarContainer:
    """Handle to an additional container running in a Sandbox."""

    _result: api_pb2.GenericResult | None
    _filesystem: _SandboxFilesystem | None

    def __init__(
        self,
        sandbox: "modal.sandbox._sandbox._Sandbox",
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
    def _from_container_info(
        sandbox: "modal.sandbox._sandbox._Sandbox", container_info: sr_pb2.TaskContainerInfo
    ) -> "_SidecarContainer":
        result = container_info.result if container_info.HasField("result") else None
        return _SidecarContainer(sandbox, container_info.container_id, container_info.container_name, result)

    async def _get_command_router(self) -> tuple[str, "TaskCommandRouterClient"]:
        """Get task ID and command router client."""
        task_id = await self._sandbox._get_task_id()
        command_router_client = await self._sandbox._get_command_router_client(task_id)
        return task_id, command_router_client

    async def snapshot_filesystem(
        self,
        timeout: int = 55,
        *,
        ttl: int | None = 30 * 24 * 3600,
    ) -> _Image:
        """Snapshot this Sidecar container's filesystem.

        Args:
            timeout:
                Maximum time in seconds to wait for the snapshot operation.
            ttl:
                The resulting Image is retained for `ttl` seconds (default: 30 days). Pass `ttl=None` to retain
                the image indefinitely.

        Returns:
            An [`Image`](https://modal.com/docs/sdk/py/latest/Image) containing a snapshot of this Sidecar's
            filesystem.
        """
        return await self._sandbox._snapshot_filesystem(
            timeout,
            ttl=ttl,
            container_id=self._container_id,
        )

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
        """Namespace for Sandbox filesystem APIs."""
        if self._filesystem is None:
            self._filesystem = _SandboxFilesystem(self)
        return self._filesystem

    async def mount_image(
        self,
        path: PurePosixPath | str,
        image: _Image,
        *,
        _experimental_encryption_key: bytes | None = None,
    ) -> None:
        """Mount an Image at a specified path in this Sidecar container.

        `path` should be a directory that is **not** the root path (`/`). If the path doesn't exist,
        it will be created. If it exists and contains data, the previous directory will be replaced
        by the mount.

        The `image` argument supports any Image that has an object ID, including:
        - Images built using `image.build()`
        - Images referenced by ID, e.g. `Image.from_id(...)`
        - Filesystem/directory snapshots, e.g. created by `.snapshot_directory()` or `.snapshot_filesystem()`
        - Empty images created with `Image.from_scratch()`

        Args:
            path: Absolute mount point directory inside the Sidecar container (not `/`).
            image: Image to mount at `path` (must be built, referenced by ID, or snapshot-based as described above).

        Examples:
            ```py notest
            sidecar_1.mount_image("/workspace", modal.Image.from_scratch())
            workspace_snapshot = sidecar_1.snapshot_directory("/workspace")

            # You can later mount this snapshot in another Sidecar:
            sidecar_2.mount_image("/workspace", workspace_snapshot)
            sidecar_2.filesystem.list_files("/workspace")
            ```
        """
        image_id = _image_id_for_mount(image, "SidecarContainer.mount_image")

        posix_path = PurePosixPath(path)
        if not posix_path.is_absolute():
            raise InvalidError(f"Mount path must be absolute; got: {posix_path}")

        task_id, command_router_client = await self._get_command_router()
        await command_router_client.mount_image(
            sr_pb2.TaskMountDirectoryRequest(
                task_id=task_id,
                path=posix_path.as_posix().encode("utf8"),
                image_id=image_id,
                customer_supplied_encryption_key=_validate_experimental_encryption_key(_experimental_encryption_key),
                container_id=self._container_id,
            )
        )

    async def unmount_image(self, path: PurePosixPath | str) -> None:
        """Unmount a previously mounted Image from this Sidecar container.

        `path` must be the exact mount point that was passed to `.mount_image()`.
        After unmounting, the underlying Sidecar filesystem at that path becomes
        visible again.

        Args:
            path: Absolute mount point directory to unmount.

        """
        posix_path = PurePosixPath(path)
        if not posix_path.is_absolute():
            raise InvalidError(f"Unmount path must be absolute; got: {posix_path}")

        task_id, command_router_client = await self._get_command_router()
        await command_router_client.unmount_image(
            sr_pb2.TaskUnmountDirectoryRequest(
                task_id=task_id,
                path=posix_path.as_posix().encode("utf8"),
                container_id=self._container_id,
            )
        )

    async def snapshot_directory(
        self,
        path: PurePosixPath | str,
        *,
        timeout: int = 55,
        ttl: int | None = 30 * 24 * 3600,
        _experimental_encryption_key: bytes | None = None,
    ) -> _Image:
        """Snapshot a directory in this Sidecar container, creating a new Image with its content.

        `timeout` If the snapshot does not return within that window, the call is cancelled
        and `modal.exception.TimeoutError` is raised.

        `ttl` The resulting Image is retained for `ttl` seconds (default: 30 days).
        Pass `ttl=None` to retain the Image indefinitely.

        The returned Image can be used anywhere an Image is accepted, including
        as a mount or as the base filesystem for another container.

        Args:
            path: Absolute path of the directory inside the Sidecar container to snapshot.

        Returns:
            An `Image` containing the directory contents.

        Examples:
            ```py notest
            workspace_snapshot = sidecar_1.snapshot_directory("/workspace")

            # You can later mount this snapshot in another Sidecar:
            sidecar_2.mount_image("/workspace", workspace_snapshot)
            sidecar_2.filesystem.list_files("/workspace")
            ```
        """
        wire_ttl_seconds = _ttl_to_wire_ttl(ttl)
        posix_path = PurePosixPath(path)
        if not posix_path.is_absolute():
            raise InvalidError(f"Snapshot path must be absolute; got: {posix_path}")

        task_id, command_router_client = await self._get_command_router()
        response = await command_router_client.snapshot_directory(
            sr_pb2.TaskSnapshotDirectoryRequest(
                task_id=task_id,
                path=posix_path.as_posix().encode("utf8"),
                snapshot_id=str(uuid.uuid4()),
                ttl_seconds=wire_ttl_seconds,
                customer_supplied_encryption_key=_validate_experimental_encryption_key(_experimental_encryption_key),
                container_id=self._container_id,
            ),
            timeout=float(timeout),
        )
        return _Image._new_hydrated(response.image_id, self._sandbox._client, None)

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

    async def reload_volumes(self, *, timeout: int = 55) -> None:
        """Reload all Volumes mounted in this sidecar container.

        EXPERIMENTAL: the API is subject to change.

        Blocks until the reload completes, or raises `modal.exception.TimeoutError` on timeout (the reload
        may still complete in the background).

        Args:
            timeout: Defaults to 55 seconds.
        """
        await self._sandbox._reload_volumes(timeout=timeout, container_id=self._container_id)


_MAIN_CONTAINER_NAME: str = "main"

_CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR = "MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE"


def _use_control_plane_sidecar_create(is_v2: bool) -> bool:
    """Whether a sidecar create request goes to the Modal server rather than over the Sandbox connection.

    V2 Sandboxes do unless opted out via the `use_control_plane_sidecar_create` config
    setting (`MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE=0`). V1 Sandboxes always create
    sidecars over the Sandbox connection.
    """
    return is_v2 and config.get("use_control_plane_sidecar_create")


class _SidecarManager:
    """Creates and manages sidecar containers in a Sandbox."""

    def __init__(self, sandbox: "modal.sandbox._sandbox._Sandbox") -> None:
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
        volumes: dict[str | os.PathLike, _Volume | _CloudBucketMount] | None = None,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        include_oidc_identity_token: bool = False,
        proxy: _Proxy | None = None,
        pty: bool = False,
        experimental_memory_reserve_consume_mib: int | None = None,
    ) -> _SidecarContainer:
        """Create a sidecar container running alongside the Sandbox's main container.

        Sidecar containers share the Sandbox's lifecycle but run their own Image and command. They
        can be used to run auxiliary processes, such as a database or a service the main container
        depends on.

        The sidecar's outbound network policy is independent of the main container's and defaults to
        open network access. To restrict it, pass `outbound_cidr_allowlist` and/or
        `outbound_domain_allowlist`. To block all external egress while keeping connectivity to the
        main container, pass an empty allowlist (`outbound_cidr_allowlist=[]`); a fully network-blocked
        sidecar is not supported because it would have no IP and could not reach the main container.

        Args:
            *args: Command and arguments to run inside the sidecar container.
            name: Unique name for the sidecar container. The name ``"main"`` is reserved.
            image: Image to run the sidecar container with. Must be a pre-built or referenced Image.
            env: Environment variables to set in the sidecar container.
            secrets: Secrets to inject as environment variables in the sidecar container.
            workdir: Working directory for the command; must be absolute if set.
            volumes: Mapping of mount paths to `Volume` or `CloudBucketMount` objects to mount in the
                sidecar container. Cloud bucket mounts are not supported for GPU Sandboxes.
            outbound_cidr_allowlist: If set, restrict the sidecar's outbound traffic to these CIDR
                blocks. An empty list blocks all external egress while preserving connectivity to the
                main container.
            outbound_domain_allowlist: If set, restrict the sidecar's outbound TLS connections (port
                443) to these SNI domains. Supports wildcards like ``*.example.com``.
            include_oidc_identity_token: If True, the sidecar receives a MODAL_IDENTITY_TOKEN env var for
                OIDC-based auth (e.g. to AWS, GCP). The token identifies the sidecar container itself,
                not the main container. Not supported for GPU Sandboxes.
            proxy: Reference to a Modal Proxy to use in front of this sidecar. Not supported for GPU
                Sandboxes.
            pty: Whether to enable PTY for the sidecar container.
            experimental_memory_reserve_consume_mib: Memory, in MiB, this sidecar consumes from the Sandbox's
                sidecar memory reserve (the experimental `vm_sidecar_memory_reserve_mib` option).
                Unset consumes whatever is left of the reserve; creation fails if the request exceeds
                what is left. Ignored by Sandboxes without a reserve.

        Returns:
            A `SidecarContainer` handle for the running container.
        """
        if name == _MAIN_CONTAINER_NAME:
            raise InvalidError(f"The name {_MAIN_CONTAINER_NAME!r} is reserved for the sandbox's main container.")
        if workdir is not None and not workdir.startswith("/"):
            raise InvalidError(f"workdir must be an absolute path, got: {workdir}")
        _validate_exec_args(args)
        if experimental_memory_reserve_consume_mib is not None and experimental_memory_reserve_consume_mib <= 0:
            raise InvalidError(
                "experimental_memory_reserve_consume_mib must be a positive number of MiB, "
                f"got: {experimental_memory_reserve_consume_mib}"
            )

        via_control_plane = _use_control_plane_sidecar_create(self._sandbox._is_v2)
        mounted_objects = validate_volumes(volumes if volumes is not None else {})
        cloud_bucket_mounts = [(path, v) for path, v in mounted_objects if isinstance(v, _CloudBucketMount)]
        validated_volumes = [(path, v) for path, v in mounted_objects if isinstance(v, _Volume)]
        if cloud_bucket_mounts and not self._sandbox._is_v2:
            raise InvalidError(
                "CloudBucketMount is not supported in sidecars of V1 Sandboxes. A Sandbox is V1 when it has a GPU, "
                "network file systems or a PTY, or when MODAL_SANDBOX_V2=0 is set; contact Modal support for more "
                "information."
            )
        if cloud_bucket_mounts and not via_control_plane:
            raise InvalidError(
                "CloudBucketMount is not supported in sidecars when MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE=0 is set; "
                "unset it to use cloud bucket mounts."
            )
        if proxy is not None and not self._sandbox._is_v2:
            raise InvalidError("Sandbox._experimental_sidecars.create(proxy=...) is not supported for GPU Sandboxes.")
        if proxy is not None and not via_control_plane:
            raise InvalidError(
                "Sandbox._experimental_sidecars.create(proxy=...) is not supported when "
                f"{_CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR}=0 is set; unset it to use a proxy."
            )

        if include_oidc_identity_token and not self._sandbox._is_v2:
            raise InvalidError("include_oidc_identity_token is not supported for GPU Sandboxes.")
        if include_oidc_identity_token and not via_control_plane:
            raise InvalidError(
                f"include_oidc_identity_token is not supported when {_CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR}=0 is set; "
                "unset it to use it."
            )

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

        secrets = list(secrets or [])
        resolvable_secrets = _resolvable_secrets(secrets)
        _validate_sandbox_env(_local_secret_env(secrets) | (env or {}))

        bucket_credential_secrets = [
            mount.secret
            for _, mount in cloud_bucket_mounts
            if mount.secret is not None and not mount.secret._is_ephemeral
        ]
        resolver = Resolver()
        async with TaskContext() as tc:
            load_context = LoadContext(client=self._sandbox._client, task_context=tc)
            dependencies = [
                *resolvable_secrets,
                *(volume for _, volume in validated_volumes),
                *bucket_credential_secrets,
                *([proxy] if proxy is not None else []),
            ]
            await asyncio.gather(
                *(resolver.load(dependency, load_context) for dependency in dependencies if not dependency._is_hydrated)
            )

        secret_sources = _secret_sources(secrets, env)

        # Validate that the same volume (by object_id) isn't mounted at multiple paths. This relies on
        # the volumes being hydrated above, since it compares object_ids.
        validate_volumes_by_object_id(validated_volumes)

        # Relies on dicts being ordered (true as of Python 3.6).
        volume_mounts = [_volume_to_mount_proto(path, volume) for path, volume in validated_volumes]

        network_access = _build_outbound_network_access(False, outbound_cidr_allowlist, outbound_domain_allowlist)
        pty_info = self._sandbox._default_pty_info() if pty else None

        if via_control_plane:
            cloud_bucket_mount_protos, cloud_bucket_credentials = cloud_bucket_mounts_to_proto(
                cloud_bucket_mounts, split_ephemeral_credentials=True
            )
            definition = api_pb2.Sandbox(
                entrypoint_args=list(args),
                image_id=image.object_id,
                secret_ids=[secret.object_id for secret in resolvable_secrets],
                workdir=workdir,
                volume_mounts=volume_mounts,
                cloud_bucket_mounts=cloud_bucket_mount_protos,
                network_access=network_access,
                include_oidc_identity_token=include_oidc_identity_token,
                proxy_id=(proxy.object_id if proxy else None),
                pty_info=pty_info,
                resources=(
                    api_pb2.Resources(memory_mb=experimental_memory_reserve_consume_mib)
                    if experimental_memory_reserve_consume_mib is not None
                    else None
                ),
            )
            create_req = api_pb2.SandboxContainerCreateV2Request(
                sandbox_id=self._sandbox.object_id,
                container_name=name,
                definition=definition,
                secret_sources=secret_sources,
                cloud_bucket_mount_credentials=cloud_bucket_credentials,
            )
            client = self._sandbox._client
            create_resp = await client._stub.SandboxContainerCreateV2(
                create_req, metadata=await self._sandbox._v2_metadata(client)
            )
        else:
            task_id, command_router_client = await self._get_command_router()
            create_resp = await command_router_client.container_create(
                sr_pb2.TaskContainerCreateRequest(
                    task_id=task_id,
                    container_name=name,
                    image_id=image.object_id,
                    args=list(args),
                    workdir=workdir or "",
                    secret_sources=secret_sources,
                    volume_mounts=volume_mounts,
                    network_access=network_access,
                    pty_info=pty_info,
                    memory_reserve_consume_mib=experimental_memory_reserve_consume_mib,
                )
            )

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
