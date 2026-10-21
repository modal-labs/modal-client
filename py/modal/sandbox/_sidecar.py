# Copyright Modal Labs 2026
import asyncio
import builtins
import typing
from collections.abc import Collection, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Literal, overload

import modal.sandbox._sandbox
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .._image import _Image
from .._load_context import LoadContext
from .._resolver import Resolver
from .._utils.async_utils import TaskContext
from .._utils.mount_utils import validate_volumes, validate_volumes_by_object_id
from ..cloud_bucket_mount import _CloudBucketMount
from ..container_process import _ContainerProcess
from ..exception import InvalidError, SandboxTerminatedError
from ..proxy import _Proxy
from ..secret import _local_secret_env, _resolvable_secrets, _Secret, _secret_sources
from ..stream_type import StreamType
from ..volume import _Volume, _volume_to_mount_proto
from ._common import (
    _build_outbound_network_access,
    _image_id_for_mount,
    _result_returncode,
    _validate_exec_args,
    _validate_sandbox_env,
    default_pty_info,
)
from ._filesystem import _SandboxFilesystem

_MAIN_CONTAINER_NAME: str = "main"


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
        self._sandbox._ensure_attached()
        return await self._sandbox._impl.snapshot_filesystem(
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
        pty_info = default_pty_info() if pty else None
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
        self._sandbox._ensure_attached()
        image_id = _image_id_for_mount(image, "SidecarContainer.mount_image")
        await self._sandbox._impl.mount_image(
            path, image_id, encryption_key=_experimental_encryption_key, container_id=self._container_id
        )

    async def unmount_image(self, path: PurePosixPath | str) -> None:
        """Unmount a previously mounted Image from this Sidecar container.

        `path` must be the exact mount point that was passed to `.mount_image()`.
        After unmounting, the underlying Sidecar filesystem at that path becomes
        visible again.

        Args:
            path: Absolute mount point directory to unmount.

        """
        self._sandbox._ensure_attached()
        await self._sandbox._impl.unmount_image(path, container_id=self._container_id)

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
        self._sandbox._ensure_attached()
        return await self._sandbox._impl.snapshot_directory(
            path,
            timeout=timeout,
            ttl=ttl,
            encryption_key=_experimental_encryption_key,
            container_id=self._container_id,
        )

    async def wait(self, raise_on_termination: bool = True) -> None:
        if self._result is not None and self._result.status != api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED:
            if self._result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED and raise_on_termination:
                raise SandboxTerminatedError()
            return

        self._sandbox._ensure_attached()
        while True:
            result = await self._sandbox._impl.wait_sidecar(self._container_id, timeout=10)
            if result is not None:
                self._result = result
                if result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED and raise_on_termination:
                    raise SandboxTerminatedError()
                return

    async def poll(self) -> int | None:
        if self._result is not None and self._result.status != api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED:
            return _result_returncode(self._result)

        self._sandbox._ensure_attached()
        result = await self._sandbox._impl.wait_sidecar(self._container_id, timeout=0)
        if result is not None:
            self._result = result
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
        self._sandbox._ensure_attached()
        await self._sandbox._impl.terminate_sidecar(self._container_id)
        if wait:
            await self.wait(raise_on_termination=False)
            return _result_returncode(self._result)
        return None

    async def reload_volumes(self, *, timeout: int = 55) -> None:
        """Reload all Volumes mounted in this sidecar container.

        EXPERIMENTAL: the API is subject to change.

        Blocks until the reload completes, or raises `modal.exception.TimeoutError` on timeout (the reload
        may still complete in the background).

        Args:
            timeout: Defaults to 55 seconds.
        """
        self._sandbox._ensure_attached()
        await self._sandbox._impl.reload_volumes(timeout=timeout, container_id=self._container_id)


class _SidecarManager:
    """Creates and manages sidecar containers in a Sandbox."""

    def __init__(self, sandbox: "modal.sandbox._sandbox._Sandbox") -> None:
        self._sandbox = sandbox

    async def create(
        self,
        *args: str,
        name: str,
        image: _Image,
        env: dict[str, str] | None = None,
        secrets: Collection[_Secret] | None = None,
        workdir: str | None = None,
        volumes: Mapping[str | PurePosixPath, _Volume | _CloudBucketMount] | None = None,
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
        self._sandbox._ensure_attached()
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

        mounted_objects = validate_volumes(volumes if volumes is not None else {})
        cloud_bucket_mounts = [(path, v) for path, v in mounted_objects if isinstance(v, _CloudBucketMount)]
        validated_volumes = [(path, v) for path, v in mounted_objects if isinstance(v, _Volume)]
        self._sandbox._impl.validate_sidecar_create(
            cloud_bucket_mounts=cloud_bucket_mounts,
            has_proxy=proxy is not None,
            include_oidc_identity_token=include_oidc_identity_token,
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
        pty_info = default_pty_info() if pty else None

        container_id, container_name = await self._sandbox._impl.create_sidecar(
            *args,
            name=name,
            image_id=image.object_id,
            secret_ids=[secret.object_id for secret in resolvable_secrets],
            secret_sources=secret_sources,
            workdir=workdir,
            volume_mounts=volume_mounts,
            cloud_bucket_mounts=cloud_bucket_mounts,
            network_access=network_access,
            include_oidc_identity_token=include_oidc_identity_token,
            proxy_id=proxy.object_id if proxy else None,
            pty_info=pty_info,
            memory_reserve_consume_mib=experimental_memory_reserve_consume_mib,
        )
        return _SidecarContainer(self._sandbox, container_id, container_name)

    async def get(self, *, name: str, include_terminated: bool = False) -> "_SidecarContainer":
        self._sandbox._ensure_attached()
        if name == _MAIN_CONTAINER_NAME:
            raise InvalidError(
                "Cannot get the main sandbox container through the sidecars API. "
                "Use Sandbox methods directly to interact with the main container."
            )
        container_info = await self._sandbox._impl.get_sidecar(name, include_terminated)
        return _SidecarContainer._from_container_info(self._sandbox, container_info)

    async def list(self, include_terminated: bool = False) -> builtins.list[_SidecarContainer]:
        self._sandbox._ensure_attached()
        containers = await self._sandbox._impl.list_sidecars(include_terminated)
        return [
            _SidecarContainer._from_container_info(self._sandbox, container)
            for container in containers
            if container.container_name != _MAIN_CONTAINER_NAME
        ]
