# Copyright Modal Labs 2022
import asyncio
import os
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, AsyncIterator, Literal, Optional, Union, overload

if TYPE_CHECKING:
    import _typeshed

from google.protobuf.message import Message
from grpclib import GRPCError, Status

from modal._tunnel import Tunnel
from modal.cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from modal.mount import _Mount
from modal.volume import _Volume
from modal_proto import api_pb2

from ._object import _get_environment_name, _Object
from ._resolver import Resolver
from ._resources import convert_fn_config_to_resources_config
from ._utils.async_utils import TaskContext, synchronize_api
from ._utils.grpc_utils import retry_transient_errors
from ._utils.mount_utils import validate_network_file_systems, validate_volumes
from .client import _Client
from .config import config
from .container_process import _ContainerProcess
from .exception import ExecutionError, InvalidError, SandboxTerminatedError, SandboxTimeoutError
from .file_io import FileWatchEvent, FileWatchEventType, _FileIO
from .gpu import GPU_T
from .image import _Image
from .io_streams import StreamReader, StreamWriter, _StreamReader, _StreamWriter
from .network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from .proxy import _Proxy
from .scheduler_placement import SchedulerPlacement
from .secret import _Secret
from .snapshot import _SandboxSnapshot
from .stream_type import StreamType

_default_image: _Image = _Image.debian_slim()


# The maximum number of bytes that can be passed to an exec on Linux.
# Though this is technically a 'server side' limit, it is unlikely to change.
# getconf ARG_MAX will show this value on a host.
#
# By probing in production, the limit is 131072 bytes (2**17).
# We need some bytes of overhead for the rest of the command line besides the args,
# e.g. 'runsc exec ...'. So we use 2**16 as the limit.
ARG_MAX_BYTES = 2**16

if TYPE_CHECKING:
    import modal.app


def _validate_exec_args(entrypoint_args: Sequence[str]) -> None:
    # Entrypoint args must be strings.
    if not all(isinstance(arg, str) for arg in entrypoint_args):
        raise InvalidError("All entrypoint arguments must be strings")
    # Avoid "[Errno 7] Argument list too long" errors.
    total_arg_len = sum(len(arg) for arg in entrypoint_args)
    if total_arg_len > ARG_MAX_BYTES:
        raise InvalidError(
            f"Total length of entrypoint arguments must be less than {ARG_MAX_BYTES} bytes (ARG_MAX). "
            f"Got {total_arg_len} bytes."
        )


class _Sandbox(_Object, type_prefix="sb"):
    """A `Sandbox` object lets you interact with a running sandbox. This API is similar to Python's
    [asyncio.subprocess.Process](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.subprocess.Process).

    Refer to the [guide](https://modal.com/docs/guide/sandbox) on how to spawn and use sandboxes.
    """

    _result: Optional[api_pb2.GenericResult]
    _stdout: _StreamReader[str]
    _stderr: _StreamReader[str]
    _stdin: _StreamWriter
    _task_id: Optional[str] = None
    _tunnels: Optional[dict[int, Tunnel]] = None
    _enable_snapshot: bool = False

    @staticmethod
    def _new(
        entrypoint_args: Sequence[str],
        image: _Image,
        secrets: Sequence[_Secret],
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        gpu: GPU_T = None,
        cloud: Optional[str] = None,
        region: Optional[Union[str, Sequence[str]]] = None,
        cpu: Optional[float] = None,
        memory: Optional[Union[int, tuple[int, int]]] = None,
        mounts: Sequence[_Mount] = (),
        network_file_systems: dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        block_network: bool = False,
        cidr_allowlist: Optional[Sequence[str]] = None,
        volumes: dict[Union[str, os.PathLike], Union[_Volume, _CloudBucketMount]] = {},
        pty_info: Optional[api_pb2.PTYInfo] = None,
        encrypted_ports: Sequence[int] = [],
        h2_ports: Sequence[int] = [],
        unencrypted_ports: Sequence[int] = [],
        proxy: Optional[_Proxy] = None,
        _experimental_scheduler_placement: Optional[SchedulerPlacement] = None,
        enable_snapshot: bool = False,
        verbose: bool = False,
    ) -> "_Sandbox":
        """mdmd:hidden"""

        validated_network_file_systems = validate_network_file_systems(network_file_systems)

        scheduler_placement: Optional[SchedulerPlacement] = _experimental_scheduler_placement
        if region:
            if scheduler_placement:
                raise InvalidError("`region` and `_experimental_scheduler_placement` cannot be used together")
            scheduler_placement = SchedulerPlacement(region=region)

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

        def _deps() -> list[_Object]:
            deps: list[_Object] = [image] + list(mounts) + list(secrets)
            for _, vol in validated_network_file_systems:
                deps.append(vol)
            for _, vol in validated_volumes:
                deps.append(vol)
            for _, cloud_bucket_mount in cloud_bucket_mounts:
                if cloud_bucket_mount.secret:
                    deps.append(cloud_bucket_mount.secret)
            if proxy:
                deps.append(proxy)
            return deps

        async def _load(self: _Sandbox, resolver: Resolver, _existing_object_id: Optional[str]):
            # Relies on dicts being ordered (true as of Python 3.6).
            volume_mounts = [
                api_pb2.VolumeMount(
                    mount_path=path,
                    volume_id=volume.object_id,
                    allow_background_commits=True,
                )
                for path, volume in validated_volumes
            ]

            open_ports = [api_pb2.PortSpec(port=port, unencrypted=False) for port in encrypted_ports]
            open_ports.extend([api_pb2.PortSpec(port=port, unencrypted=True) for port in unencrypted_ports])
            open_ports.extend(
                [
                    api_pb2.PortSpec(port=port, unencrypted=False, tunnel_type=api_pb2.TUNNEL_TYPE_H2)
                    for port in h2_ports
                ]
            )

            if block_network:
                # If the network is blocked, cidr_allowlist is invalid as we don't allow any network access.
                if cidr_allowlist is not None:
                    raise InvalidError("`cidr_allowlist` cannot be used when `block_network` is enabled")
                network_access = api_pb2.NetworkAccess(
                    network_access_type=api_pb2.NetworkAccess.NetworkAccessType.BLOCKED,
                )
            elif cidr_allowlist is None:
                # If the allowlist is empty, we allow all network access.
                network_access = api_pb2.NetworkAccess(
                    network_access_type=api_pb2.NetworkAccess.NetworkAccessType.OPEN,
                )
            else:
                network_access = api_pb2.NetworkAccess(
                    network_access_type=api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST,
                    allowed_cidrs=cidr_allowlist,
                )

            ephemeral_disk = None  # Ephemeral disk requests not supported on Sandboxes.
            definition = api_pb2.Sandbox(
                entrypoint_args=entrypoint_args,
                image_id=image.object_id,
                mount_ids=[mount.object_id for mount in mounts] + [mount.object_id for mount in image._mount_layers],
                secret_ids=[secret.object_id for secret in secrets],
                timeout_secs=timeout,
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
                scheduler_placement=scheduler_placement.proto if scheduler_placement else None,
                worker_id=config.get("worker_id"),
                open_ports=api_pb2.PortSpecs(ports=open_ports),
                network_access=network_access,
                proxy_id=(proxy.object_id if proxy else None),
                enable_snapshot=enable_snapshot,
                verbose=verbose,
            )

            # Note - `resolver.app_id` will be `None` for app-less sandboxes
            create_req = api_pb2.SandboxCreateRequest(
                app_id=resolver.app_id, definition=definition, environment_name=resolver.environment_name
            )
            create_resp = await retry_transient_errors(resolver.client.stub.SandboxCreate, create_req)

            sandbox_id = create_resp.sandbox_id
            self._hydrate(sandbox_id, resolver.client, None)

        return _Sandbox._from_loader(_load, "Sandbox()", deps=_deps)

    @staticmethod
    async def create(
        *entrypoint_args: str,
        app: Optional["modal.app._App"] = None,  # Optionally associate the sandbox with an app
        environment_name: Optional[str] = None,  # Optionally override the default environment
        image: Optional[_Image] = None,  # The image to run as the container for the sandbox.
        secrets: Sequence[_Secret] = (),  # Environment variables to inject into the sandbox.
        network_file_systems: dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        timeout: Optional[int] = None,  # Maximum execution time of the sandbox in seconds.
        workdir: Optional[str] = None,  # Working directory of the sandbox.
        gpu: GPU_T = None,
        cloud: Optional[str] = None,
        region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the sandbox on.
        # Specify, in fractional CPU cores, how many CPU cores to request.
        # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
        # CPU throttling will prevent a container from exceeding its specified limit.
        cpu: Optional[Union[float, tuple[float, float]]] = None,
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, tuple[int, int]]] = None,
        block_network: bool = False,  # Whether to block network access
        # List of CIDRs the sandbox is allowed to access. If None, all CIDRs are allowed.
        cidr_allowlist: Optional[Sequence[str]] = None,
        volumes: dict[
            Union[str, os.PathLike], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes and CloudBucketMounts
        pty_info: Optional[api_pb2.PTYInfo] = None,
        # List of ports to tunnel into the sandbox. Encrypted ports are tunneled with TLS.
        encrypted_ports: Sequence[int] = [],
        # List of encrypted ports to tunnel into the sandbox, using HTTP/2.
        h2_ports: Sequence[int] = [],
        # List of ports to tunnel into the sandbox without encryption.
        unencrypted_ports: Sequence[int] = [],
        # Reference to a Modal Proxy to use in front of this Sandbox.
        proxy: Optional[_Proxy] = None,
        # Enable verbose logging for sandbox operations.
        verbose: bool = False,
        # Enable memory snapshots.
        _experimental_enable_snapshot: bool = False,
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
        client: Optional[_Client] = None,
    ) -> "_Sandbox":
        """
        Create a new Sandbox to run untrusted, arbitrary code. The Sandbox's corresponding container
        will be created asynchronously.

        **Usage**

        ```python
        app = modal.App.lookup('sandbox-hello-world', create_if_missing=True)
        sandbox = modal.Sandbox.create("echo", "hello world", app=app)
        print(sandbox.stdout.read())
        sandbox.wait()
        ```
        """
        return await _Sandbox._create(
            *entrypoint_args,
            app=app,
            environment_name=environment_name,
            image=image,
            secrets=secrets,
            network_file_systems=network_file_systems,
            timeout=timeout,
            workdir=workdir,
            gpu=gpu,
            cloud=cloud,
            region=region,
            cpu=cpu,
            memory=memory,
            block_network=block_network,
            cidr_allowlist=cidr_allowlist,
            volumes=volumes,
            pty_info=pty_info,
            encrypted_ports=encrypted_ports,
            h2_ports=h2_ports,
            unencrypted_ports=unencrypted_ports,
            proxy=proxy,
            _experimental_enable_snapshot=_experimental_enable_snapshot,
            _experimental_scheduler_placement=_experimental_scheduler_placement,
            client=client,
            verbose=verbose,
        )

    @staticmethod
    async def _create(
        *entrypoint_args: str,
        app: Optional["modal.app._App"] = None,  # Optionally associate the sandbox with an app
        environment_name: Optional[str] = None,  # Optionally override the default environment
        image: Optional[_Image] = None,  # The image to run as the container for the sandbox.
        secrets: Sequence[_Secret] = (),  # Environment variables to inject into the sandbox.
        mounts: Sequence[_Mount] = (),
        network_file_systems: dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        timeout: Optional[int] = None,  # Maximum execution time of the sandbox in seconds.
        workdir: Optional[str] = None,  # Working directory of the sandbox.
        gpu: GPU_T = None,
        cloud: Optional[str] = None,
        region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the sandbox on.
        # Specify, in fractional CPU cores, how many CPU cores to request.
        # Or, pass (request, limit) to additionally specify a hard limit in fractional CPU cores.
        # CPU throttling will prevent a container from exceeding its specified limit.
        cpu: Optional[Union[float, tuple[float, float]]] = None,
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, tuple[int, int]]] = None,
        block_network: bool = False,  # Whether to block network access
        # List of CIDRs the sandbox is allowed to access. If None, all CIDRs are allowed.
        cidr_allowlist: Optional[Sequence[str]] = None,
        volumes: dict[
            Union[str, os.PathLike], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes and CloudBucketMounts
        pty_info: Optional[api_pb2.PTYInfo] = None,
        # List of ports to tunnel into the sandbox. Encrypted ports are tunneled with TLS.
        encrypted_ports: Sequence[int] = [],
        # List of encrypted ports to tunnel into the sandbox, using HTTP/2.
        h2_ports: Sequence[int] = [],
        # List of ports to tunnel into the sandbox without encryption.
        unencrypted_ports: Sequence[int] = [],
        # Reference to a Modal Proxy to use in front of this Sandbox.
        proxy: Optional[_Proxy] = None,
        # Enable memory snapshots.
        _experimental_enable_snapshot: bool = False,
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
        client: Optional[_Client] = None,
        verbose: bool = False,
    ):
        # This method exposes some internal arguments (currently `mounts`) which are not in the public API
        # `mounts` is currently only used by modal shell (cli) to provide a function's mounts to the
        # sandbox that runs the shell session
        from .app import _App

        environment_name = _get_environment_name(environment_name)

        _validate_exec_args(entrypoint_args)

        # TODO(erikbern): Get rid of the `_new` method and create an already-hydrated object
        obj = _Sandbox._new(
            entrypoint_args,
            image=image or _default_image,
            secrets=secrets,
            timeout=timeout,
            workdir=workdir,
            gpu=gpu,
            cloud=cloud,
            region=region,
            cpu=cpu,
            memory=memory,
            mounts=mounts,
            network_file_systems=network_file_systems,
            block_network=block_network,
            cidr_allowlist=cidr_allowlist,
            volumes=volumes,
            pty_info=pty_info,
            encrypted_ports=encrypted_ports,
            h2_ports=h2_ports,
            unencrypted_ports=unencrypted_ports,
            proxy=proxy,
            _experimental_scheduler_placement=_experimental_scheduler_placement,
            enable_snapshot=_experimental_enable_snapshot,
            verbose=verbose,
        )
        obj._enable_snapshot = _experimental_enable_snapshot

        app_id: Optional[str] = None
        app_client: Optional[_Client] = None

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

        client = client or app_client or await _Client.from_env()

        resolver = Resolver(client, environment_name=environment_name, app_id=app_id)
        await resolver.load(obj)
        return obj

    def _hydrate_metadata(self, handle_metadata: Optional[Message]):
        self._stdout: _StreamReader[str] = StreamReader[str](
            api_pb2.FILE_DESCRIPTOR_STDOUT, self.object_id, "sandbox", self._client, by_line=True
        )
        self._stderr: _StreamReader[str] = StreamReader[str](
            api_pb2.FILE_DESCRIPTOR_STDERR, self.object_id, "sandbox", self._client, by_line=True
        )
        self._stdin = StreamWriter(self.object_id, "sandbox", self._client)
        self._result = None

    @staticmethod
    async def from_id(sandbox_id: str, client: Optional[_Client] = None) -> "_Sandbox":
        """Construct a Sandbox from an id and look up the Sandbox result.

        The ID of a Sandbox object can be accessed using `.object_id`.
        """
        if client is None:
            client = await _Client.from_env()

        req = api_pb2.SandboxWaitRequest(sandbox_id=sandbox_id, timeout=0)
        resp = await retry_transient_errors(client.stub.SandboxWait, req)

        obj = _Sandbox._new_hydrated(sandbox_id, client, None)

        if resp.result.status:
            obj._result = resp.result

        return obj

    async def set_tags(self, tags: dict[str, str], *, client: Optional[_Client] = None):
        """Set tags (key-value pairs) on the Sandbox. Tags can be used to filter results in `Sandbox.list`."""
        environment_name = _get_environment_name()
        if client is None:
            client = await _Client.from_env()

        tags_list = [api_pb2.SandboxTag(tag_name=name, tag_value=value) for name, value in tags.items()]

        req = api_pb2.SandboxTagsSetRequest(
            environment_name=environment_name,
            sandbox_id=self.object_id,
            tags=tags_list,
        )
        try:
            await retry_transient_errors(client.stub.SandboxTagsSet, req)
        except GRPCError as exc:
            raise InvalidError(exc.message) if exc.status == Status.INVALID_ARGUMENT else exc

    async def snapshot_filesystem(self, timeout: int = 55) -> _Image:
        """Snapshot the filesystem of the Sandbox.

        Returns an [`Image`](https://modal.com/docs/reference/modal.Image) object which
        can be used to spawn a new Sandbox with the same filesystem.
        """
        await self._get_task_id()  # Ensure the sandbox has started
        req = api_pb2.SandboxSnapshotFsRequest(sandbox_id=self.object_id, timeout=timeout)
        resp = await retry_transient_errors(self._client.stub.SandboxSnapshotFs, req)

        if resp.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
            raise ExecutionError(resp.result.exception)

        image_id = resp.image_id
        metadata = resp.image_metadata

        async def _load(self: _Image, resolver: Resolver, existing_object_id: Optional[str]):
            # no need to hydrate again since we do it eagerly below
            pass

        rep = "Image()"
        image = _Image._from_loader(_load, rep, hydrate_lazily=True)
        image._hydrate(image_id, self._client, metadata)  # hydrating eagerly since we have all of the data

        return image

    # Live handle methods

    async def wait(self, raise_on_termination: bool = True):
        """Wait for the Sandbox to finish running."""

        while True:
            req = api_pb2.SandboxWaitRequest(sandbox_id=self.object_id, timeout=10)
            resp = await retry_transient_errors(self._client.stub.SandboxWait, req)
            if resp.result.status:
                self._result = resp.result

                if resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
                    raise SandboxTimeoutError()
                elif resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED and raise_on_termination:
                    raise SandboxTerminatedError()
                break

    async def tunnels(self, timeout: int = 50) -> dict[int, Tunnel]:
        """Get Tunnel metadata for the sandbox.

        Raises `SandboxTimeoutError` if the tunnels are not available after the timeout.

        Returns a dictionary of `Tunnel` objects which are keyed by the container port.

        NOTE: Previous to client [v0.64.153](https://modal.com/docs/reference/changelog#064153-2024-09-30), this
        returned a list of `TunnelData` objects.
        """

        if self._tunnels:
            return self._tunnels

        req = api_pb2.SandboxGetTunnelsRequest(sandbox_id=self.object_id, timeout=timeout)
        resp = await retry_transient_errors(self._client.stub.SandboxGetTunnels, req)

        # If we couldn't get the tunnels in time, report the timeout.
        if resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
            raise SandboxTimeoutError()

        # Otherwise, we got the tunnels and can report the result.
        self._tunnels = {
            t.container_port: Tunnel(t.host, t.port, t.unencrypted_host, t.unencrypted_port) for t in resp.tunnels
        }

        return self._tunnels

    async def terminate(self) -> None:
        """Terminate Sandbox execution.

        This is a no-op if the Sandbox has already finished running."""

        await retry_transient_errors(
            self._client.stub.SandboxTerminate, api_pb2.SandboxTerminateRequest(sandbox_id=self.object_id)
        )

    async def poll(self) -> Optional[int]:
        """Check if the Sandbox has finished running.

        Returns `None` if the Sandbox is still running, else returns the exit code.
        """

        req = api_pb2.SandboxWaitRequest(sandbox_id=self.object_id, timeout=0)
        resp = await retry_transient_errors(self._client.stub.SandboxWait, req)

        if resp.result.status:
            self._result = resp.result

        return self.returncode

    async def _get_task_id(self) -> str:
        while not self._task_id:
            resp = await self._client.stub.SandboxGetTaskId(api_pb2.SandboxGetTaskIdRequest(sandbox_id=self.object_id))
            self._task_id = resp.task_id
            if not self._task_id:
                await asyncio.sleep(0.5)
        return self._task_id

    @overload
    async def exec(
        self,
        *cmds: str,
        pty_info: Optional[api_pb2.PTYInfo] = None,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        secrets: Sequence[_Secret] = (),
        text: Literal[True] = True,
        bufsize: Literal[-1, 1] = -1,
        _pty_info: Optional[api_pb2.PTYInfo] = None,
    ) -> _ContainerProcess[str]: ...

    @overload
    async def exec(
        self,
        *cmds: str,
        pty_info: Optional[api_pb2.PTYInfo] = None,
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        secrets: Sequence[_Secret] = (),
        text: Literal[False] = False,
        bufsize: Literal[-1, 1] = -1,
        _pty_info: Optional[api_pb2.PTYInfo] = None,
    ) -> _ContainerProcess[bytes]: ...

    async def exec(
        self,
        *cmds: str,
        pty_info: Optional[api_pb2.PTYInfo] = None,  # Deprecated: internal use only
        stdout: StreamType = StreamType.PIPE,
        stderr: StreamType = StreamType.PIPE,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        secrets: Sequence[_Secret] = (),
        # Encode output as text.
        text: bool = True,
        # Control line-buffered output.
        # -1 means unbuffered, 1 means line-buffered (only available if `text=True`).
        bufsize: Literal[-1, 1] = -1,
        # Internal option to set terminal size and metadata
        _pty_info: Optional[api_pb2.PTYInfo] = None,
    ):
        """Execute a command in the Sandbox and return a ContainerProcess handle.

        See the [`ContainerProcess`](https://modal.com/docs/reference/modal.container_process#modalcontainer_processcontainerprocess)
        docs for more information.

        **Usage**

        ```python
        app = modal.App.lookup("my-app", create_if_missing=True)

        sandbox = modal.Sandbox.create("sleep", "infinity", app=app)

        process = sandbox.exec("bash", "-c", "for i in $(seq 1 10); do echo foo $i; sleep 0.5; done")

        for line in process.stdout:
            print(line)
        ```
        """

        if workdir is not None and not workdir.startswith("/"):
            raise InvalidError(f"workdir must be an absolute path, got: {workdir}")
        _validate_exec_args(cmds)

        # Force secret resolution so we can pass the secret IDs to the backend.
        secret_coros = [secret.hydrate(client=self._client) for secret in secrets]
        await TaskContext.gather(*secret_coros)

        task_id = await self._get_task_id()
        req = api_pb2.ContainerExecRequest(
            task_id=task_id,
            command=cmds,
            pty_info=_pty_info or pty_info,
            runtime_debug=config.get("function_runtime_debug"),
            timeout_secs=timeout or 0,
            workdir=workdir,
            secret_ids=[secret.object_id for secret in secrets],
        )
        resp = await retry_transient_errors(self._client.stub.ContainerExec, req)
        by_line = bufsize == 1
        return _ContainerProcess(resp.exec_id, self._client, stdout=stdout, stderr=stderr, text=text, by_line=by_line)

    async def _experimental_snapshot(self) -> _SandboxSnapshot:
        await self._get_task_id()
        snap_req = api_pb2.SandboxSnapshotRequest(sandbox_id=self.object_id)
        snap_resp = await retry_transient_errors(self._client.stub.SandboxSnapshot, snap_req)

        snapshot_id = snap_resp.snapshot_id

        # wait for the snapshot to succeed. this is implemented as a second idempotent rpc
        # because the snapshot itself may take a while to complete.
        wait_req = api_pb2.SandboxSnapshotWaitRequest(snapshot_id=snapshot_id, timeout=55.0)
        wait_resp = await retry_transient_errors(self._client.stub.SandboxSnapshotWait, wait_req)
        if wait_resp.result.status != api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
            raise ExecutionError(wait_resp.result.exception)

        async def _load(self: _SandboxSnapshot, resolver: Resolver, existing_object_id: Optional[str]):
            # we eagerly hydrate the sandbox snapshot below
            pass

        rep = "SandboxSnapshot()"
        obj = _SandboxSnapshot._from_loader(_load, rep, hydrate_lazily=True)
        obj._hydrate(snapshot_id, self._client, None)

        return obj

    @staticmethod
    async def _experimental_from_snapshot(snapshot: _SandboxSnapshot, client: Optional[_Client] = None):
        client = client or await _Client.from_env()

        restore_req = api_pb2.SandboxRestoreRequest(snapshot_id=snapshot.object_id)
        restore_resp: api_pb2.SandboxRestoreResponse = await retry_transient_errors(
            client.stub.SandboxRestore, restore_req
        )
        sandbox = await _Sandbox.from_id(restore_resp.sandbox_id, client)

        task_id_req = api_pb2.SandboxGetTaskIdRequest(
            sandbox_id=restore_resp.sandbox_id, wait_until_ready=True, timeout=55.0
        )
        resp = await retry_transient_errors(client.stub.SandboxGetTaskId, task_id_req)
        if resp.task_result.status not in [
            api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED,
            api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
        ]:
            raise ExecutionError(resp.task_result.exception)
        return sandbox

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
        """Open a file in the Sandbox and return a FileIO handle.

        See the [`FileIO`](https://modal.com/docs/reference/modal.file_io#modalfile_iofileio) docs for more information.

        **Usage**

        ```python notest
        sb = modal.Sandbox.create(app=sb_app)
        f = sb.open("/test.txt", "w")
        f.write("hello")
        f.close()
        ```
        """
        task_id = await self._get_task_id()
        return await _FileIO.create(path, mode, self._client, task_id)

    async def ls(self, path: str) -> list[str]:
        """List the contents of a directory in the Sandbox."""
        task_id = await self._get_task_id()
        return await _FileIO.ls(path, self._client, task_id)

    async def mkdir(self, path: str, parents: bool = False) -> None:
        """Create a new directory in the Sandbox."""
        task_id = await self._get_task_id()
        return await _FileIO.mkdir(path, self._client, task_id, parents)

    async def rm(self, path: str, recursive: bool = False) -> None:
        """Remove a file or directory in the Sandbox."""
        task_id = await self._get_task_id()
        return await _FileIO.rm(path, self._client, task_id, recursive)

    async def watch(
        self,
        path: str,
        filter: Optional[list[FileWatchEventType]] = None,
        recursive: Optional[bool] = None,
        timeout: Optional[int] = None,
    ) -> AsyncIterator[FileWatchEvent]:
        """Watch a file or directory in the Sandbox for changes."""
        task_id = await self._get_task_id()
        async for event in _FileIO.watch(path, self._client, task_id, filter, recursive, timeout):
            yield event

    @property
    def stdout(self) -> _StreamReader[str]:
        """
        [`StreamReader`](https://modal.com/docs/reference/modal.io_streams#modalio_streamsstreamreader) for
        the sandbox's stdout stream.
        """

        return self._stdout

    @property
    def stderr(self) -> _StreamReader[str]:
        """[`StreamReader`](https://modal.com/docs/reference/modal.io_streams#modalio_streamsstreamreader) for
        the Sandbox's stderr stream.
        """

        return self._stderr

    @property
    def stdin(self) -> _StreamWriter:
        """
        [`StreamWriter`](https://modal.com/docs/reference/modal.io_streams#modalio_streamsstreamwriter) for
        the Sandbox's stdin stream.
        """

        return self._stdin

    @property
    def returncode(self) -> Optional[int]:
        """Return code of the Sandbox process if it has finished running, else `None`."""
        if self._result is None or self._result.status == api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED:
            return None

        # Statuses are converted to exitcodes so we can conform to subprocess API.
        # TODO: perhaps there should be a separate property that returns an enum directly?
        elif self._result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
            return 124
        elif self._result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED:
            return 137
        else:
            return self._result.exitcode

    @staticmethod
    async def list(
        *, app_id: Optional[str] = None, tags: Optional[dict[str, str]] = None, client: Optional[_Client] = None
    ) -> AsyncGenerator["_Sandbox", None]:
        """List all Sandboxes for the current Environment or App ID (if specified). If tags are specified, only
        Sandboxes that have at least those tags are returned. Returns an iterator over `Sandbox` objects."""
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
            try:
                resp = await retry_transient_errors(client.stub.SandboxList, req)
            except GRPCError as exc:
                raise InvalidError(exc.message) if exc.status == Status.INVALID_ARGUMENT else exc

            if not resp.sandboxes:
                return

            for sandbox_info in resp.sandboxes:
                sandbox_info: api_pb2.SandboxInfo
                obj = _Sandbox._new_hydrated(sandbox_info.id, client, None)
                obj._result = sandbox_info.task_info.result  # TODO: send SandboxInfo as metadata to _new_hydrated?
                yield obj

            # Fetch the next batch starting from the end of the current one.
            before_timestamp = resp.sandboxes[-1].created_at


Sandbox = synchronize_api(_Sandbox)
