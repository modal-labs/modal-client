# Copyright Modal Labs 2022
import asyncio
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from google.protobuf.message import Message

from modal.cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from modal.exception import InvalidError, SandboxTerminatedError, SandboxTimeoutError
from modal.volume import _Volume
from modal_proto import api_pb2

from ._location import parse_cloud_provider
from ._resolver import Resolver
from ._resources import convert_fn_config_to_resources_config
from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import retry_transient_errors
from ._utils.mount_utils import validate_network_file_systems, validate_volumes
from .client import _Client
from .config import config
from .container_process import _ContainerProcess
from .exception import deprecation_warning
from .gpu import GPU_T
from .image import _Image
from .io_streams import StreamReader, StreamWriter, _StreamReader, _StreamWriter
from .mount import _Mount
from .network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from .object import _Object
from .scheduler_placement import SchedulerPlacement
from .secret import _Secret

_default_image: _Image = _Image.debian_slim()


if TYPE_CHECKING:
    import modal.app


class _Sandbox(_Object, type_prefix="sb"):
    """A `Sandbox` object lets you interact with a running sandbox. This API is similar to Python's
    [asyncio.subprocess.Process](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.subprocess.Process).

    Refer to the [guide](/docs/guide/sandbox) on how to spawn and use sandboxes.
    """

    _result: Optional[api_pb2.GenericResult]
    _stdout: _StreamReader
    _stderr: _StreamReader
    _stdin: _StreamWriter
    _task_id: Optional[str] = None

    @staticmethod
    def _new(
        entrypoint_args: Sequence[str],
        image: _Image,
        mounts: Sequence[_Mount],
        secrets: Sequence[_Secret],
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        gpu: GPU_T = None,
        cloud: Optional[str] = None,
        region: Optional[Union[str, Sequence[str]]] = None,
        cpu: Optional[float] = None,
        memory: Optional[Union[int, Tuple[int, int]]] = None,
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        block_network: bool = False,
        volumes: Dict[Union[str, os.PathLike], Union[_Volume, _CloudBucketMount]] = {},
        pty_info: Optional[api_pb2.PTYInfo] = None,
        _experimental_scheduler_placement: Optional[SchedulerPlacement] = None,
        _experimental_gpus: Sequence[GPU_T] = [],
    ) -> "_Sandbox":
        """mdmd:hidden"""

        if len(entrypoint_args) == 0:
            raise InvalidError("entrypoint_args must not be empty")

        validated_network_file_systems = validate_network_file_systems(network_file_systems)

        scheduler_placement: Optional[SchedulerPlacement] = _experimental_scheduler_placement
        if region:
            if scheduler_placement:
                raise InvalidError("`region` and `_experimental_scheduler_placement` cannot be used together")
            scheduler_placement = SchedulerPlacement(region=region)

        # Validate volumes
        validated_volumes = validate_volumes(volumes)
        cloud_bucket_mounts = [(k, v) for k, v in validated_volumes if isinstance(v, _CloudBucketMount)]
        validated_volumes = [(k, v) for k, v in validated_volumes if isinstance(v, _Volume)]

        def _deps() -> List[_Object]:
            deps: List[_Object] = [image] + list(mounts) + list(secrets)
            for _, vol in validated_network_file_systems:
                deps.append(vol)
            for _, vol in validated_volumes:
                deps.append(vol)
            for _, cloud_bucket_mount in cloud_bucket_mounts:
                if cloud_bucket_mount.secret:
                    deps.append(cloud_bucket_mount.secret)
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

            ephemeral_disk = None  # Ephemeral disk requests not supported on Sandboxes.
            definition = api_pb2.Sandbox(
                entrypoint_args=entrypoint_args,
                image_id=image.object_id,
                mount_ids=[mount.object_id for mount in mounts],
                secret_ids=[secret.object_id for secret in secrets],
                timeout_secs=timeout,
                workdir=workdir,
                resources=convert_fn_config_to_resources_config(
                    cpu=cpu, memory=memory, gpu=gpu, ephemeral_disk=ephemeral_disk
                ),
                cloud_provider=parse_cloud_provider(cloud) if cloud else None,
                nfs_mounts=network_file_system_mount_protos(validated_network_file_systems, False),
                runtime_debug=config.get("function_runtime_debug"),
                block_network=block_network,
                cloud_bucket_mounts=cloud_bucket_mounts_to_proto(cloud_bucket_mounts),
                volume_mounts=volume_mounts,
                pty_info=pty_info,
                scheduler_placement=scheduler_placement.proto if scheduler_placement else None,
                worker_id=config.get("worker_id"),
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
        mounts: Sequence[_Mount] = (),  # Mounts to attach to the sandbox.
        secrets: Sequence[_Secret] = (),  # Environment variables to inject into the sandbox.
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        timeout: Optional[int] = None,  # Maximum execution time of the sandbox in seconds.
        workdir: Optional[str] = None,  # Working directory of the sandbox.
        gpu: GPU_T = None,
        cloud: Optional[str] = None,
        region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the sandbox on.
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        # Specify, in MiB, a memory request which is the minimum memory required.
        # Or, pass (request, limit) to additionally specify a hard limit in MiB.
        memory: Optional[Union[int, Tuple[int, int]]] = None,
        block_network: bool = False,  # Whether to block network access
        volumes: Dict[
            Union[str, os.PathLike], Union[_Volume, _CloudBucketMount]
        ] = {},  # Mount points for Modal Volumes and CloudBucketMounts
        pty_info: Optional[api_pb2.PTYInfo] = None,
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
        client: Optional[_Client] = None,
        _experimental_gpus: Sequence[GPU_T] = [],
    ) -> "_Sandbox":
        if environment_name is None:
            environment_name = config.get("environment")

        # TODO(erikbern): Get rid of the `_new` method and create an already-hydrated object
        obj = _Sandbox._new(
            entrypoint_args,
            image=image or _default_image,
            mounts=mounts,
            secrets=secrets,
            timeout=timeout,
            workdir=workdir,
            gpu=gpu,
            cloud=cloud,
            region=region,
            cpu=cpu,
            memory=memory,
            network_file_systems=network_file_systems,
            block_network=block_network,
            volumes=volumes,
            pty_info=pty_info,
            _experimental_scheduler_placement=_experimental_scheduler_placement,
            _experimental_gpus=_experimental_gpus,
        )
        if client is None:
            if app and app._client:
                client = app._client
            else:
                client = await _Client.from_env()
        app_id: Optional[str] = app.app_id if app else None
        resolver = Resolver(client, environment_name=environment_name, app_id=app_id)
        await resolver.load(obj)
        return obj

    def _hydrate_metadata(self, handle_metadata: Optional[Message]):
        self._stdout = StreamReader(api_pb2.FILE_DESCRIPTOR_STDOUT, self.object_id, "sandbox", self._client)
        self._stderr = StreamReader(api_pb2.FILE_DESCRIPTOR_STDERR, self.object_id, "sandbox", self._client)
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
        obj._result = resp.result

        return obj

    # Live handle methods

    async def wait(self, raise_on_termination: bool = True):
        """Wait for the Sandbox to finish running."""

        while True:
            req = api_pb2.SandboxWaitRequest(sandbox_id=self.object_id, timeout=50)
            resp = await retry_transient_errors(self._client.stub.SandboxWait, req)
            if resp.result.status:
                self._result = resp.result

                if resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
                    raise SandboxTimeoutError()
                elif resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED and raise_on_termination:
                    raise SandboxTerminatedError()
                break

    async def terminate(self):
        """Terminate Sandbox execution.

        This is a no-op if the Sandbox has already finished running."""

        await retry_transient_errors(
            self._client.stub.SandboxTerminate, api_pb2.SandboxTerminateRequest(sandbox_id=self.object_id)
        )
        await self.wait(raise_on_termination=False)

    async def poll(self) -> Optional[int]:
        """Check if the Sandbox has finished running.

        Returns `None` if the Sandbox is still running, else returns the exit code.
        """

        req = api_pb2.SandboxWaitRequest(sandbox_id=self.object_id, timeout=0)
        resp = await retry_transient_errors(self._client.stub.SandboxWait, req)

        if resp.result.status:
            self._result = resp.result

        return self.returncode

    async def _get_task_id(self):
        while not self._task_id:
            resp = await self._client.stub.SandboxGetTaskId(api_pb2.SandboxGetTaskIdRequest(sandbox_id=self.object_id))
            self._task_id = resp.task_id
            # TODO: debug why sending an exec right after a task ID exists fails silently
            await asyncio.sleep(0.5)
        return self._task_id

    async def exec(self, *cmds: str, pty_info: Optional[api_pb2.PTYInfo] = None):
        """Execute a command in the Sandbox and return
        a [`ContainerProcess`](/docs/reference/modal.ContainerProcess#modalcontainer_process) handle.

        **Usage**

        ```python
        sandbox = modal.Sandbox.create("sleep", "infinity")

        process = sandbox.exec("bash", "-c", "for i in $(seq 1 10); do echo foo $i; sleep 0.5; done")

        for line in process.stdout:
            print(line)
        ```
        """

        task_id = await self._get_task_id()
        resp = await self._client.stub.ContainerExec(
            api_pb2.ContainerExecRequest(
                task_id=task_id,
                command=cmds,
                pty_info=pty_info,
            )
        )
        return _ContainerProcess(resp.exec_id, self._client)

    @property
    def stdout(self) -> _StreamReader:
        """
        [`StreamReader`](/docs/reference/modal.io_streams#modalio_streamsstreamreader) for
        the sandbox's stdout stream.
        """

        return self._stdout

    @property
    def stderr(self) -> _StreamReader:
        """[`StreamReader`](/docs/reference/modal.io_streams#modalio_streamsstreamreader) for
        the sandbox's stderr stream.
        """

        return self._stderr

    @property
    def stdin(self) -> _StreamWriter:
        """
        [`StreamWriter`](/docs/reference/modal.io_streams#modalio_streamsstreamwriter) for
        the sandbox's stdin stream.
        """

        return self._stdin

    @property
    def returncode(self) -> Optional[int]:
        """Return code of the sandbox process if it has finished running, else `None`."""

        if self._result is None:
            return None
        # Statuses are converted to exitcodes so we can conform to subprocess API.
        # TODO: perhaps there should be a separate property that returns an enum directly?
        elif self._result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
            return 124
        elif self._result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED:
            return 137
        else:
            return self._result.exitcode


Sandbox = synchronize_api(_Sandbox)


def __getattr__(name):
    if name == "LogsReader":
        deprecation_warning(
            (2024, 8, 12),
            "`modal.sandbox.LogsReader` is deprecated. Please import `modal.io_streams.StreamReader` instead.",
        )
        from .io_streams import StreamReader

        return StreamReader
    elif name == "StreamWriter":
        deprecation_warning(
            (2024, 8, 12),
            "`modal.sandbox.StreamWriter` is deprecated. Please import `modal.io_streams.StreamWriter` instead.",
        )
        from .io_streams import StreamWriter

        return StreamWriter
    raise AttributeError(f"module {__name__} has no attribute {name}")
