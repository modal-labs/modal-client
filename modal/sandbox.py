# Copyright Modal Labs 2022
import os
from typing import Dict, List, Optional, Sequence, Union

from google.protobuf.message import Message
from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal.cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from modal.exception import InvalidError, SandboxTerminatedError, SandboxTimeoutError
from modal.volume import _Volume
from modal_proto import api_pb2

from ._location import parse_cloud_provider
from ._resolver import Resolver
from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream
from ._utils.mount_utils import validate_mount_points, validate_volumes
from .client import _Client
from .config import config
from .gpu import GPU_T, parse_gpu_config
from .image import _Image
from .mount import _Mount
from .network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from .object import _Object
from .secret import _Secret


class _LogsReader:
    """Provides an interface to buffer and fetch logs from a sandbox stream (`stdout` or `stderr`)."""

    def __init__(self, file_descriptor: int, sandbox_id: str, client: _Client) -> None:
        """mdmd:hidden"""

        self._file_descriptor = file_descriptor
        self._sandbox_id = sandbox_id
        self._client = client

    async def read(self) -> str:
        """Fetch and return contents of the entire stream.

        **Usage**

        ```python
        sandbox = stub.app.spawn_sandbox("echo", "hello")
        sandbox.wait()

        print(sandbox.stdout.read())
        ```

        """

        last_log_batch_entry_id = ""
        completed = False
        data = ""

        # TODO: maybe combine this with get_app_logs_loop

        async def _get_logs():
            nonlocal last_log_batch_entry_id, completed, data

            req = api_pb2.SandboxGetLogsRequest(
                sandbox_id=self._sandbox_id,
                file_descriptor=self._file_descriptor,
                timeout=55,
                last_entry_id=last_log_batch_entry_id,
            )
            log_batch: api_pb2.TaskLogsBatch
            async for log_batch in unary_stream(self._client.stub.SandboxGetLogs, req):
                if log_batch.entry_id:
                    # log_batch entry_id is empty for fd="server" messages from AppGetLogs
                    last_log_batch_entry_id = log_batch.entry_id

                if log_batch.eof:
                    completed = True
                    break

                for item in log_batch.items:
                    data += item.data

        while not completed:
            try:
                await _get_logs()
            except (GRPCError, StreamTerminatedError) as exc:
                if isinstance(exc, GRPCError):
                    if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                        continue
                elif isinstance(exc, StreamTerminatedError):
                    continue
                raise

        return data


LogsReader = synchronize_api(_LogsReader)


class _Sandbox(_Object, type_prefix="sb"):
    """A `Sandbox` object lets you interact with a running sandbox. This API is similar to Python's
    [asyncio.subprocess.Process](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.subprocess.Process).

    Refer to the [guide](/docs/guide/sandbox) on how to spawn and use sandboxes.
    """

    _result: Optional[api_pb2.GenericResult]
    _stdout: _LogsReader
    _stderr: _LogsReader

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
        cpu: Optional[float] = None,
        memory: Optional[int] = None,
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        block_network: bool = False,
        volumes: Dict[Union[str, os.PathLike], Union[_Volume, _CloudBucketMount]] = {},
        allow_background_volume_commits: bool = False,
    ) -> "_Sandbox":
        """mdmd:hidden"""

        if len(entrypoint_args) == 0:
            raise InvalidError("entrypoint_args must not be empty")

        if not isinstance(network_file_systems, dict):
            raise InvalidError("network_file_systems must be a dict[str, NetworkFileSystem] where the keys are paths")
        validated_network_file_systems = validate_mount_points("Network file system", network_file_systems)

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
            gpu_config = parse_gpu_config(gpu)

            cloud_provider = parse_cloud_provider(cloud) if cloud else None

            if cpu is not None and cpu < 0.25:
                raise InvalidError(f"Invalid fractional CPU value {cpu}. Cannot have less than 0.25 CPU resources.")
            milli_cpu = int(1000 * cpu) if cpu is not None else None

            # Relies on dicts being ordered (true as of Python 3.6).
            volume_mounts = [
                api_pb2.VolumeMount(
                    mount_path=path,
                    volume_id=volume.object_id,
                    allow_background_commits=allow_background_volume_commits,
                )
                for path, volume in validated_volumes
            ]

            definition = api_pb2.Sandbox(
                entrypoint_args=entrypoint_args,
                image_id=image.object_id,
                mount_ids=[mount.object_id for mount in mounts],
                secret_ids=[secret.object_id for secret in secrets],
                timeout_secs=timeout,
                workdir=workdir,
                resources=api_pb2.Resources(gpu_config=gpu_config, milli_cpu=milli_cpu, memory_mb=memory),
                cloud_provider=cloud_provider,
                nfs_mounts=network_file_system_mount_protos(validated_network_file_systems, False),
                runtime_debug=config.get("function_runtime_debug"),
                block_network=block_network,
                cloud_bucket_mounts=cloud_bucket_mounts_to_proto(cloud_bucket_mounts),
                volume_mounts=volume_mounts,
            )

            create_req = api_pb2.SandboxCreateRequest(app_id=resolver.app_id, definition=definition)
            create_resp = await retry_transient_errors(resolver.client.stub.SandboxCreate, create_req)

            sandbox_id = create_resp.sandbox_id
            self._hydrate(sandbox_id, resolver.client, None)

        return _Sandbox._from_loader(_load, "Sandbox()", deps=_deps)

    def _hydrate_metadata(self, handle_metadata: Optional[Message]):
        self._stdout = LogsReader(api_pb2.FILE_DESCRIPTOR_STDOUT, self.object_id, self._client)
        self._stderr = LogsReader(api_pb2.FILE_DESCRIPTOR_STDERR, self.object_id, self._client)
        self._result = None

    @staticmethod
    async def from_id(sandbox_id: str, client: Optional[_Client] = None) -> "_Sandbox":
        """Construct a Sandbox from an id and look up the sandbox result."""
        if client is None:
            client = await _Client.from_env()

        req = api_pb2.SandboxWaitRequest(sandbox_id=sandbox_id, timeout=0)
        resp = await retry_transient_errors(client.stub.SandboxWait, req)

        obj = _Sandbox._new_hydrated(sandbox_id, client, None)
        obj._result = resp.result

        return obj

    # Live handle methods

    async def wait(self, raise_on_termination: bool = True):
        """Wait for the sandbox to finish running."""

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
        """Terminate sandbox execution.

        This is a no-op if the sandbox has already finished running."""

        await retry_transient_errors(
            self._client.stub.SandboxTerminate, api_pb2.SandboxTerminateRequest(sandbox_id=self.object_id)
        )
        await self.wait(raise_on_termination=False)

    async def poll(self) -> Optional[int]:
        """Check if the sandbox has finished running.

        Returns `None` if the sandbox is still running, else returns the exit code.
        """

        req = api_pb2.SandboxWaitRequest(sandbox_id=self.object_id, timeout=0)
        resp = await retry_transient_errors(self._client.stub.SandboxWait, req)

        if resp.result.status:
            self._result = resp.result

        return self.returncode

    @property
    def stdout(self) -> _LogsReader:
        """`LogsReader` for the sandbox's stdout stream."""

        return self._stdout

    @property
    def stderr(self) -> _LogsReader:
        """`LogsReader` for the sandbox's stderr stream."""

        return self._stderr

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
