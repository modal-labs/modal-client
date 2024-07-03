# Copyright Modal Labs 2022
import asyncio
import os
from typing import TYPE_CHECKING, AsyncIterator, Dict, List, Optional, Sequence, Tuple, Union

from google.protobuf.message import Message
from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal.cloud_bucket_mount import _CloudBucketMount, cloud_bucket_mounts_to_proto
from modal.exception import InvalidError, SandboxTerminatedError, SandboxTimeoutError
from modal.volume import _Volume
from modal_proto import api_pb2

from ._location import parse_cloud_provider
from ._resolver import Resolver
from ._resources import convert_fn_config_to_resources_config
from ._utils.async_utils import synchronize_api
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream
from ._utils.mount_utils import validate_mount_points, validate_volumes
from .client import _Client
from .config import config
from .gpu import GPU_T
from .image import _Image
from .mount import _Mount
from .network_file_system import _NetworkFileSystem, network_file_system_mount_protos
from .object import _Object
from .scheduler_placement import SchedulerPlacement
from .secret import _Secret

_default_image: _Image = _Image.debian_slim()


if TYPE_CHECKING:
    import modal.app


class _LogsReader:
    """Provides an interface to buffer and fetch logs from a sandbox stream (`stdout` or `stderr`).

    As an asynchronous iterable, the object supports the async for statement.

    **Usage**

    ```python
    @app.function()
    async def my_fn():
        sandbox = app.spawn_sandbox(
            "bash",
            "-c",
            "while true; do echo foo; sleep 1; done"
        )
        async for message in sandbox.stdout:
            print(f"Message: {message}")
    ```
    """

    def __init__(self, file_descriptor: int, sandbox_id: str, client: _Client) -> None:
        """mdmd:hidden"""

        self._file_descriptor = file_descriptor
        self._sandbox_id = sandbox_id
        self._client = client
        self._stream = None
        self._last_log_batch_entry_id = ""
        # Whether the reader received an EOF. Once EOF is True, it returns
        # an empty string for any subsequent reads (including async for)
        self.eof = False

    async def read(self) -> str:
        """Fetch and return contents of the entire stream. If EOF was received,
        return an empty string.

        **Usage**

        ```python
        sandbox = app.app.spawn_sandbox("echo", "hello")
        sandbox.wait()

        print(sandbox.stdout.read())
        ```

        """
        data = ""
        # TODO: maybe combine this with get_app_logs_loop
        async for message in self._get_logs():
            if message is None:
                break
            data += message.data

        return data

    async def _get_logs(self) -> AsyncIterator[Optional[api_pb2.TaskLogs]]:
        """mdmd:hidden
        Streams sandbox logs from the server to the reader.

        When the stream receives an EOF, it yields None. Once an EOF is received,
        subsequent invocations will not yield logs.
        """
        if self.eof:
            yield None
            return

        completed = False

        retries_remaining = 10
        while not completed:
            req = api_pb2.SandboxGetLogsRequest(
                sandbox_id=self._sandbox_id,
                file_descriptor=self._file_descriptor,
                timeout=55,
                last_entry_id=self._last_log_batch_entry_id,
            )
            try:
                async for log_batch in unary_stream(self._client.stub.SandboxGetLogs, req):
                    self._last_log_batch_entry_id = log_batch.entry_id

                    for message in log_batch.items:
                        yield message
                    if log_batch.eof:
                        self.eof = True
                        completed = True
                        yield None
                        break
            except (GRPCError, StreamTerminatedError) as exc:
                if retries_remaining > 0:
                    retries_remaining -= 1
                    if isinstance(exc, GRPCError):
                        if exc.status in RETRYABLE_GRPC_STATUS_CODES:
                            await asyncio.sleep(1.0)
                            continue
                    elif isinstance(exc, StreamTerminatedError):
                        continue
                raise

    def __aiter__(self):
        """mdmd:hidden"""
        self._stream = self._get_logs()
        return self

    async def __anext__(self):
        """mdmd:hidden"""
        value = await self._stream.__anext__()

        # The stream yields None if it receives an EOF batch.
        if value is None:
            raise StopAsyncIteration

        return value.data


MAX_BUFFER_SIZE = 128 * 1024


class _StreamWriter:
    """Provides an interface to buffer and write logs to a sandbox stream (`stdin`)."""

    def __init__(self, sandbox_id: str, client: _Client):
        self._index = 1
        self._sandbox_id = sandbox_id
        self._client = client
        self._is_closed = False
        self._buffer = bytearray()

    def get_next_index(self):
        """mdmd:hidden"""
        index = self._index
        self._index += 1
        return index

    def write(self, data: Union[bytes, bytearray, memoryview]):
        """
        Writes data to stream's internal buffer, but does not drain/flush the write.

        This method needs to be used along with the `drain()` method which flushes the buffer.

        **Usage**

        ```python
        @app.local_entrypoint()
        def main():
            sandbox = app.spawn_sandbox(
                "bash",
                "-c",
                "while read line; do echo $line; done",
            )
            sandbox.stdin.write(b"foo\\n")
            sandbox.stdin.write(b"bar\\n")
            sandbox.stdin.write_eof()

            sandbox.stdin.drain()
            sandbox.wait()
        ```
        """
        if self._is_closed:
            raise EOFError("Stdin is closed. Cannot write to it.")
        if isinstance(data, (bytes, bytearray, memoryview)):
            if len(self._buffer) + len(data) > MAX_BUFFER_SIZE:
                raise BufferError("Buffer size exceed limit. Call drain to clear the buffer.")
            self._buffer.extend(data)
        else:
            raise TypeError(f"data argument must be a bytes-like object, not {type(data).__name__}")

    def write_eof(self):
        """
        Closes the write end of the stream after the buffered write data is drained.
        If the sandbox process was blocked on input, it will become unblocked after `write_eof()`.

        This method needs to be used along with the `drain()` method which flushes the EOF to the process.
        """
        self._is_closed = True

    async def drain(self):
        """
        Flushes the write buffer and EOF to the running Sandbox process.
        """
        data = bytes(self._buffer)
        self._buffer.clear()
        index = self.get_next_index()
        await retry_transient_errors(
            self._client.stub.SandboxStdinWrite,
            api_pb2.SandboxStdinWriteRequest(sandbox_id=self._sandbox_id, index=index, eof=self._is_closed, input=data),
        )


LogsReader = synchronize_api(_LogsReader)
StreamWriter = synchronize_api(_StreamWriter)


class _Sandbox(_Object, type_prefix="sb"):
    """A `Sandbox` object lets you interact with a running sandbox. This API is similar to Python's
    [asyncio.subprocess.Process](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.subprocess.Process).

    Refer to the [guide](/docs/guide/sandbox) on how to spawn and use sandboxes.
    """

    _result: Optional[api_pb2.GenericResult]
    _stdout: _LogsReader
    _stderr: _LogsReader
    _stdin: _StreamWriter

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
        _allow_background_volume_commits: Optional[bool] = None,
        _experimental_scheduler_placement: Optional[SchedulerPlacement] = None,
    ) -> "_Sandbox":
        """mdmd:hidden"""

        if len(entrypoint_args) == 0:
            raise InvalidError("entrypoint_args must not be empty")

        if not isinstance(network_file_systems, dict):
            raise InvalidError("network_file_systems must be a dict[str, NetworkFileSystem] where the keys are paths")
        validated_network_file_systems = validate_mount_points("Network file system", network_file_systems)

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
                    allow_background_commits=_allow_background_volume_commits,
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
            )

            # Note - `resolver.app_id` will be `None` for app-less sandboxes
            create_req = api_pb2.SandboxCreateRequest(app_id=resolver.app_id, definition=definition)
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
        _allow_background_volume_commits: Optional[bool] = None,
        _experimental_scheduler_placement: Optional[
            SchedulerPlacement
        ] = None,  # Experimental controls over fine-grained scheduling (alpha).
        client: Optional[_Client] = None,
    ) -> "_Sandbox":
        if client is None:
            client = await _Client.from_env()

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
            _allow_background_volume_commits=_allow_background_volume_commits,
            _experimental_scheduler_placement=_experimental_scheduler_placement,
        )
        app_id: Optional[str] = app.app_id if app else None
        resolver = Resolver(client, environment_name=environment_name, app_id=app_id)
        await resolver.load(obj)
        return obj

    def _hydrate_metadata(self, handle_metadata: Optional[Message]):
        self._stdout = LogsReader(api_pb2.FILE_DESCRIPTOR_STDOUT, self.object_id, self._client)
        self._stderr = LogsReader(api_pb2.FILE_DESCRIPTOR_STDERR, self.object_id, self._client)
        self._stdin = StreamWriter(self.object_id, self._client)
        self._result = None

    @staticmethod
    async def from_id(sandbox_id: str, client: Optional[_Client] = None) -> "_Sandbox":
        """Construct a Sandbox from an id and look up the sandbox result.

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
    def stdin(self) -> _StreamWriter:
        """`StreamWriter` for the sandbox's stdin stream."""

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
