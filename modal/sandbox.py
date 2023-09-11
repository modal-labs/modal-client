# Copyright Modal Labs 2022
import asyncio
import os
from typing import Dict, List, Optional, Sequence, Union

from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal.exception import InvalidError, SandboxTerminatedError, SandboxTimeoutError
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream

from ._location import parse_cloud_provider
from ._resolver import Resolver
from .client import _Client
from .gpu import GPU_T, parse_gpu_config
from .image import _Image
from .mount import _Mount
from .network_file_system import _NetworkFileSystem, load_network_file_systems
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
    ) -> "_Sandbox":
        """mdmd:hidden"""

        if len(entrypoint_args) == 0:
            raise InvalidError("entrypoint_args must not be empty")

        async def _load(provider: _Sandbox, resolver: Resolver, _existing_object_id: Optional[str]):
            async def _load_ids(objs: Sequence[_Object]) -> List[str]:
                handles = await asyncio.gather(*[resolver.load(obj) for obj in objs])
                return [handle.object_id for handle in handles]

            async def _load_image():
                image_handle = await resolver.load(image)
                return image_handle.object_id

            gpu_config = parse_gpu_config(gpu)

            cloud_provider = parse_cloud_provider(cloud) if cloud else None

            if cpu is not None and cpu < 0.25:
                raise InvalidError(f"Invalid fractional CPU value {cpu}. Cannot have less than 0.25 CPU resources.")
            milli_cpu = int(1000 * cpu) if cpu is not None else None

            image_id, mount_ids, secret_ids, nfs_mounts = await asyncio.gather(
                _load_image(),
                _load_ids(mounts),
                _load_ids(secrets),
                load_network_file_systems(network_file_systems, False, resolver),
            )
            definition = api_pb2.Sandbox(
                entrypoint_args=entrypoint_args,
                image_id=image_id,
                mount_ids=mount_ids,
                secret_ids=secret_ids,
                timeout_secs=timeout,
                workdir=workdir,
                resources=api_pb2.Resources(gpu_config=gpu_config, milli_cpu=milli_cpu, memory_mb=memory),
                cloud_provider=cloud_provider,
                nfs_mounts=nfs_mounts,
            )

            create_req = api_pb2.SandboxCreateRequest(app_id=resolver.app_id, definition=definition)
            create_resp = await retry_transient_errors(resolver.client.stub.SandboxCreate, create_req)

            sandbox_id = create_resp.sandbox_id
            provider._hydrate(sandbox_id, resolver.client, None)
            provider._stdout = LogsReader(api_pb2.FILE_DESCRIPTOR_STDOUT, sandbox_id, resolver.client)
            provider._stderr = LogsReader(api_pb2.FILE_DESCRIPTOR_STDERR, sandbox_id, resolver.client)

        return _Sandbox._from_loader(_load, "Sandbox()")

    # Live handle methods

    async def wait(self):
        """Wait for the sandbox to finish running."""

        while True:
            req = api_pb2.SandboxWaitRequest(sandbox_id=self.object_id, timeout=50)
            resp = await retry_transient_errors(self._client.stub.SandboxWait, req)
            if resp.result.status:
                self._result = resp.result

                if resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
                    raise SandboxTimeoutError()
                elif resp.result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED:
                    raise SandboxTerminatedError()
                break

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
