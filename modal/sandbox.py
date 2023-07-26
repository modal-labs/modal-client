# Copyright Modal Labs 2022
import asyncio
from typing import Optional, Sequence

from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal.exception import InvalidError
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream

from ._resolver import Resolver
from .client import _Client
from .image import _Image
from .mount import _Mount
from .object import _Handle, _Provider


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


class _SandboxHandle(_Handle, type_prefix="sb"):
    """A `SandboxHandle` lets you interact with a spawned sandbox. This API is similar to Python's
    [asyncio.subprocess.Process](https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.subprocess.Process).

    Refer to the [docs](/docs/guide/sandbox) on how to spawn and use sandboxes.
    """

    _result: Optional[api_pb2.GenericResult]
    _stdout: _LogsReader
    _stderr: _LogsReader

    async def wait(self):
        """Wait for the sandbox to finish running."""

        while True:
            req = api_pb2.SandboxWaitRequest(sandbox_id=self._object_id, timeout=50)
            resp = await retry_transient_errors(self._client.stub.SandboxWait, req)
            if resp.result:
                self._result = resp.result
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
        return self._result.exitcode


SandboxHandle = synchronize_api(_SandboxHandle)


class _Sandbox(_Provider[_SandboxHandle]):
    """mdmd:hidden"""

    @staticmethod
    def _new(
        entrypoint_args: Sequence[str],
        image: _Image,
        mounts: Sequence[_Mount],
        timeout: Optional[int] = None,
    ) -> _SandboxHandle:
        """mdmd:hidden"""

        if len(entrypoint_args) == 0:
            raise InvalidError("entrypoint_args must not be empty")

        async def _load(resolver: Resolver, _existing_object_id: Optional[str], handle: _SandboxHandle):
            async def _load_mounts():
                handles = await asyncio.gather(*[resolver.load(mount) for mount in mounts])
                return [handle.object_id for handle in handles]

            async def _load_image():
                image_handle = await resolver.load(image)
                return image_handle.object_id

            image_id, mount_ids = await asyncio.gather(_load_image(), _load_mounts())
            definition = api_pb2.Sandbox(
                entrypoint_args=entrypoint_args,
                image_id=image_id,
                mount_ids=mount_ids,
                timeout_secs=timeout,
            )

            create_req = api_pb2.SandboxCreateRequest(app_id=resolver.app_id, definition=definition)
            create_resp = await retry_transient_errors(resolver.client.stub.SandboxCreate, create_req)

            sandbox_id = create_resp.sandbox_id
            handle._hydrate(sandbox_id, resolver.client, None)

            handle._stdout = LogsReader(api_pb2.FILE_DESCRIPTOR_STDOUT, sandbox_id, resolver.client)
            handle._stderr = LogsReader(api_pb2.FILE_DESCRIPTOR_STDERR, sandbox_id, resolver.client)

        return _Sandbox._from_loader(_load, "Sandbox()")


Sandbox = synchronize_api(_Sandbox)
