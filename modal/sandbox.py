from typing import Optional, Type

from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream

from .client import _Client
from .object import _Handle


class _LogsReader:
    """mdmd:hidden"""

    def __init__(self, file_descriptor: api_pb2.FileDescriptor.ValueType, sandbox_id: str, client: _Client) -> None:
        self._file_descriptor = file_descriptor
        self._sandbox_id = sandbox_id
        self._client = client

    async def read(self):
        last_log_batch_entry_id = ""
        completed = False
        data = ""

        # TODO: maybe combine this with get_app_logs_loop

        async def _get_logs():
            nonlocal last_log_batch_entry_id, completed, data

            req = api_pb2.SandboxGetLogsRequest(
                sandbox_id=self._sandbox_id, file_descriptor=self._file_descriptor, timeout=55
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


class _SandboxHandle(_Handle, type_prefix="sa"):
    """mdmd:hidden"""

    _result: Optional[api_pb2.GenericResult]
    # TODO: fix typing for synchronized class?
    _stdout: _LogsReader
    _stderr: _LogsReader

    @classmethod
    def from_id(cls: Type["_SandboxHandle"], object_id: str, client: Optional[_Client] = None) -> "_SandboxHandle":
        obj = cls._from_id(object_id, client, None)
        obj._stdout = LogsReader(api_pb2.FILE_DESCRIPTOR_STDOUT, object_id, client)
        obj._stderr = LogsReader(api_pb2.FILE_DESCRIPTOR_STDERR, object_id, client)
        return obj

    async def wait(self):
        while True:
            req = api_pb2.SandboxWaitRequest(sandbox_id=self._object_id, timeout=50)
            resp = await retry_transient_errors(self._client.stub.SandboxWait, req)
            if resp.result:
                self._result = resp.result
                break

    @property
    def stdout(self):
        return self._stdout

    @property
    def stderr(self):
        return self._stderr


SandboxHandle = synchronize_api(_SandboxHandle)
