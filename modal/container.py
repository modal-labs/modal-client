from .object import _Handle
from .client import _Client
from typing import Optional, Type

from modal_utils.grpc_utils import retry_transient_errors
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api


class _LogsReader:
    def __init__(self, file_descriptor: api_pb2.FileDescriptor.ValueType, container_id: str, client: _Client) -> None:
        self._file_descriptor = file_descriptor
        self._container_id = container_id
        self._client = client

    async def read(self):
        req = api_pb2.ContainerGetLogsRequest(container_id=self._container_id, file_descriptor=self._file_descriptor)
        # TODO: maybe add a mode that preserves timestamps, since we have them?
        data = ""
        for batch in await retry_transient_errors(self._client.stub.ContainerGetLogs, req):
            for item in batch.items:
                data += item.data
        return data


LogsReader = synchronize_api(_LogsReader)


class _ContainerHandle(_Handle, type_prefix="co"):
    """mdmd:hidden"""

    _result: Optional[api_pb2.GenericResult]
    # TODO: fix typing for synchronized class?
    _stdout: _LogsReader
    _stderr: _LogsReader

    @classmethod
    def from_id(cls: Type["_ContainerHandle"], object_id: str, client: Optional[_Client] = None) -> "_ContainerHandle":
        obj = cls._from_id(object_id, client, None)
        obj._stdout = LogsReader(api_pb2.FILE_DESCRIPTOR_STDOUT, object_id, client)
        obj._stderr = LogsReader(api_pb2.FILE_DESCRIPTOR_STDERR, object_id, client)
        return obj

    async def wait(self):
        while True:
            req = api_pb2.ContainerWaitRequest(container_id=self._object_id, timeout=50)
            resp = await retry_transient_errors(self._client.stub.ContainerWait, req)
            if resp.result:
                self._result = resp.result
                break

    @property
    def stdout(self):
        return self._stdout

    @property
    def stderr(self):
        return self._stderr


ContainerHandle = synchronize_api(_ContainerHandle)
