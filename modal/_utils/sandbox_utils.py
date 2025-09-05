# Copyright Modal Labs 2025
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterator

from modal_proto import sandbox_router_pb2 as sr_pb2
from modal_proto.sandbox_router_grpc import SandboxRouterServiceStub

from .grpc_utils import connect_channel, create_channel


class SandboxRouterServiceClient:
    """Lightweight client to talk directly to SandboxRouterService on worker
    hosts.

    A new instance should be created per sandbox, using the direct-access URL and JWT
    returned by SandboxGetDirectAccess.

    TODO(saltzm): Review carefully, LLM generated.
    """

    def __init__(self, server_url: str, jwt: str) -> None:
        # Attach bearer token on all requests to the worker-side router service
        self._metadata = {"authorization": f"bearer {jwt}"}
        self._channel = create_channel(server_url, self._metadata)
        # Don't access this directly, use _get_stub() instead.
        self._stub = SandboxRouterServiceStub(self._channel)
        self._connected = False

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await connect_channel(self._channel)
            self._connected = True

    async def _get_stub(self) -> SandboxRouterServiceStub:
        await self._ensure_connected()
        return self._stub

    async def close(self) -> None:
        self._channel.close()
        self._connected = False

    async def exec_start(self, request: sr_pb2.SandboxExecStartRequest) -> sr_pb2.SandboxExecStartResponse:
        stub = await self._get_stub()
        return await stub.SandboxExecStart(request)

    async def stdio_read(
        self,
        request: sr_pb2.SandboxExecStdioReadRequest,
    ) -> AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]:
        stub = await self._get_stub()
        stream = stub.SandboxExecStdioRead.open()
        async with stream as s:
            await s.send_message(request, end=True)
            async for item in s:
                yield item

    async def stdout_read(
        self,
        exec_id: str,
        offset: int = 0,
    ) -> AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]:
        req = sr_pb2.SandboxExecStdioReadRequest(
            exec_id=exec_id,
            offset=offset,
            file_descriptor=sr_pb2.SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT,
        )
        async for item in self.stdio_read(req):
            yield item

    async def stderr_read(
        self,
        exec_id: str,
        offset: int = 0,
    ) -> AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]:
        req = sr_pb2.SandboxExecStdioReadRequest(
            exec_id=exec_id,
            offset=offset,
            file_descriptor=sr_pb2.SANDBOX_STDIO_FILE_DESCRIPTOR_STDERR,
        )
        async for item in self.stdio_read(req):
            yield item

    async def stdin_write(
        self,
        requests: AsyncIterator[sr_pb2.SandboxExecStdinWriteRequest]
        | AsyncGenerator[sr_pb2.SandboxExecStdinWriteRequest, None],
    ) -> sr_pb2.SandboxExecStdinWriteResponse:
        stub = await self._get_stub()
        stream = stub.SandboxExecStdinWrite.open()
        async with stream as s:
            async for req in requests:
                await s.send_message(req)
            await s.end()
            # Receive the single response
            return await s.recv_message()

    async def wait(self, request: sr_pb2.SandboxExecWaitRequest) -> sr_pb2.SandboxExecWaitResponse:
        stub = await self._get_stub()
        return await stub.SandboxExecWait(request)


@dataclass
class DirectAccessMetadata:
    jwt: str
    url: str
