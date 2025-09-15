# Copyright Modal Labs 2025
import os
import ssl
import urllib.parse
from dataclasses import dataclass
from typing import AsyncIterator

import grpclib.client
import grpclib.config
import grpclib.events

from modal_proto import sandbox_router_pb2 as sr_pb2
from modal_proto.sandbox_router_grpc import SandboxRouterServiceStub

from .grpc_utils import connect_channel


class SandboxRouterServiceClient:
    """Lightweight client to talk directly to SandboxRouterService on worker
    hosts.

    A new instance should be created per sandbox, using the direct-access URL and JWT
    returned by SandboxGetDirectAccess.

    TODO(saltzm): Review carefully, LLM generated.
    TODO(saltzm): Handle network errors.
    """

    def __init__(self, server_url: str, jwt: str) -> None:
        # Attach bearer token on all requests to the worker-side router service
        self._metadata = {"authorization": f"bearer {jwt}"}

        # Only https URLs are supported for the sandbox router. Build a channel with a TLS context.
        o = urllib.parse.urlparse(server_url)
        if o.scheme != "https":
            raise ValueError(f"Sandbox router URL must be https, got: {server_url}")

        host, _, port_str = o.netloc.partition(":")
        port = int(port_str) if port_str else 443

        config = grpclib.config.Configuration(
            http2_connection_window_size=64 * 1024 * 1024,  # 64 MiB
            http2_stream_window_size=64 * 1024 * 1024,  # 64 MiB
        )
        ssl_context = ssl.create_default_context()

        # TODO(saltzm): Figure out something more proper for local testing.
        # Optional local testing override: disable verification (INSECURE).
        if os.environ.get("MODAL_SANDBOX_ROUTER_INSECURE") == "1":
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        channel = grpclib.client.Channel(host, port, ssl=ssl_context, config=config)

        async def send_request(event: grpclib.events.SendRequest) -> None:
            for k, v in self._metadata.items():
                event.metadata[k] = v

        grpclib.events.listen(channel, grpclib.events.SendRequest, send_request)

        self._channel = channel

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

    async def exec_stdio_read(
        self,
        request: sr_pb2.SandboxExecStdioReadRequest,
    ) -> AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]:
        stub = await self._get_stub()
        stream = stub.SandboxExecStdioRead.open()
        async with stream as s:
            await s.send_message(request, end=True)
            async for item in s:
                yield item

    async def exec_stdout_read(
        self,
        task_id: str,
        exec_id: str,
        offset: int = 0,
    ) -> AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]:
        req = sr_pb2.SandboxExecStdioReadRequest(
            task_id=task_id,
            exec_id=exec_id,
            offset=offset,
            file_descriptor=sr_pb2.SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT,
        )
        async for item in self.exec_stdio_read(req):
            yield item

    async def exec_stderr_read(
        self,
        task_id: str,
        exec_id: str,
        offset: int = 0,
    ) -> AsyncIterator[sr_pb2.SandboxExecStdioReadResponse]:
        req = sr_pb2.SandboxExecStdioReadRequest(
            task_id=task_id,
            exec_id=exec_id,
            offset=offset,
            file_descriptor=sr_pb2.SANDBOX_STDIO_FILE_DESCRIPTOR_STDERR,
        )
        async for item in self.exec_stdio_read(req):
            yield item

    async def exec_stdin_write(
        self, request: sr_pb2.SandboxExecStdinWriteRequest
    ) -> sr_pb2.SandboxExecStdinWriteResponse:
        stub = await self._get_stub()
        return await stub.SandboxExecStdinWrite(request)

    async def exec_wait(
        self,
        task_id: str,
        exec_id: str,
        timeout: int | None = None,
    ) -> sr_pb2.SandboxExecWaitResponse:
        stub = await self._get_stub()
        request = sr_pb2.SandboxExecWaitRequest(task_id=task_id, exec_id=exec_id)
        # TODO(saltzm): Add timeout.
        return await stub.SandboxExecWait(request)


@dataclass
class DirectAccessMetadata:
    jwt: str
    url: str
