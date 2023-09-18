# Copyright Modal Labs 2023
import aiohttp.web
from contextlib import asynccontextmanager
import platform
from typing import Optional, Tuple

from modal_utils.async_utils import synchronize_api
from modal_utils.http_utils import run_temporary_http_server
from modal_proto import api_grpc, api_pb2

from .client import _Client


class _TokenFlow:
    def __init__(self, client: _Client):
        self.stub = client.stub

    @asynccontextmanager
    async def start(self, utm_source: Optional[str] = None) -> Tuple[str, str]:
        """mdmd:hidden"""
        # Run a temporary http server returning the token id on /
        # This helps us add direct validation later

        async def slash(request):
            return aiohttp.web.Response(text=self.token_flow_id)

        app = aiohttp.web.Application()
        app.add_routes([aiohttp.web.get("/", slash)])
        async with run_temporary_http_server(app) as url:
            # Create request
            # Send some strings identifying the computer (these are shown to the user for security reasons)
            req = api_pb2.TokenFlowCreateRequest(
                node_name=platform.node(),
                platform_name=platform.platform(),
                utm_source=utm_source,
                localhost_port=int(url.split(":")[-1]),
            )
            resp = await self.stub.TokenFlowCreate(req)
            self.token_flow_id = resp.token_flow_id
            yield (resp.token_flow_id, resp.web_url)

    async def finish(self, timeout: float = 40) -> Optional[Tuple[str, str]]:
        """mdmd:hidden"""
        # Wait for token flow to finish
        req = api_pb2.TokenFlowWaitRequest(token_flow_id=self.token_flow_id, timeout=timeout)
        resp = await self.stub.TokenFlowWait(req)
        if not resp.timeout:
            return (resp.token_id, resp.token_secret)
        else:
            return


TokenFlow = synchronize_api(_TokenFlow)
