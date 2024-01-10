# Copyright Modal Labs 2023
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Tuple

import aiohttp.web

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.http_utils import run_temporary_http_server

from .client import _Client


class _TokenFlow:
    def __init__(self, client: _Client):
        self.stub = client.stub

    @asynccontextmanager
    async def start(
        self, utm_source: Optional[str] = None, next_url: Optional[str] = None
    ) -> AsyncGenerator[Tuple[str, str, str], None]:
        """mdmd:hidden"""
        # Run a temporary http server returning the token id on /
        # This helps us add direct validation later
        # TODO(erikbern): handle failure launching server

        async def slash(request):
            headers = {"Access-Control-Allow-Origin": "*"}
            return aiohttp.web.Response(text=self.token_flow_id, headers=headers)

        app = aiohttp.web.Application()
        app.add_routes([aiohttp.web.get("/", slash)])
        async with run_temporary_http_server(app) as url:
            req = api_pb2.TokenFlowCreateRequest(
                utm_source=utm_source,
                next_url=next_url,
                localhost_port=int(url.split(":")[-1]),
            )
            resp = await self.stub.TokenFlowCreate(req)
            self.token_flow_id = resp.token_flow_id
            self.wait_secret = resp.wait_secret
            yield (resp.token_flow_id, resp.web_url, resp.code)

    async def finish(
        self, timeout: float = 40.0, grpc_extra_timeout: float = 5.0
    ) -> Optional[api_pb2.TokenFlowWaitResponse]:
        """mdmd:hidden"""
        # Wait for token flow to finish
        req = api_pb2.TokenFlowWaitRequest(
            token_flow_id=self.token_flow_id, timeout=timeout, wait_secret=self.wait_secret
        )
        resp = await self.stub.TokenFlowWait(req, timeout=(timeout + grpc_extra_timeout))
        if not resp.timeout:
            return resp
        else:
            return None


TokenFlow = synchronize_api(_TokenFlow)
