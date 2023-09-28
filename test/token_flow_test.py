# Copyright Modal Labs 2023
import pytest

import aiohttp

from modal.token_flow import TokenFlow


@pytest.mark.asyncio
async def test_token_flow_server(servicer, client):
    tf = TokenFlow(client)
    async with tf.start() as (token_flow_id, _, _):
        # Make a request against the local web server and make sure it validates
        localhost_url = f"http://localhost:{servicer.token_flow_localhost_port}"
        async with aiohttp.ClientSession() as session:
            async with session.get(localhost_url) as resp:
                text = await resp.text()
                assert text == token_flow_id
