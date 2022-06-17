import pytest

from modal.aio import AioStub

stub = AioStub()


@stub.webhook(method="POST")
async def f(x):
    return {"square": x**2}


@pytest.mark.asyncio
async def test_webhook(servicer, aio_client):
    async with stub.run(client=aio_client):
        assert f.web_url
