import pytest

from modal import ref
from modal.aio import AioQueue, AioStub


@pytest.mark.asyncio
async def test_async_factory(servicer, client):
    stub = AioStub()
    stub["my_factory"] = AioQueue()
    async with stub.run(client=client) as running_app:
        assert isinstance(running_app["my_factory"], AioQueue)
        assert running_app["my_factory"].object_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(servicer, client):
    stub = AioStub()
    stub["my_q"] = ref("foo-queue")
    async with stub.run(client=client) as running_app:
        assert running_app["my_q"].object_id == "qu-foo"
