import pytest

from modal.aio import AioApp, AioQueue

app = AioApp()

app["my_factory"] = AioQueue()


@pytest.mark.asyncio
async def test_async_factory(servicer, client):
    assert isinstance(app["my_factory"], AioQueue)
    async with app.run(client=client):
        assert isinstance(app["my_factory"], AioQueue)
        assert app["my_factory"].object_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(servicer, client):
    # Object reuse is conceptually also done through factories
    q = AioQueue.include(app, "foo-queue")
    async with app.run(client=client):
        q_id = await app.create_object(q)
        assert q_id == "qu-foo"
