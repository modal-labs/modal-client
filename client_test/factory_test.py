import pytest

from modal.aio import AioApp, AioQueue


@AioQueue.factory
async def my_factory(initial_value=42):
    q = await AioQueue.create()
    await q.put(initial_value)
    return q


@pytest.mark.asyncio
async def test_async_factory(servicer, client):
    app = AioApp()

    async with app.run(client=client):
        q = my_factory(43)
        q_id = await app.create_object(q)
        assert q_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(servicer, client):
    # Object reuse is conceptually also done through factories
    app = AioApp()
    q = AioQueue.include("foo-queue")
    async with app.run(client=client):
        q_id = await app.create_object(q)
        assert q_id == "qu-foo"
