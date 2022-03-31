import pytest

from modal.aio import AioApp, AioQueue

app = AioApp()


@app.local_construction(AioQueue)
async def my_factory():
    return AioQueue(app=app)


@pytest.mark.asyncio
async def test_async_factory(servicer, client):
    assert my_factory.tag == "client_test.factory_test.my_factory"
    async with app.run(client=client):
        assert isinstance(my_factory, AioQueue)
        assert my_factory.object_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(servicer, client):
    # Object reuse is conceptually also done through factories
    q = AioQueue.include(app, "foo-queue")
    async with app.run(client=client):
        q_id = await app.create_object(q)
        assert q_id == "qu-foo"
