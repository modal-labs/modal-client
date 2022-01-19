import pytest

from modal import Queue, Session


@Queue.factory
async def my_factory(initial_value=42):
    q = await Queue.create()
    await q.put(initial_value)
    return q


@pytest.mark.asyncio
async def test_async_factory(servicer, client):
    q = my_factory(43)
    session = Session()
    async with session.run(client=client):
        q_id = await session.create_object(q)
        assert q_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(servicer, client):
    # Object reuse is conceptually also done through factories
    session = Session()
    q = Queue.use("foo-queue")
    async with session.run(client=client):
        q_id = await session.create_object(q)
        assert q_id == "qu-foo"
