import pytest

from modal import Queue, Session


@Queue.factory
async def my_factory(session, initial_value=42):
    q = Queue(session=session)
    await q.put(initial_value)
    return q


@pytest.mark.asyncio
async def test_async_factory(servicer, client):
    q = my_factory(43)
    session = Session()
    async with session.run(client=client):
        q_id = await session.create_object(q)
        assert q_id == "qu-123456"
