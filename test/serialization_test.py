import pytest

from modal import Queue, Session

session = Session()


@Queue.factory
async def qf():
    return await Queue.create()


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    async with session.run(client=client):
        # Serialize "dynamic" object and deserialize
        q = await Queue.create(session=session)
        data = session.serialize(q)
        q_roundtrip = session.deserialize(data)
        assert isinstance(q_roundtrip, Queue)
        assert q.object_id == q_roundtrip.object_id

        # Serialize factory object and deserialize
        await session.create_object(qf)
        data = session.serialize(qf)
        qf_roundtrip = session.deserialize(data)
        assert isinstance(qf_roundtrip, Queue)
        assert qf.object_id == qf_roundtrip.object_id
