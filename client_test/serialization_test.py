import pytest

from modal.aio import AioApp, AioQueue

app = AioApp()


@AioQueue.factory
async def qf():
    return await AioQueue.create()


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    async with app.run(client=client):
        # Serialize "dynamic" object and deserialize
        q = await AioQueue.create(app=app)
        data = app._serialize(q)
        q_roundtrip = app._deserialize(data)
        assert isinstance(q_roundtrip, AioQueue)
        assert q.object_id == q_roundtrip.object_id

        # Serialize factory object and deserialize
        await app.create_object(qf)
        data = app._serialize(qf)
        qf_roundtrip = app._deserialize(data)
        assert isinstance(qf_roundtrip, AioQueue)
        assert qf.object_id == qf_roundtrip.object_id
