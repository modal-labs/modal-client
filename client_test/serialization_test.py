import pytest

from modal.aio import AioApp, AioQueue

app = AioApp()


@app.local_construction(AioQueue)
async def qf():
    return AioQueue(app=app)


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
        q = app["qf"]
        assert isinstance(q, AioQueue)
        data = app._serialize(q)
        q_roundtrip = app._deserialize(data)
        assert isinstance(q_roundtrip, AioQueue)
        assert q.object_id == q_roundtrip.object_id
