import pytest

from modal._serialization import deserialize, serialize
from modal.aio import AioApp, AioQueue

app = AioApp()

app["qf"] = AioQueue()


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    async with app.run(client=client) as running_app:
        # Serialize "dynamic" object and deserialize
        q = await AioQueue().create(app)
        data = serialize(q)
        q_roundtrip = deserialize(data, running_app)
        assert isinstance(q_roundtrip, AioQueue)
        assert q.object_id == q_roundtrip.object_id

        # Serialize factory object and deserialize
        q = app["qf"]
        assert isinstance(q, AioQueue)
        data = serialize(q)
        q_roundtrip = deserialize(data, running_app)
        assert isinstance(q_roundtrip, AioQueue)
        assert q.object_id == q_roundtrip.object_id
