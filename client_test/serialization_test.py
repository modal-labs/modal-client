# Copyright Modal Labs 2022
import pytest

from modal._serialization import deserialize, serialize
from modal.aio import AioQueue, AioStub

stub = AioStub()

stub.q = AioQueue()


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    async with stub.run(client=client) as running_app:
        q = running_app.q
        data = serialize(q)
        assert len(data) < 256  # Currently 93
        q_roundtrip = deserialize(data, running_app)
        assert q.object_id == q_roundtrip.object_id
