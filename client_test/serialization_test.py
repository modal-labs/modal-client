import pytest

from modal._serialization import deserialize, serialize
from modal.aio import AioQueue, AioStub

stub = AioStub()

stub.q = AioQueue()


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    async with stub.run(client=client) as running_app:
        q = running_app.q
        # assert isinstance(q, AioQueue)  # TODO(erikbern): is a Handle now
        data = serialize(q)
        q_roundtrip = deserialize(data, running_app)
        # assert isinstance(q_roundtrip, AioQueue)  # TODO(erikbern): is a Handle now
        assert q.object_id == q_roundtrip.object_id
