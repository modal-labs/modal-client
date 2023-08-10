# Copyright Modal Labs 2022
import pytest

from modal import Queue, Stub
from modal._serialization import deserialize, serialize

stub = Stub()

stub.q = Queue.new()


@pytest.mark.asyncio
async def test_roundtrip(servicer, client):
    async with stub.run(client=client):
        q = stub.q
        data = serialize(q)
        # TODO: strip synchronizer reference from synchronicity entities!
        assert len(data) < 350  # Used to be 93...
        # Note: if this blows up significantly, it's most likely because
        # cloudpickle can't find a class in the global scope. When this
        # happens, it tries to serialize the entire class along with the
        # object. The reason it doesn't find the class in the global scope
        # is most likely because the name doesn't match. To fix this, make
        # sure that cls.__name__ (which is something synchronicity sets)
        # is the same as the symbol defined in the global scope.
        q_roundtrip = deserialize(data, client)
        assert isinstance(q_roundtrip, Queue)
        assert q.object_id == q_roundtrip.object_id
