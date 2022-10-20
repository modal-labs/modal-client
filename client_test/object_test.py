# Copyright Modal Labs 2022
import pytest

from modal import ref
from modal.aio import AioQueue, AioStub
from modal.exception import DeprecationError


@pytest.mark.asyncio
async def test_async_factory(servicer, client):
    stub = AioStub()
    stub["my_factory"] = AioQueue()
    async with stub.run(client=client) as running_app:
        # assert isinstance(running_app["my_factory"], AioQueue)  # TODO(erikbern(): is a handle now
        assert running_app["my_factory"].object_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(servicer, client):
    stub = AioStub()
    with pytest.warns(DeprecationError):
        stub["my_q_1"] = ref("foo-queue")
    stub["my_q_2"] = AioQueue.from_name("foo-queue")
    async with stub.run(client=client) as running_app:
        assert running_app["my_q_1"].object_id == "qu-foo"
        assert running_app["my_q_2"].object_id == "qu-foo"
