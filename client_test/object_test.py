# Copyright Modal Labs 2022
import pytest

from modal.aio import AioQueue, AioStub
from modal.exception import InvalidError


@pytest.mark.asyncio
async def test_async_factory(aio_client):
    stub = AioStub()
    stub["my_factory"] = AioQueue()
    async with stub.run(client=aio_client) as running_app:
        # assert isinstance(running_app["my_factory"], AioQueue)  # TODO(erikbern(): is a handle now
        assert running_app["my_factory"].object_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(aio_client):
    stub = AioStub()
    q = AioQueue.from_name("foo-queue")
    assert isinstance(q, AioQueue)
    stub["my_q"] = q
    async with stub.run(client=aio_client) as running_app:
        assert running_app["my_q"].object_id == "qu-foo"


def test_from_id(client):
    from modal.object import _Handle
    from modal.dict import _DictHandle
    from modal.queue import _QueueHandle

    assert isinstance(_DictHandle._from_id("di-123", client, None), _DictHandle)
    assert isinstance(_QueueHandle._from_id("qu-123", client, None), _QueueHandle)

    with pytest.raises(InvalidError):
        _QueueHandle._from_id("di-123", client, None)  # Wrong prefix for type

    assert isinstance(_Handle._from_id("qu-123", client, None), _QueueHandle)
    assert isinstance(_Handle._from_id("di-123", client, None), _DictHandle)

    with pytest.raises(InvalidError):
        _Handle._from_id("xy-123", client, None)
