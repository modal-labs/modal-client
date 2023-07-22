# Copyright Modal Labs 2022
import pytest

from modal import Queue, Stub
from modal.exception import InvalidError


@pytest.mark.asyncio
async def test_async_factory(client):
    stub = Stub()
    stub["my_factory"] = Queue.new()
    async with stub.run(client=client) as running_app:
        # assert isinstance(running_app["my_factory"], Queue)  # TODO(erikbern(): is a handle now
        assert running_app["my_factory"].object_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(client):
    stub = Stub()
    q = Queue.from_name("foo-queue")
    assert isinstance(q, Queue)
    stub["my_q"] = q
    async with stub.run(client=client) as running_app:
        assert running_app["my_q"].object_id == "qu-foo"


def test_new_hydrated(client):
    from modal.dict import _DictHandle
    from modal.object import _Handle
    from modal.queue import _QueueHandle

    assert isinstance(_DictHandle._new_hydrated("di-123", client, None), _DictHandle)
    assert isinstance(_QueueHandle._new_hydrated("qu-123", client, None), _QueueHandle)

    with pytest.raises(InvalidError):
        _QueueHandle._new_hydrated("di-123", client, None)  # Wrong prefix for type

    assert isinstance(_Handle._new_hydrated("qu-123", client, None), _QueueHandle)
    assert isinstance(_Handle._new_hydrated("di-123", client, None), _DictHandle)

    with pytest.raises(InvalidError):
        _Handle._new_hydrated("xy-123", client, None)
