# Copyright Modal Labs 2022
import pytest

from modal import Queue, Secret, Stub
from modal.exception import DeprecationError, InvalidError


@pytest.mark.asyncio
async def test_async_factory(client):
    stub = Stub()
    stub["my_factory"] = Queue.new()
    async with stub.run(client=client):
        assert isinstance(stub["my_factory"], Queue)
        assert stub["my_factory"].object_id == "qu-1"
        with pytest.raises(DeprecationError):
            stub.app["my_factory"]


@pytest.mark.asyncio
async def test_use_object(client):
    # Deploy object
    q = await Queue.lookup.aio("foo-queue", create_if_missing=True, client=client)

    # Use object
    stub = Stub()
    q = Queue.from_name("foo-queue")
    assert isinstance(q, Queue)
    stub["my_q"] = q
    async with stub.run(client=client):
        assert stub["my_q"].object_id == "qu-1"
        with pytest.raises(DeprecationError):
            stub.app["my_q"]


def test_new_hydrated(client):
    from modal.dict import _Dict
    from modal.object import _Object
    from modal.queue import _Queue

    assert isinstance(_Dict._new_hydrated("di-123", client, None), _Dict)
    assert isinstance(_Queue._new_hydrated("qu-123", client, None), _Queue)

    with pytest.raises(InvalidError):
        _Queue._new_hydrated("di-123", client, None)  # Wrong prefix for type

    assert isinstance(_Object._new_hydrated("qu-123", client, None), _Queue)
    assert isinstance(_Object._new_hydrated("di-123", client, None), _Dict)

    with pytest.raises(InvalidError):
        _Object._new_hydrated("xy-123", client, None)


def test_constructor():
    with pytest.raises(InvalidError) as excinfo:
        Secret({"foo": 123})

    assert "Secret" in str(excinfo.value)
    assert "constructor" in str(excinfo.value)
