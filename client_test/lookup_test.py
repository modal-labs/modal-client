# Copyright Modal Labs 2023
import pytest

from modal.aio import AioFunction, AioQueue, AioStub, aio_lookup
from modal.exception import DeprecationError, NotFoundError


@pytest.mark.asyncio
async def test_persistent_object(servicer, aio_client):
    stub = AioStub()
    stub["q_1"] = AioQueue()
    await stub.deploy("my-queue", client=aio_client)

    q = await AioQueue.lookup("my-queue", client=aio_client)
    # assert isinstance(q_3, AioQueue)  # TODO(erikbern): it's a AioQueueHandler
    assert q.object_id == "qu-1"

    with pytest.raises(NotFoundError):
        await AioQueue.lookup("bazbazbaz", client=aio_client)


def square(x):
    # This function isn't deployed anyway
    pass


@pytest.mark.asyncio
async def test_lookup_function(servicer, aio_client):
    stub = AioStub()

    stub.function(square)
    await stub.deploy("my-function", client=aio_client)

    f = await AioFunction.lookup("my-function", client=aio_client)
    assert f.object_id == "fu-1"

    # Call it using two arguments
    f = await AioFunction.lookup("my-function", "square", client=aio_client)
    assert f.object_id == "fu-1"
    with pytest.raises(NotFoundError):
        f = await AioFunction.lookup("my-function", "cube", client=aio_client)

    # Make sure we can call this function
    assert await f.call(2, 4) == 20
    assert [r async for r in f.map([5, 2], [4, 3])] == [41, 13]


@pytest.mark.asyncio
async def test_webhook_lookup(servicer, aio_client):
    stub = AioStub()
    stub.webhook(square, method="POST")
    await stub.deploy("my-webhook", client=aio_client)

    f = await AioFunction.lookup("my-webhook", client=aio_client)
    assert f.web_url


@pytest.mark.asyncio
async def test_deprecated(servicer, aio_client):
    stub = AioStub()
    stub["q_1"] = AioQueue()
    await stub.deploy("my-queue", client=aio_client)

    servicer.enforce_object_entity = False
    with pytest.warns(DeprecationError):
        q = await aio_lookup("my-queue", client=aio_client)
    assert q.object_id == "qu-1"


@pytest.mark.asyncio
async def test_deploy_exists(servicer, aio_client):
    assert not await AioQueue._exists("my-queue", client=aio_client)
    h1 = await AioQueue()._deploy("my-queue", client=aio_client)
    assert await AioQueue._exists("my-queue", client=aio_client)
    h2 = await AioQueue().lookup("my-queue", client=aio_client)
    assert h1.object_id == h2.object_id
