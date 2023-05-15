# Copyright Modal Labs 2023
import pytest

from modal.aio import AioFunction, AioQueue, AioStub, aio_web_endpoint
from modal.exception import NotFoundError
from modal.queue import AioQueueHandle
from modal.runner import aio_deploy_stub


@pytest.mark.asyncio
async def test_persistent_object(servicer, aio_client):
    stub = AioStub()
    stub["q_1"] = AioQueue()
    await aio_deploy_stub(stub, "my-queue", client=aio_client)

    q: AioQueueHandle = await AioQueue.lookup("my-queue", client=aio_client)  # type: ignore
    # TODO: remove type annotation here after genstub gets better Generic base class support
    assert isinstance(q, AioQueueHandle)  # TODO(erikbern): it's a AioQueueHandler
    assert q.object_id == "qu-1"

    with pytest.raises(NotFoundError):
        await AioQueue.lookup("bazbazbaz", client=aio_client)  # type: ignore


def square(x):
    # This function isn't deployed anyway
    pass


@pytest.mark.asyncio
async def test_lookup_function(servicer, aio_client):
    stub = AioStub()

    stub.function()(square)
    await aio_deploy_stub(stub, "my-function", client=aio_client)

    f = await AioFunction.lookup("my-function", client=aio_client)  # type: ignore
    assert f.object_id == "fu-1"

    # Call it using two arguments
    f = await AioFunction.lookup("my-function", "square", client=aio_client)  # type: ignore
    assert f.object_id == "fu-1"
    with pytest.raises(NotFoundError):
        f = await AioFunction.lookup("my-function", "cube", client=aio_client)  # type: ignore

    # Make sure we can call this function
    assert await f.call(2, 4) == 20
    assert [r async for r in f.map([5, 2], [4, 3])] == [41, 13]


@pytest.mark.asyncio
async def test_webhook_lookup(servicer, aio_client):
    stub = AioStub()
    stub.function()(aio_web_endpoint(method="POST")(square))
    await aio_deploy_stub(stub, "my-webhook", client=aio_client)

    f = await AioFunction.lookup("my-webhook", client=aio_client)  # type: ignore
    assert f.web_url


@pytest.mark.asyncio
async def test_deploy_exists(servicer, aio_client):
    assert not await AioQueue._exists("my-queue", client=aio_client)  # type: ignore
    h1 = await AioQueue()._deploy("my-queue", client=aio_client)
    assert await AioQueue._exists("my-queue", client=aio_client)  # type: ignore
    h2 = await AioQueue().lookup("my-queue", client=aio_client)  # type: ignore
    assert h1.object_id == h2.object_id


@pytest.mark.asyncio
async def test_deploy_retain_id(servicer, aio_client):
    h1 = await AioQueue()._deploy("my-queue", client=aio_client)
    h2 = await AioQueue()._deploy("my-queue", client=aio_client)
    assert h1.object_id == h2.object_id
