# Copyright Modal Labs 2023
import pytest

from modal import Function, Queue, Stub, web_endpoint
from modal.exception import NotFoundError
from modal.queue import QueueHandle
from modal.runner import deploy_stub


@pytest.mark.asyncio
async def test_persistent_object(servicer, client):
    stub = Stub()
    stub["q_1"] = Queue.new()
    await deploy_stub.aio(stub, "my-queue", client=client)

    q: QueueHandle = await Queue.lookup.aio("my-queue", client=client)  # type: ignore
    # TODO: remove type annotation here after genstub gets better Generic base class support
    assert isinstance(q, QueueHandle)  # TODO(erikbern): it's a QueueHandler
    assert q.object_id == "qu-1"

    with pytest.raises(NotFoundError):
        await Queue.lookup.aio("bazbazbaz", client=client)  # type: ignore


def square(x):
    # This function isn't deployed anyway
    pass


@pytest.mark.asyncio
async def test_lookup_function(servicer, client):
    stub = Stub()

    stub.function()(square)
    await deploy_stub.aio(stub, "my-function", client=client)

    f = await Function.lookup.aio("my-function", client=client)  # type: ignore
    assert f.object_id == "fu-1"

    # Call it using two arguments
    f = await Function.lookup.aio("my-function", "square", client=client)  # type: ignore
    assert f.object_id == "fu-1"
    with pytest.raises(NotFoundError):
        f = await Function.lookup.aio("my-function", "cube", client=client)  # type: ignore

    # Make sure we can call this function
    assert await f.call.aio(2, 4) == 20
    assert [r async for r in f.map([5, 2], [4, 3])] == [41, 13]


@pytest.mark.asyncio
async def test_webhook_lookup(servicer, client):
    stub = Stub()
    stub.function()(web_endpoint(method="POST")(square))
    await deploy_stub.aio(stub, "my-webhook", client=client)

    f = await Function.lookup.aio("my-webhook", client=client)  # type: ignore
    assert f.web_url


@pytest.mark.asyncio
async def test_deploy_exists(servicer, client):
    assert not await Queue._exists.aio("my-queue", client=client)  # type: ignore
    h1: QueueHandle = await Queue.new()._deploy.aio("my-queue", client=client)
    assert await Queue._exists.aio("my-queue", client=client)  # type: ignore
    h2: QueueHandle = await Queue.lookup.aio("my-queue", client=client)  # type: ignore
    assert h1.object_id == h2.object_id


@pytest.mark.asyncio
async def test_deploy_retain_id(servicer, client):
    h1: QueueHandle = await Queue.new()._deploy.aio("my-queue", client=client)
    h2: QueueHandle = await Queue.new()._deploy.aio("my-queue", client=client)
    assert h1.object_id == h2.object_id
