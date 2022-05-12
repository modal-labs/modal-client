import pytest

from modal import ref
from modal.aio import AioApp, AioQueue


@pytest.mark.asyncio
async def test_async_factory(servicer, client):
    app = AioApp()
    app["my_factory"] = AioQueue()
    assert isinstance(app["my_factory"], AioQueue)
    async with app.run(client=client):
        assert isinstance(app["my_factory"], AioQueue)
        assert app["my_factory"].object_id == "qu-1"


@pytest.mark.asyncio
async def test_use_object(servicer, client):
    app = AioApp()
    with pytest.deprecated_call():
        app["my_q1"] = AioQueue.include(app, "foo-queue")
    app["my_q2"] = ref("foo-queue")
    async with app.run(client=client):
        assert app["my_q1"].object_id == "qu-foo"
        assert app["my_q2"].object_id == "qu-foo"
