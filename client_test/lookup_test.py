# Copyright Modal Labs 2023
import pytest

from modal.aio import AioApp, AioQueue, AioStub, aio_lookup
from modal.exception import InvalidError, NotFoundError


@pytest.mark.asyncio
async def test_persistent_object(servicer, aio_client):
    stub_1 = AioStub()
    stub_1["q_1"] = AioQueue()
    await stub_1.deploy("my-queue", client=aio_client)

    stub_2 = AioStub()
    async with stub_2.run(client=aio_client) as app_2:
        assert isinstance(app_2, AioApp)

        q_3 = await aio_lookup("my-queue", client=aio_client)
        # assert isinstance(q_3, AioQueue)  # TODO(erikbern): it's a AioQueueHandler
        assert q_3.object_id == "qu-1"

        with pytest.raises(NotFoundError):
            await aio_lookup("bazbazbaz", client=aio_client)
