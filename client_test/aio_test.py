import pytest

from modal.exception import DeprecationError, InvalidError


@pytest.mark.asyncio
async def test_deprecated(servicer, client):
    with pytest.warns(DeprecationError):
        from modal.aio import AioStub

    stub = AioStub()

    async with stub.run(client=client) as app:
        pass


@pytest.mark.asyncio
async def test_new(servicer, client):
    from modal import Stub

    stub = Stub()

    async with stub.run(client=client) as app:
        pass
