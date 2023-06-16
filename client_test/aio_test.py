# Copyright Modal Labs 2023
import pytest

from modal.exception import DeprecationError


@pytest.mark.asyncio
async def test_deprecated(servicer, client):
    with pytest.raises(DeprecationError):
        import modal.aio  # noqa


@pytest.mark.asyncio
async def test_new(servicer, client):
    from modal import Stub

    stub = Stub()

    async with stub.run(client=client):
        pass
