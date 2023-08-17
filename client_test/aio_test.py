# Copyright Modal Labs 2023
import pytest


@pytest.mark.asyncio
async def test_new(servicer, client):
    from modal import Stub

    stub = Stub()

    async with stub.run(client=client):
        pass
