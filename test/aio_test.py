# Copyright Modal Labs 2023
import pytest


@pytest.mark.asyncio
async def test_new(servicer, client):
    from modal import App

    app = App()

    async with app.run(client=client):
        pass
