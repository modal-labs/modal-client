# Copyright Modal Labs 2022
import asyncio

from modal.aio import AioImage, AioStub


async def main(client=None):
    stub = AioStub(image=AioImage.conda())
    async with stub.run(client=client):
        pass


if __name__ == "__main__":
    asyncio.run(main())
