# Copyright Modal Labs 2022
import asyncio

from modal import Image, Stub


async def main(client=None):
    stub = Stub(image=Image.conda())
    async with stub.run.aio(client=client):
        pass


if __name__ == "__main__":
    asyncio.run(main())
