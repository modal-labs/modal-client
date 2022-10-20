# Copyright Modal Labs 2022
import asyncio
import sys

from modal.aio import AioImage, AioStub


async def main(client=None, python_version=None):
    stub = AioStub(image=AioImage.debian_slim(python_version))
    async with stub.run(client=client):
        pass


if __name__ == "__main__":
    asyncio.run(main(python_version=sys.argv[1]))
