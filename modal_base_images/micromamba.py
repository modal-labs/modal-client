# Copyright Modal Labs 2022
import sys
import asyncio

from modal import Image, Stub


async def main(client=None, python_version=None):
    stub = Stub(image=Image.micromamba(python_version))
    async with stub.run.aio(client=client):
        pass


if __name__ == "__main__":
    asyncio.run(main(python_version=sys.argv[1]))
