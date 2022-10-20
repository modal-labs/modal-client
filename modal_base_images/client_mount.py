# Copyright Modal Labs 2022
import asyncio

import modal.aio
from modal.mount import aio_create_client_mount, client_mount_name
from modal_proto import api_pb2


async def main(client=None):
    stub = modal.aio.AioStub()
    stub["mount"] = aio_create_client_mount()
    await stub.deploy(client_mount_name(), api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL, client=client)


if __name__ == "__main__":
    asyncio.run(main())
