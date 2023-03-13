# Copyright Modal Labs 2022
import asyncio

from modal.mount import aio_create_client_mount, client_mount_name
from modal_proto import api_pb2


async def main(client=None):
    mount = aio_create_client_mount()
    name = client_mount_name()
    await mount._deploy(client_mount_name(), api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL, client=client)
    print(f"✅ Deployed client mount {name} to global namespace.")


if __name__ == "__main__":
    asyncio.run(main())
