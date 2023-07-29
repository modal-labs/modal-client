# Copyright Modal Labs 2022
import asyncio

from modal.config import config
from modal.mount import client_mount_name, create_client_mount
from modal_proto import api_pb2


async def main(client=None):
    mount = create_client_mount()
    name = client_mount_name()
    profile_environment = config.get("environment")
    # TODO: change how namespaces work, so we don't have to use unrelated workspaces when deploying to global?.
    await mount._deploy.aio(
        client_mount_name(),
        api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        client=client,
        environment_name=profile_environment,
    )
    print(f"✅ Deployed client mount {name} to global namespace.")


if __name__ == "__main__":
    asyncio.run(main())
