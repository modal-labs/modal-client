# Copyright Modal Labs 2022
import asyncio

from modal.mount import create_client_mount, client_mount_name
from modal_proto import api_pb2


async def main(client=None):
    mount = create_client_mount()
    name = client_mount_name()
    # TODO: change how namespaces work, so we don't have to use unrelated workspaces when deploying to global.
    # In the meanwhile, this command has assume an environment exists since AppCreate is always using one before
    # the AppDeploy runs and unassociates the global deploy with workspaces/environments
    await mount._deploy.aio(
        client_mount_name(), api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL, client=client, environment_name="default"
    )
    print(f"✅ Deployed client mount {name} to global namespace.")


if __name__ == "__main__":
    asyncio.run(main())
