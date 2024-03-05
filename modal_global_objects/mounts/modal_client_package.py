# Copyright Modal Labs 2022
import asyncio
import sys
from typing import Optional

from modal._packaging import client_mount_name, mount_client_dependencies, mount_client_package
from modal.client import _Client
from modal.config import config
from modal_proto import api_pb2


async def publish_client_mount(dependency_packages_dir: str, client: _Client):
    profile_environment = config.get("environment")

    dependency_mount = mount_client_dependencies(dependency_packages_dir)
    dependency_mount_name = client_mount_name(qualifier="-dependencies")
    await dependency_mount._deploy(
        dependency_mount_name,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        client=client,
        environment_name=profile_environment,
    )
    print(f"✅ Deployed client dependencies mount `{dependency_mount_name}` to global namespace.")

    client_package_mount = mount_client_package(dependency_mount)
    client_package_mount_name = client_mount_name()
    await client_package_mount._deploy(
        client_package_mount_name,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        client=client,
        environment_name=profile_environment,
    )
    print(f"✅ Deployed client mount `{client_package_mount_name}` to global namespace.")


async def main(dependency_packages_dir: str, client: Optional[_Client] = None):
    await publish_client_mount(dependency_packages_dir, client)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"USAGE: {__file__} DEPENDENCY_PACKAGES_DIR")
    asyncio.run(main(sys.argv[1]))
