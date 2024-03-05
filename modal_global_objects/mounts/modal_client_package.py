# Copyright Modal Labs 2022
import asyncio
import sys
from pathlib import Path
from typing import Optional

from modal._packaging import client_mount_name, create_external_client_mount, create_internal_client_mount
from modal.client import _Client
from modal.config import config
from modal_proto import api_pb2


async def publish_client_mount(external_source_dir: str, client: Optional[_Client] = None):
    profile_environment = config.get("environment")

    external_mount = create_external_client_mount(Path(external_source_dir))
    await external_mount._deploy(
        client_mount_name("external"),
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        environment_name=profile_environment,
        client=client,
    )
    print("✅ Deployed external client package mount to global namespace.")

    internal_mount = create_internal_client_mount()
    await internal_mount._deploy(
        client_mount_name("internal"),
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        environment_name=profile_environment,
        client=client,
    )
    print("✅ Deployed internal client package mount to global namespace.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"USAGE: {__file__} EXTERNAL_SOURCE_DIR")
    asyncio.run(publish_client_mount(sys.argv[1]))
