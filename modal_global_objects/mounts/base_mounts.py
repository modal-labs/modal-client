# Copyright Modal Labs 2022
from modal.config import config
from modal.mount import (
    client_mount_name,
    create_client_mount,
)
from modal_proto import api_pb2


def publish_client_mount(client):
    mount = create_client_mount()
    name = client_mount_name()
    profile_environment = config.get("environment")
    # TODO: change how namespaces work, so we don't have to use unrelated workspaces when deploying to global?.
    mount._deploy(
        client_mount_name(),
        api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        client=client,
        environment_name=profile_environment,
    )
    print(f"✅ Deployed client mount {name} to global namespace.")


def main(client=None):
    publish_client_mount(client)


if __name__ == "__main__":
    main()
