# Copyright Modal Labs 2022
from modal.config import config
from modal.exception import NotFoundError
from modal.mount import (
    Mount,
    client_mount_name,
    create_client_mount,
)
from modal_proto import api_pb2


def publish_client_mount(client):
    mount = create_client_mount()
    name = client_mount_name()
    profile_environment = config.get("environment")
    try:
        Mount.from_name(name, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL).hydrate(client)
        print(f"➖ Found existing mount {name} in global namespace.")
    except NotFoundError:
        mount._deploy(
            name,
            api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
            client=client,
            environment_name=profile_environment,
        )
        print(f"✅ Deployed client mount {name} to global namespace.")


def main(client=None):
    publish_client_mount(client)


if __name__ == "__main__":
    main()
