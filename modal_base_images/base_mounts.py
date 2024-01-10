# Copyright Modal Labs 2022
import shutil
import tempfile
import urllib.request

from modal.config import config
from modal.exception import NotFoundError
from modal.mount import (
    PYTHON_STANDALONE_VERSIONS,
    Mount,
    client_mount_name,
    create_client_mount,
    python_standalone_mount_name,
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


def publish_python_standalone(client, version: str, libc: str) -> None:
    release, full_version = PYTHON_STANDALONE_VERSIONS[version]

    arch = "x86_64" if version == "3.8" else "x86_64_v3"
    url = (
        "https://github.com/indygreg/python-build-standalone/releases/download"
        + f"/{release}/cpython-{full_version}+{release}-{arch}-unknown-linux-{libc}-install_only.tar.gz"
    )

    profile_environment = config.get("environment")
    mount_name = python_standalone_mount_name(f"{version}-{libc}")
    try:
        Mount.lookup(mount_name, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL, client=client)
        print(f"✅ Found existing mount {mount_name} in global namespace.")
    except NotFoundError:
        print(f"📦 Unpacking python-build-standalone for {version}-{libc}.")
        with tempfile.TemporaryDirectory() as d:
            urllib.request.urlretrieve(url, f"{d}/cpython.tar.gz")
            shutil.unpack_archive(f"{d}/cpython.tar.gz", d)
            print(f"🌐 Downloaded and unpacked archive to {d}.")
            python_mount = Mount.from_local_dir(f"{d}/python")
            python_mount._deploy(
                mount_name,
                api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
                client=client,
                environment_name=profile_environment,
            )
            print(f"✅ Deployed mount {mount_name} to global namespace.")


def main(client=None):
    publish_client_mount(client)
    for version in PYTHON_STANDALONE_VERSIONS:
        publish_python_standalone(client, version, "gnu")
        publish_python_standalone(client, version, "musl")


if __name__ == "__main__":
    main()
