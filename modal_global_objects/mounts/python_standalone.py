# Copyright Modal Labs 2022
import shutil
import tempfile
import urllib.request

from modal.config import config
from modal.exception import NotFoundError
from modal.mount import (
    PYTHON_STANDALONE_VERSIONS,
    Mount,
    python_standalone_mount_name,
)
from modal_proto import api_pb2


def publish_python_standalone_mount(client, version: str) -> None:
    release, full_version = PYTHON_STANDALONE_VERSIONS[version]

    libc = "gnu"
    arch = "x86_64" if version == "3.8" else "x86_64_v3"
    url = (
        "https://github.com/indygreg/python-build-standalone/releases/download"
        + f"/{release}/cpython-{full_version}+{release}-{arch}-unknown-linux-gnu-install_only.tar.gz"
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
    for version in PYTHON_STANDALONE_VERSIONS:
        publish_python_standalone_mount(client, version)


if __name__ == "__main__":
    main()
