# Copyright Modal Labs 2025
import subprocess
import tempfile

from modal.config import config
from modal.exception import NotFoundError
from modal.image import ImageBuilderVersion
from modal.mount import (
    PYTHON_STANDALONE_VERSIONS,
    Mount,
)
from modal_proto import api_pb2


def create_client_dependencies(
    builder_version: ImageBuilderVersion,
    python_version: str,
    platform: str,
    arch: str,
):
    profile_environment = config.get("environment")

    abi_tag = "cp" + python_version.replace(".", "")
    mount_name = f"{builder_version}-{abi_tag}-{platform}-{arch}"

    try:
        Mount.from_name(mount_name, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL).hydrate()
        print(f"✅ Found existing mount {mount_name} in global namespace.")
        return
    except NotFoundError:
        pass

    with tempfile.TemporaryDirectory() as tmpd:
        print(f"📦 Building {mount_name}.")
        requirements = f"modal/requirements/{builder_version}.txt"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-r",
                requirements,
                "--compile-bytecode",
                "--target",
                tmpd,
                "--python-platform",
                f"{arch}-{platform}",
                "--python-version",
                python_version,
            ],
            check=True,
            capture_output=True,
        )
        print(f"🌐 Downloaded and unpacked packages to {tmpd}.")

        python_mount = Mount._from_local_dir(tmpd, remote_path="/pkg")
        python_mount._deploy(
            mount_name,
            api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
            environment_name=profile_environment,
        )
        print(f"✅ Deployed mount {mount_name} to global namespace.")


def main(client=None):
    for python_version in PYTHON_STANDALONE_VERSIONS:
        for platform in ["manylinux_2_17"]:
            create_client_dependencies("PREVIEW", python_version, platform, "x86_64")


if __name__ == "__main__":
    main()
