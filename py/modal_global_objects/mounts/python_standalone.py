# Copyright Modal Labs 2022
import argparse
import shutil
import tarfile
import tempfile
import urllib.request

import zstandard as zstd

from modal.config import config
from modal.exception import NotFoundError
from modal.mount import (
    PYTHON_STANDALONE_VERSIONS,
    Mount,
    python_standalone_mount_name,
)
from modal_proto import api_pb2


def publish_python_standalone_mount(client, version: str, builder_version: str) -> None:
    release, full_version = PYTHON_STANDALONE_VERSIONS[builder_version][version]

    libc = "gnu"
    arch = "x86_64_v3"

    root_url = "https://github.com/astral-sh/python-build-standalone/releases/download"
    if full_version.endswith("t"):
        # free-threaded python does not have an install_only artifact:
        # https://github.com/astral-sh/python-build-standalone/issues/536
        # and only uses .tar.zst for compression.
        url = (
            f"{root_url}/{release}/cpython-{full_version[:-1]}+{release}-{arch}-"
            "unknown-linux-gnu-freethreaded+pgo+lto-full.tar.zst"
        )
    else:
        url = f"{root_url}/{release}/cpython-{full_version}+{release}-{arch}-unknown-linux-gnu-install_only.tar.gz"

    profile_environment = config.get("environment") or ""
    mount_name = python_standalone_mount_name(f"{version}-{libc}", builder_version)
    try:
        Mount.from_name(mount_name, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL).hydrate(client)
        print(f"➖ Found existing mount {mount_name} in global namespace.")
    except NotFoundError:
        print(f"📦 Unpacking python-build-standalone for {version}-{libc}.")
        with tempfile.TemporaryDirectory() as d:
            if url.endswith("tar.zst"):
                # The free-threaded standalone build does not have an install_only artifact. Here we
                # decompress the file and only extract the files in the install directory and
                # match the directory structure from the `install_only` builds
                PREFIX = "python/install"
                urllib.request.urlretrieve(url, f"{d}/cpython.tar.zst")
                with open(f"{d}/cpython.tar.zst", "rb") as f:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(f) as reader, tarfile.open(fileobj=reader, mode="r|") as tar:
                        members = (member for member in tar if member.name.startswith(PREFIX))
                        for member in members:
                            member.name = f"python{member.name.removeprefix(PREFIX)}"
                            try:
                                tar.extract(member, path=d, filter="fully_trusted")
                            except TypeError:
                                # filter parameter not supported in older Python versions
                                tar.extract(member, path=d)
            else:
                urllib.request.urlretrieve(url, f"{d}/cpython.tar.gz")
                try:
                    shutil.unpack_archive(f"{d}/cpython.tar.gz", d, filter="fully_trusted")
                except TypeError:
                    # filter parameter not supported in older Python versions
                    shutil.unpack_archive(f"{d}/cpython.tar.gz", d)

            print(f"🌐 Downloaded and unpacked archive to {d}.")
            python_mount = Mount._from_local_dir(f"{d}/python")
            python_mount._deploy(
                mount_name,
                api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
                client=client,
                environment_name=profile_environment,
            )
            print(f"✅ Deployed mount {mount_name} to global namespace.")


def main(client=None, builder_versions: list[str] | None = None):
    if builder_versions is None:
        builder_versions = list(PYTHON_STANDALONE_VERSIONS)

    published = set()
    for builder_version in builder_versions:
        for version in PYTHON_STANDALONE_VERSIONS[builder_version]:
            mount_name = python_standalone_mount_name(version, builder_version)
            if mount_name not in published:
                publish_python_standalone_mount(client, version, builder_version)
                published.add(mount_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish python-build-standalone mounts")
    parser.add_argument(
        "--builder-version",
        action="append",
        choices=list(PYTHON_STANDALONE_VERSIONS),
        dest="builder_versions",
        help="Image builder version to publish mounts for; repeatable. Defaults to all builder versions.",
    )
    args = parser.parse_args()
    main(builder_versions=args.builder_versions)
