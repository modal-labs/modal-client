# Copyright Modal Labs 2024
"""Utilities related to packaging dependencies for the container runtime."""
from pathlib import Path
from typing import Dict, Literal, Tuple

from modal_proto import api_pb2
from modal_version import __version__

from .config import config
from .exception import InvalidError
from .mount import _Mount, module_mount_condition


def client_mount_name(package_set: Literal["internal", "external"]) -> str:
    """Get the deployed name of the client package mount."""
    return f"modal-client-{package_set}-mount-{__version__}"


def create_external_client_mount(source_dir: Path) -> _Mount:
    return _Mount.from_local_dir(source_dir.resolve(), remote_path="/pkg/external", condition=module_mount_condition)


def create_internal_client_mount() -> _Mount:
    packages = ["modal", "modal_proto", "modal_utils", "modal_version", "synchronicity"]
    return _Mount.from_local_python_packages(*packages, remote_dir="/pkg/internal", condition=module_mount_condition)


def get_client_package_mounts() -> Tuple[_Mount, _Mount]:
    if config["sync_entrypoint"]:
        internal_mount = create_internal_client_mount()
    else:
        internal_mount = _Mount.from_name(client_mount_name("internal"), namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
    external_mount = _Mount.from_name(client_mount_name("external"), namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
    return internal_mount, external_mount


# Supported releases and versions for python-build-standalone.
#
# These can be updated safely, but changes will trigger a rebuild for all images
# that rely on `add_python()` in their constructor.
PYTHON_STANDALONE_VERSIONS: Dict[str, Tuple[str, str]] = {
    "3.8": ("20230826", "3.8.17"),
    "3.9": ("20230826", "3.9.18"),
    "3.10": ("20230826", "3.10.13"),
    "3.11": ("20230826", "3.11.5"),
    "3.12": ("20240107", "3.12.1"),
}


def python_standalone_mount_name(version: str) -> str:
    """Get the deployed name of the python-build-standalone mount."""
    if "-" in version:  # default to glibc
        version, libc = version.split("-")
    else:
        libc = "gnu"
    if version not in PYTHON_STANDALONE_VERSIONS:
        raise InvalidError(
            f"Unsupported standalone python version: {version}, supported values are {list(PYTHON_STANDALONE_VERSIONS.keys())}"
        )
    if libc != "gnu":
        raise InvalidError(f"Unsupported libc identifier: {libc}")
    release, full_version = PYTHON_STANDALONE_VERSIONS[version]
    return f"python-build-standalone.{release}.{full_version}-{libc}"
