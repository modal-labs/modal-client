# Copyright Modal Labs 2024
"""Utilities related to packaging dependencies for the container runtime."""
from pathlib import Path
from typing import Literal, Tuple

from modal_proto import api_pb2
from modal_version import __version__

from .config import config
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
