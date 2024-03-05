# Copyright Modal Labs 2024
"""Utilities related to packaging dependencies for the container runtime."""
from importlib.util import find_spec
from pathlib import Path
from typing import Union

from modal_proto import api_pb2
from modal_version import __version__

from .config import config
from .mount import _Mount, module_mount_condition


def client_mount_name(qualifier: str = "") -> str:
    """Get the deployed name of the client package mount."""
    return f"modal-client{qualifier}-mount-{__version__}"


def mount_client_dependencies(install_dir: Union[str, Path]) -> _Mount:
    mount = _Mount.from_local_dir(Path(install_dir).resolve(), remote_path="/pkg", condition=module_mount_condition)
    return mount


def mount_client_package(base_mount: _Mount) -> _Mount:
    mount = base_mount
    internal_packages = ["modal", "modal_proto", "modal_utils", "modal_version", "synchronicity"]
    for pkg in internal_packages:
        spec = find_spec(pkg)
        pkg_dir = Path(spec.origin).parent
        mount = mount.add_local_dir(pkg_dir, remote_path=f"/pkg/{pkg}", condition=module_mount_condition)
    return mount


def _get_client_mount() -> _Mount:  # TODO rethink name
    if config["sync_entrypoint"]:
        base_name = client_mount_name(qualifier="-dependencies")
        base_mount = _Mount.from_name(base_name, namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
        full_mount = mount_client_package(base_mount)
        return full_mount
    else:
        return _Mount.from_name(client_mount_name(), namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)
