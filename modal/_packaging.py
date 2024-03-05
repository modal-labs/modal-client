# Copyright Modal Labs 2024
"""Utilities related to packaging dependencies for the container runtime."""
from pathlib import Path
from typing import Union

from modal_version import __version__

from .mount import _Mount, module_mount_condition


def client_mount_name(qualifier: str = "") -> str:
    """Get the deployed name of the client package mount."""
    return f"modal-client{qualifier}-mount-{__version__}"


def mount_client_dependencies(pkg_dir: Union[str, Path]) -> _Mount:
    mount = _Mount.from_local_dir(Path(pkg_dir).resolve(), remote_path="/pkg", condition=module_mount_condition)
    return mount


def mount_client_package(base_mount: _Mount) -> _Mount:
    mount = base_mount
    internal_packages = ["modal", "modal_proto", "modal_utils", "modal_version", "synchronicity"]
    for pkg in internal_packages:
        module = __import__(pkg)
        mount = mount.add_local_dir(module.__path__[0], remote_path=f"/pkg/{pkg}", condition=module_mount_condition)
    return mount
