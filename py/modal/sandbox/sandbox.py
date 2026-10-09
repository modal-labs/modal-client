# Copyright Modal Labs 2026
from .._utils.async_utils import synchronize_api, synchronizer
from . import _filesystem, _sandbox, _sidecar, _snapshot

SandboxFilesystem = synchronize_api(_filesystem._SandboxFilesystem, target_module=__name__)
SandboxSnapshot = synchronize_api(_snapshot._SandboxSnapshot, target_module=__name__)
SidecarContainer = synchronize_api(_sidecar._SidecarContainer, target_module=__name__)
SidecarManager = synchronize_api(_sidecar._SidecarManager, target_module=__name__)
Sandbox = synchronize_api(_sandbox._Sandbox, target_module=__name__)

_container_exec = synchronizer.create_blocking(_sandbox._container_exec, "_container_exec", target_module=__name__)
