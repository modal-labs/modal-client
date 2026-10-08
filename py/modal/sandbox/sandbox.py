# Copyright Modal Labs 2026
from .._utils.async_utils import synchronize_api, synchronizer
from . import _sandbox

SidecarContainer = synchronize_api(_sandbox._SidecarContainer, target_module=__name__)
SidecarManager = synchronize_api(_sandbox._SidecarManager, target_module=__name__)
Sandbox = synchronize_api(_sandbox._Sandbox, target_module=__name__)

_container_exec = synchronizer.create_blocking(_sandbox._container_exec, "_container_exec", target_module=__name__)
