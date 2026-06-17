# Copyright Modal Labs 2025
from ._utils.async_utils import synchronize_api
from ._workspace import WorkspaceMemberInfo, _Workspace, _WorkspaceBillingManager, _WorkspaceMembersManager

Workspace = synchronize_api(_Workspace, target_module=__name__)
WorkspaceMembersManager = synchronize_api(_WorkspaceMembersManager, target_module=__name__)
WorkspaceBillingManager = synchronize_api(_WorkspaceBillingManager, target_module=__name__)

__all__ = [
    "Workspace",
    "WorkspaceMemberInfo",
    "WorkspaceMembersManager",
]
