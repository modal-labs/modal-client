# Copyright Modal Labs 2025
from ._utils.async_utils import synchronize_api
from ._workspace import (
    ProxyTokenInfo,
    TokenData,
    WorkspaceMemberInfo,
    WorkspaceSettings,
    _Workspace,
    _WorkspaceBillingManager,
    _WorkspaceMembersManager,
    _WorkspaceProxyTokenManager,
    _WorkspaceSettingsManager,
)

Workspace = synchronize_api(_Workspace, target_module=__name__)
WorkspaceMembersManager = synchronize_api(_WorkspaceMembersManager, target_module=__name__)
WorkspaceBillingManager = synchronize_api(_WorkspaceBillingManager, target_module=__name__)
WorkspaceProxyTokenManager = synchronize_api(_WorkspaceProxyTokenManager, target_module=__name__)
WorkspaceSettingsManager = synchronize_api(_WorkspaceSettingsManager, target_module=__name__)

__all__ = [
    "ProxyTokenInfo",
    "TokenData",
    "Workspace",
    "WorkspaceMemberInfo",
    "WorkspaceMembersManager",
    "WorkspaceProxyTokenManager",
    "WorkspaceSettingsManager",
    "WorkspaceSettings",
]
