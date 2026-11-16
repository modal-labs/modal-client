# Copyright Modal Labs 2025
from ._utils.async_utils import synchronize_api
from ._workspace import (
    AuditEvent,
    ProxyTokenInfo,
    TokenData,
    WorkspaceMemberInfo,
    WorkspaceSettings,
    _Workspace,
    _WorkspaceAuditLogsManager,
    _WorkspaceBillingManager,
    _WorkspaceMembersManager,
    _WorkspaceMemberTokenManager,
    _WorkspaceProxyTokenManager,
    _WorkspaceServiceUserManager,
    _WorkspaceSettingsManager,
)

Workspace = synchronize_api(_Workspace, target_module=__name__)
WorkspaceAuditLogsManager = synchronize_api(_WorkspaceAuditLogsManager, target_module=__name__)
WorkspaceMembersManager = synchronize_api(_WorkspaceMembersManager, target_module=__name__)
WorkspaceBillingManager = synchronize_api(_WorkspaceBillingManager, target_module=__name__)
WorkspaceProxyTokenManager = synchronize_api(_WorkspaceProxyTokenManager, target_module=__name__)
WorkspaceSettingsManager = synchronize_api(_WorkspaceSettingsManager, target_module=__name__)
WorkspaceServiceUserManager = synchronize_api(_WorkspaceServiceUserManager, target_module=__name__)
WorkspaceMemberTokenManager = synchronize_api(_WorkspaceMemberTokenManager, target_module=__name__)

__all__ = [
    "AuditEvent",
    "ProxyTokenInfo",
    "TokenData",
    "Workspace",
    "WorkspaceAuditLogsManager",
    "WorkspaceMemberInfo",
    "WorkspaceMembersManager",
    "WorkspaceMemberTokenManager",
    "WorkspaceProxyTokenManager",
    "WorkspaceServiceUserManager",
    "WorkspaceSettingsManager",
    "WorkspaceSettings",
]
