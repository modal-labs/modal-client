# Copyright Modal Labs 2025
import pytest
from datetime import datetime, timezone

from modal._workspace import _member_role_from_proto
from modal.workspace import Workspace, WorkspaceMemberInfo
from modal_proto import api_pb2


def test_workspace_from_context(servicer, client):
    workspace = Workspace.from_context(client=client)
    workspace.hydrate()
    assert workspace.name == "test-username"


def test_workspace_from_context_lazy_hydration(servicer, client):
    with servicer.intercept() as ctx:
        workspace = Workspace.from_context(client=client)
        assert len(ctx.get_requests("WorkspaceNameLookup")) == 0

        # Explicit hydrate triggers the RPC
        workspace.hydrate()
        assert len(ctx.get_requests("WorkspaceNameLookup")) == 1


def test_workspace_members_list(servicer, client):
    workspace = Workspace.from_context(client=client)
    members = workspace.members.list()

    assert members == [
        WorkspaceMemberInfo(
            user_id="us-1",
            name="Alice",
            email="alice@example.com",
            role="owner",
            joined_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            last_active_at=datetime(2021, 1, 1, tzinfo=timezone.utc),
        ),
        WorkspaceMemberInfo(
            user_id="us-2",
            name="Bob",
            email="bob@example.com",
            role="user",
            joined_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            last_active_at=None,  # Bob has never been active
        ),
    ]


def test_workspace_members_list_empty(servicer, client):
    servicer.workspace_members = []
    workspace = Workspace.from_context(client=client)
    assert workspace.members.list() == []


def test_member_role_from_proto():
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_USER) == "user"
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_MANAGER) == "manager"
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_OWNER) == "owner"

    with pytest.raises(ValueError, match="Unknown workspace member role"):
        _member_role_from_proto(api_pb2.MEMBER_ROLE_UNSPECIFIED)

    with pytest.raises(ValueError, match="Unknown workspace member role"):
        _member_role_from_proto(999)
