# Copyright Modal Labs 2025
import pytest
from datetime import datetime, timezone

from modal._utils.time_utils import timestamp_to_localized_dt
from modal._workspace import _member_role_from_proto
from modal.workspace import ProxyTokenInfo, TokenData, Workspace, WorkspaceMemberInfo
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


def test_workspace_proxy_tokens_create(servicer, client):
    workspace = Workspace.from_context(client=client)
    token = workspace.proxy_tokens.create()

    assert isinstance(token, TokenData)
    assert token.token_id in servicer.webhook_tokens
    assert token.token_secret == f"secret-{token.token_id}"


def test_workspace_proxy_tokens_list(servicer, client):
    servicer.webhook_tokens = {
        "wt-1": {"created_at": 1577836800.0, "scoped": True},
        "wt-2": {"created_at": 1609459200.0, "scoped": False},
    }
    workspace = Workspace.from_context(client=client)

    assert workspace.proxy_tokens.list() == [
        ProxyTokenInfo(token_id="wt-1", created_at=timestamp_to_localized_dt(1577836800.0), scoped=True),
        ProxyTokenInfo(token_id="wt-2", created_at=timestamp_to_localized_dt(1609459200.0), scoped=False),
    ]


def test_workspace_proxy_tokens_list_empty(servicer, client):
    workspace = Workspace.from_context(client=client)
    assert workspace.proxy_tokens.list() == []


def test_workspace_proxy_tokens_list_for_environment(servicer, client):
    main_env_id = servicer.environments["main"]
    # A scoped token associated with "main" is returned ...
    servicer.webhook_tokens["wt-1"] = {"created_at": 1577836800.0, "scoped": True}
    servicer.webhook_token_environments["wt-1"] = {main_env_id}
    # ... while a token associated with a different environment is not.
    servicer.webhook_tokens["wt-2"] = {"created_at": 1609459200.0, "scoped": True}
    servicer.webhook_token_environments["wt-2"] = {"en-other"}
    workspace = Workspace.from_context(client=client)

    assert workspace.proxy_tokens.list(environment_name="main") == [
        ProxyTokenInfo(token_id="wt-1", created_at=timestamp_to_localized_dt(1577836800.0), scoped=True),
    ]


def test_workspace_proxy_tokens_allow_revoke(servicer, client):
    servicer.webhook_tokens["wt-1"] = {"created_at": 1577836800.0, "scoped": True}
    servicer.webhook_token_environments["wt-1"] = set()
    workspace = Workspace.from_context(client=client)

    # `allow` resolves the environment name to its ID and creates the association
    workspace.proxy_tokens.allow("wt-1", "main")
    assert servicer.webhook_token_environments["wt-1"] == {servicer.environments["main"]}

    # `revoke` removes it again
    workspace.proxy_tokens.revoke("wt-1", "main")
    assert servicer.webhook_token_environments["wt-1"] == set()


def test_workspace_proxy_tokens_delete(servicer, client):
    workspace = Workspace.from_context(client=client)
    token = workspace.proxy_tokens.create()
    assert token.token_id in servicer.webhook_tokens

    workspace.proxy_tokens.delete(token.token_id)
    assert token.token_id not in servicer.webhook_tokens


def test_member_role_from_proto():
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_USER) == "user"
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_MANAGER) == "manager"
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_OWNER) == "owner"

    with pytest.raises(ValueError, match="Unknown workspace member role"):
        _member_role_from_proto(api_pb2.MEMBER_ROLE_UNSPECIFIED)

    with pytest.raises(ValueError, match="Unknown workspace member role"):
        _member_role_from_proto(999)
