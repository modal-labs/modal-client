# Copyright Modal Labs 2025
import ipaddress
import pytest
import re
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from google.protobuf.timestamp_pb2 import Timestamp
from grpclib import GRPCError, Status

from modal._utils.time_utils import timestamp_to_localized_dt
from modal._workspace import _member_role_from_proto
from modal.environments import Environment
from modal.exception import InvalidError, NotFoundError, PermissionDeniedError
from modal.types import AuditEvent
from modal.workspace import ProxyTokenInfo, TokenData, Workspace, WorkspaceMemberInfo
from modal_proto import api_pb2

from .conftest import MockClientServicer


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
            role="member",
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
    token = workspace.proxy_tokens.create(name="production-webhooks")

    assert isinstance(token, TokenData)
    assert token.token_id in servicer.webhook_tokens
    assert servicer.webhook_tokens[token.token_id]["name"] == "production-webhooks"
    assert token.token_secret == f"secret-{token.token_id}"


def test_workspace_proxy_tokens_update(servicer, client):
    workspace = Workspace.from_context(client=client)
    token = workspace.proxy_tokens.create(name="production-webhooks")

    workspace.proxy_tokens.update(token.token_id, name="staging-webhooks")
    assert servicer.webhook_tokens[token.token_id]["name"] == "staging-webhooks"

    workspace.proxy_tokens.update(token.token_id, name="")
    assert servicer.webhook_tokens[token.token_id]["name"] == ""


def test_workspace_proxy_tokens_list(servicer, client):
    servicer.webhook_tokens = {
        "wt-1": {
            "created_at": 1577836800.0,
            "scoped": True,
            "name": "production-webhooks",
            "created_by": api_pb2.UserIdentity(
                user_id="us-1",
                username="Alice",
            ),
        },
        "wt-2": {"created_at": 1609459200.0, "scoped": False},
    }
    workspace = Workspace.from_context(client=client)

    assert workspace.proxy_tokens.list() == [
        ProxyTokenInfo(
            token_id="wt-1",
            created_at=timestamp_to_localized_dt(1577836800.0),
            scoped=True,
            name="production-webhooks",
            created_by="Alice",
        ),
        ProxyTokenInfo(token_id="wt-2", created_at=timestamp_to_localized_dt(1609459200.0), scoped=False),
    ]


def test_workspace_proxy_tokens_list_empty(servicer, client):
    workspace = Workspace.from_context(client=client)
    assert workspace.proxy_tokens.list() == []


def test_workspace_proxy_tokens_list_for_environment(servicer, client):
    main_env_id = servicer.environments["main"]
    # A scoped token associated with "main" is returned ...
    servicer.webhook_tokens["wt-1"] = {
        "created_at": 1577836800.0,
        "scoped": True,
        "name": "production-webhooks",
        "created_by": api_pb2.UserIdentity(
            user_id="us-1",
            username="Alice",
        ),
    }
    servicer.webhook_token_environments["wt-1"] = {main_env_id}
    # ... while a token associated with a different environment is not.
    servicer.webhook_tokens["wt-2"] = {"created_at": 1609459200.0, "scoped": True}
    servicer.webhook_token_environments["wt-2"] = {"en-other"}
    workspace = Workspace.from_context(client=client)

    assert workspace.proxy_tokens.list(environment_name="main") == [
        ProxyTokenInfo(
            token_id="wt-1",
            created_at=timestamp_to_localized_dt(1577836800.0),
            scoped=True,
            name="production-webhooks",
            created_by="Alice",
        ),
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


def test_workspace_settings_list(servicer, client):
    settings = Workspace.from_context(client=client).settings.list()
    assert settings.image_builder_version == "2024.10"
    assert settings.default_environment == "main"


def test_workspace_settings_set_image_builder_version(servicer, client):
    Workspace.from_context(client=client).settings.set("image-builder-version", "2025.06")
    assert servicer.workspace_image_builder_version == "2025.06"
    assert servicer.workspace_default_environment_name == "main"  # unchanged


def test_workspace_settings_set_default_environment(servicer, client):
    Workspace.from_context(client=client).settings.set("default-environment", "prod")
    assert servicer.workspace_default_environment_name == "prod"
    assert servicer.workspace_image_builder_version == "2024.10"  # unchanged


def test_workspace_settings_set_unknown_raises(servicer, client):
    with pytest.raises(ValueError, match="Unknown setting"):
        Workspace.from_context(client=client).settings.set("nonexistent", "value")


def test_workspace_settings_set_raises_on_rpc_failure(servicer, client):
    async def bad_handler(self, stream):
        await stream.recv_message()
        raise GRPCError(Status.INVALID_ARGUMENT, "invalid version")

    with servicer.intercept() as ctx:
        ctx.set_responder("WorkspaceSetImageBuilderVersion", bad_handler)
        with pytest.raises(Exception, match="invalid version"):
            Workspace.from_context(client=client).settings.set("image-builder-version", "bad")


def test_member_role_from_proto():
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_USER) == "member"
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_MANAGER) == "manager"
    assert _member_role_from_proto(api_pb2.MEMBER_ROLE_OWNER) == "owner"

    with pytest.raises(ValueError, match="Unknown workspace member role"):
        _member_role_from_proto(api_pb2.MEMBER_ROLE_UNSPECIFIED)

    with pytest.raises(ValueError, match="Unknown workspace member role"):
        _member_role_from_proto(999)


def test_workspace_service_users(servicer, client):
    ws = Workspace.from_context(client=client)
    env = Environment.from_name("main", client=client)

    service_users = list(ws.service_users.list())
    assert len(service_users) == 2

    roles = env.roles.list()
    assert roles["service_users"] == {"alice-bot": "contributor", "ops-bot": "contributor"}

    for i in range(4):
        ws.service_users.create(f"service-user-{i}", expires_in=None)

    service_users = list(ws.service_users.list())
    assert len(service_users) == 6

    info = ws.service_users.info("service-user-1")

    assert info.name == "service-user-1"
    assert info.expires_at is None
    assert len(info.environment_roles) == 0

    for i in range(4):
        ws.service_users.set_role(f"service-user-{i}", "main", role="viewer" if i % 2 else "no-access")

    roles = env.roles.list()
    assert roles["service_users"] == {
        "alice-bot": "contributor",
        "ops-bot": "contributor",
        "service-user-0": "no-access",
        "service-user-1": "viewer",
        "service-user-2": "no-access",
        "service-user-3": "viewer",
    }

    for i in range(2):
        ws.service_users.delete(f"service-user-{i * 2}")

    roles = env.roles.list()
    assert roles["service_users"] == {
        "alice-bot": "contributor",
        "ops-bot": "contributor",
        "service-user-1": "viewer",
        "service-user-3": "viewer",
    }

    with pytest.raises(NotFoundError, match="Service user 'service-user-2' not found"):
        ws.service_users.delete("service-user-2")

    ws.service_users.delete("service-user-2", allow_missing=True)

    info = ws.service_users.info("service-user-1")

    assert info.name == "service-user-1"
    assert info.expires_at is None
    assert len(info.environment_roles) == 1
    assert info.environment_roles["main"] == "viewer"


def test_workspace_member_tokens(client, servicer: MockClientServicer):
    ws = Workspace.from_context(client=client)

    tokens = ws.member_tokens.list()
    assert len(tokens) == 2
    assert set(token.token_id for token in tokens) == {f"ak-MT{i}" for i in range(1, 3)}

    for i in range(10):
        expiration: int | None = None
        if i % 2 == 0:
            expiration = (i + 1) * 86_400

        servicer._create_token(f"token-{i}", expiration=expiration)

    tokens = ws.member_tokens.list()
    assert len(tokens) == 12
    assert set(token.token_id for token in tokens) == {f"ak-MT{i}" for i in range(1, 13)}

    expr = re.compile(r"^token-(?P<idx>\d+)$")
    for token in tokens:
        if token.token_id in {"ak-MT1", "ak-MT2"}:
            continue

        assert token.name

        match = expr.fullmatch(token.name)
        assert match

        i = int(match["idx"])
        if i % 2 == 0:
            assert token.expires_at is not None
        else:
            assert token.expires_at is None

    deleted = ws.member_tokens.delete(*[f"ak-MT{2 * i + 1}" for i in range(1, 3)])
    assert len(deleted) == 2
    deleted = ws.member_tokens.delete(*[f"ak-MT{2 * i + 1}" for i in range(1, 6)], allow_missing=True)
    assert len(deleted) == 3

    tokens = ws.member_tokens.list()
    assert len(tokens) == 7
    assert set(token.token_id for token in tokens) == {
        "ak-MT1",
        "ak-MT2",
        "ak-MT4",
        "ak-MT6",
        "ak-MT8",
        "ak-MT10",
        "ak-MT12",
    }

    with pytest.raises(NotFoundError, match="Token not found"):
        ws.member_tokens.delete("ak-MT3")

    deleted = ws.member_tokens.delete("ak-MT3", allow_missing=True)
    assert len(deleted) == 0

    assert set(token.token_id for token in tokens if token.disabled) == set()
    ws.member_tokens.disable("ak-MT1", "ak-MT2", "ak-MT4")
    tokens = ws.member_tokens.list()
    assert len(tokens) == 7
    assert set(token.token_id for token in tokens if token.disabled) == {"ak-MT1", "ak-MT2", "ak-MT4"}

    ws.member_tokens.disable("ak-MT2", "ak-MT4", "ak-MT6")
    tokens = ws.member_tokens.list()
    assert set(token.token_id for token in tokens if token.disabled) == {"ak-MT1", "ak-MT2", "ak-MT4", "ak-MT6"}

    with pytest.raises(NotFoundError, match="Token not found"):
        ws.member_tokens.disable("ak-MT3")

    ws.member_tokens.enable("ak-MT2", "ak-MT4")
    tokens = ws.member_tokens.list()
    assert set(token.token_id for token in tokens if token.disabled) == {"ak-MT1", "ak-MT6"}

    ws.member_tokens.enable("ak-MT2", "ak-MT4", "ak-MT1", "ak-MT6")
    tokens = ws.member_tokens.list()
    assert set(token.token_id for token in tokens if token.disabled) == set()

    with pytest.raises(NotFoundError, match="Token not found"):
        ws.member_tokens.enable("ak-MT3")


def test_workspace_audit_logs_newest(servicer, client):
    until = datetime(2026, 10, 12, tzinfo=timezone.utc)
    servicer.audit_log_entries = [
        api_pb2.AuditLogEntry(
            id=f"al-{n}",
            event_at_timestamp=Timestamp(seconds=int(datetime(2026, 10, 11, 14, n, tzinfo=timezone.utc).timestamp())),
            action="app.deploy",
            status="success",
            actor=api_pb2.AuditLogActor(type="user", id="us-1"),
            targets=[api_pb2.AuditLogTarget(type="app", id=f"ap-{n}")],
            context=api_pb2.AuditLogContext(environment_id="en-1", source="sdk", ip_address="203.0.113.7"),
            metadata_json='{"app_name": "demo"}',
        )
        for n in range(9, 0, -1)
    ]
    with servicer.intercept() as ctx:
        entries = list(Workspace.from_context(client=client).audit_logs.fetch(until=until))
        request = ctx.pop_request("AuditLogList")
        assert request.limit == 100
        assert request.oldest_ms == int((until - timedelta(days=7)).timestamp() * 1000)
    assert [entry.id for entry in entries] == [f"al-{n}" for n in range(1, 10)]
    entry = entries[0]
    assert isinstance(entry, AuditEvent)
    assert entry.timestamp == datetime(2026, 10, 11, 14, 1, tzinfo=timezone.utc)
    assert entry.actor.id == "us-1"
    assert entry.actor == AuditEvent.Actor(type="user", id="us-1", name="", email="")
    assert entry.targets[0].id == "ap-1"
    assert entry.targets == [AuditEvent.Target(type="app", id="ap-1")]
    assert entry.context.environment_id == "en-1"
    assert entry.context == AuditEvent.Context(
        workspace_id="",
        environment_id="en-1",
        environment_name="",
        source="sdk",
        ip_address=ipaddress.IPv4Address("203.0.113.7"),
    )
    assert entry.metadata == {"app_name": "demo"}
    with pytest.raises(FrozenInstanceError):
        entry.action = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2001:db8::1", ipaddress.IPv6Address("2001:db8::1")),
        ("", None),
        ("203.0.113.7:46532", None),
    ],
)
def test_workspace_audit_logs_ip_address(servicer, client, raw, expected):
    servicer.audit_log_entries = [
        api_pb2.AuditLogEntry(
            id="al-1",
            event_at_timestamp=Timestamp(seconds=int(datetime(2026, 10, 11, 14, tzinfo=timezone.utc).timestamp())),
            context=api_pb2.AuditLogContext(ip_address=raw),
        )
    ]
    until = datetime(2026, 10, 12, tzinfo=timezone.utc)
    [entry] = Workspace.from_context(client=client).audit_logs.fetch(until=until)
    assert entry.context.ip_address == expected


@pytest.mark.parametrize(
    "name, email, environment_name",
    [
        ("Ada Lovelace", "ada@example.com", "main"),
        ("", "", ""),
    ],
)
def test_workspace_audit_logs_resolved_names(servicer, client, name, email, environment_name):
    servicer.audit_log_entries = [
        api_pb2.AuditLogEntry(
            id="al-1",
            event_at_timestamp=Timestamp(seconds=int(datetime(2026, 10, 11, 14, tzinfo=timezone.utc).timestamp())),
            actor=api_pb2.AuditLogActor(type="user", id="us-1", name=name, email=email),
            context=api_pb2.AuditLogContext(environment_id="en-1", environment_name=environment_name),
        )
    ]
    until = datetime(2026, 10, 12, tzinfo=timezone.utc)
    [entry] = Workspace.from_context(client=client).audit_logs.fetch(until=until)
    assert (entry.actor.id, entry.actor.name, entry.actor.email) == ("us-1", name, email)
    assert (entry.context.environment_id, entry.context.environment_name) == ("en-1", environment_name)


@pytest.mark.asyncio
async def test_workspace_audit_logs_since_with_limit(servicer, client):
    since = datetime(2026, 10, 11, tzinfo=timezone.utc)
    until = since + timedelta(days=1)
    servicer.audit_log_entries = [
        api_pb2.AuditLogEntry(
            id=f"al-{n}",
            event_at_timestamp=Timestamp(seconds=int(datetime(2026, 10, 11, 14, n, tzinfo=timezone.utc).timestamp())),
        )
        for n in range(9, 0, -1)
    ]
    with servicer.intercept() as ctx:
        manager = Workspace.from_context(client=client).audit_logs
        entries = [entry async for entry in manager.fetch.aio("status:success", since=since, until=until, limit=2)]
        request = ctx.pop_request("AuditLogList")
        assert request.q == "status:success"
        assert request.oldest_ms == int(since.timestamp() * 1000)
        assert request.newest_ms == int(until.timestamp() * 1000)
    assert [entry.id for entry in entries] == ["al-8", "al-9"]


@pytest.mark.parametrize(
    "timestamp, expected",
    [
        (Timestamp(seconds=1791727200, nanos=123000000), datetime(2026, 10, 11, 14, 0, 0, 123000, tzinfo=timezone.utc)),
        (Timestamp(), datetime(1970, 1, 1, tzinfo=timezone.utc)),
        (Timestamp(seconds=-1, nanos=877000000), datetime(1969, 12, 31, 23, 59, 59, 877000, tzinfo=timezone.utc)),
        (None, datetime(1970, 1, 1, tzinfo=timezone.utc)),
    ],
)
def test_workspace_audit_logs_typed_timestamp(servicer, client, timestamp, expected):
    servicer.audit_log_entries = [
        api_pb2.AuditLogEntry(id="al-1", event_at="not a timestamp", event_at_timestamp=timestamp)
    ]
    [entry] = Workspace.from_context(client=client).audit_logs.fetch()
    assert entry.timestamp == expected
    assert entry.timestamp.tzinfo is timezone.utc


def test_workspace_audit_logs_invalid_timestamp(servicer, client):
    servicer.audit_log_entries = [api_pb2.AuditLogEntry(id="al-1", event_at_timestamp=Timestamp(nanos=1000000000))]
    with pytest.raises(ValueError):
        list(Workspace.from_context(client=client).audit_logs.fetch())


def test_workspace_audit_logs_count(servicer, client):
    until = datetime(2026, 10, 12, tzinfo=timezone.utc)
    since = until - timedelta(days=3)
    servicer.audit_log_entries = [api_pb2.AuditLogEntry(id="al-1")]
    with servicer.intercept() as ctx:
        assert (
            Workspace.from_context(client=client).audit_logs.count("action:app.deploy", since=since, until=until) == 1
        )
        request = ctx.pop_request("AuditLogCount")
        assert request.q == "action:app.deploy"
        assert request.oldest_ms == int(since.timestamp() * 1000)
        assert request.newest_ms == int(until.timestamp() * 1000)


def test_workspace_audit_logs_count_requires_since(client):
    with pytest.raises(TypeError, match="since"):
        getattr(Workspace.from_context(client=client).audit_logs, "count")()


def test_workspace_audit_logs_count_defaults_until_to_now(servicer, client):
    since = datetime.now(timezone.utc) - timedelta(days=1)
    before = datetime.now(timezone.utc)
    with servicer.intercept() as ctx:
        assert Workspace.from_context(client=client).audit_logs.count(since=since) == 0
        request = ctx.pop_request("AuditLogCount")
    assert request.oldest_ms == int(since.timestamp() * 1000)
    assert int(before.timestamp() * 1000) <= request.newest_ms <= int(datetime.now(timezone.utc).timestamp() * 1000)


@pytest.mark.parametrize("limit", [0, -1])
def test_workspace_audit_logs_invalid_newest_limit(client, limit):
    with pytest.raises(InvalidError):
        list(Workspace.from_context(client=client).audit_logs.fetch(limit=limit))


@pytest.mark.parametrize("limit", [0, -1])
def test_workspace_audit_logs_invalid_since_limit(client, limit):
    until = datetime(2026, 10, 12, tzinfo=timezone.utc)
    with pytest.raises(InvalidError):
        list(
            Workspace.from_context(client=client).audit_logs.fetch(
                since=until - timedelta(days=1), until=until, limit=limit
            )
        )


@pytest.mark.parametrize("with_since", [False, True])
def test_workspace_audit_logs_empty(servicer, client, with_since):
    until = datetime(2026, 10, 12, tzinfo=timezone.utc)
    since = until - timedelta(days=1) if with_since else None
    manager = Workspace.from_context(client=client).audit_logs
    assert list(manager.fetch(since=since, until=until)) == []
    assert manager.count(since=until - timedelta(days=1), until=until) == 0


@pytest.mark.parametrize("operation", ["fetch", "count"])
def test_workspace_audit_logs_rpc_errors(servicer, client, operation):
    async def denied(servicer, stream):
        await stream.recv_message()
        raise GRPCError(Status.PERMISSION_DENIED, "Audit logs unavailable")

    with servicer.intercept() as ctx:
        ctx.set_responder("AuditLogSettingsGet", denied)
        manager = Workspace.from_context(client=client).audit_logs
        with pytest.raises(PermissionDeniedError, match="Audit logs unavailable"):
            if operation == "fetch":
                list(manager.fetch())
            else:
                manager.count(since=datetime(2026, 10, 11, tzinfo=timezone.utc))


@pytest.mark.parametrize("metadata, expected", [("", {}), ("invalid", "invalid"), ('{"key": 1}', {"key": 1})])
def test_workspace_audit_logs_metadata(servicer, client, metadata, expected):
    servicer.audit_log_entries = [
        api_pb2.AuditLogEntry(
            id="al-1", event_at_timestamp=Timestamp(seconds=1, nanos=123000000), metadata_json=metadata
        )
    ]
    entry = next(
        Workspace.from_context(client=client).audit_logs.fetch(until=datetime(2026, 10, 12, tzinfo=timezone.utc))
    )
    assert entry.metadata == expected


def test_workspace_audit_logs_naive_bounds(servicer, client):
    local_tz = datetime.now().astimezone().tzinfo
    since = datetime(2026, 10, 11)
    until = datetime(2026, 10, 12)
    servicer.audit_log_entries = [api_pb2.AuditLogEntry(id="al-1")]
    with servicer.intercept() as ctx:
        assert Workspace.from_context(client=client).audit_logs.count(since=since, until=until) == 1
        request = ctx.pop_request("AuditLogCount")
        assert request.oldest_ms == int(since.replace(tzinfo=local_tz).timestamp() * 1000)
        assert request.newest_ms == int(until.replace(tzinfo=local_tz).timestamp() * 1000)


def test_workspace_audit_logs_since_large_limit(servicer, client):
    since = datetime(2026, 10, 11, tzinfo=timezone.utc)
    servicer.audit_log_entries = [
        api_pb2.AuditLogEntry(id="al-1", event_at_timestamp=Timestamp(seconds=1, nanos=123000000))
    ]
    entries = list(
        Workspace.from_context(client=client).audit_logs.fetch(since=since, until=since + timedelta(days=1), limit=5001)
    )
    assert [entry.id for entry in entries] == ["al-1"]


def test_workspace_audit_logs_large_limit_returns_newest(servicer, client):
    since = datetime(2026, 10, 11, tzinfo=timezone.utc)
    servicer.audit_log_entries = [
        api_pb2.AuditLogEntry(id=f"al-{n}", event_at_timestamp=Timestamp(seconds=1, nanos=123000000))
        for n in range(5002, 0, -1)
    ]
    with servicer.intercept() as ctx:
        entries = list(
            Workspace.from_context(client=client).audit_logs.fetch(
                since=since, until=since + timedelta(days=1), limit=5001
            )
        )
        assert len(ctx.get_requests("AuditLogPlan")) == 1
    assert [entry.id for entry in entries] == [f"al-{n}" for n in range(2, 5003)]
