# Copyright Modal Labs 2025
import pytest

from google.protobuf.empty_pb2 import Empty
from grpclib import GRPCError, Status

from modal._environments import _role_from_proto, _role_to_proto
from modal.environments import Environment
from modal.exception import InvalidError, WorkspaceManagementError
from modal_proto import api_pb2


def test_environment_from_name(servicer, client):
    env = Environment.from_name("my-env", create_if_missing=True)
    env.hydrate(client)
    assert env.name == "my-env"
    assert env.object_id == servicer.environments["my-env"]


def test_environment_from_name_existing(servicer, client):
    # "main" is pre-populated in the servicer
    env = Environment.from_name("main")
    env.hydrate(client)
    assert env.name == "main"
    assert env.object_id == "en-1"


def test_environment_from_name_lazy_hydration(servicer, client):
    with servicer.intercept() as ctx:
        env = Environment.from_name("lazy-env", create_if_missing=True, client=client)
        assert len(ctx.get_requests("EnvironmentGetOrCreate")) == 0

        # Explicit hydrate triggers the RPC
        env.hydrate()
        assert len(ctx.get_requests("EnvironmentGetOrCreate")) == 1


def test_environment_from_context_default(servicer, client):
    """When no local config is set, from_context resolves to the server-defined default."""
    with servicer.intercept() as ctx:
        env = Environment.from_context(client=client)
        env.hydrate()

    # Server resolves empty name to the default environment ("main")
    req = ctx.pop_request("EnvironmentGetOrCreate")
    assert req.deployment_name == ""
    assert env.name == "main"
    assert env.object_id == "en-1"


def test_environment_from_context_env_var(servicer, client, monkeypatch):
    """MODAL_ENVIRONMENT env var takes highest priority."""
    monkeypatch.setenv("MODAL_ENVIRONMENT", "from-env-var")

    with servicer.intercept() as ctx:
        env = Environment.from_context(client=client)
        env.hydrate()

    req = ctx.pop_request("EnvironmentGetOrCreate")
    assert req.deployment_name == "from-env-var"
    assert env.name == "from-env-var"


def test_environment_from_context_toml_config(servicer, client, modal_config):
    """Config file environment is used when the env var is not set."""
    toml = """
    [default]
    environment = "from-toml"
    """
    with modal_config(toml):
        with servicer.intercept() as ctx:
            env = Environment.from_context(client=client)
            env.hydrate()

    req = ctx.pop_request("EnvironmentGetOrCreate")
    assert req.deployment_name == "from-toml"
    assert env.name == "from-toml"


def test_environment_from_context_env_var_overrides_toml(servicer, client, monkeypatch, modal_config):
    """MODAL_ENVIRONMENT env var takes priority over the config file."""
    toml = """
    [default]
    environment = "from-toml"
    """
    monkeypatch.setenv("MODAL_ENVIRONMENT", "from-env-var")
    with modal_config(toml):
        with servicer.intercept() as ctx:
            env = Environment.from_context(client=client)
            env.hydrate()

    req = ctx.pop_request("EnvironmentGetOrCreate")
    assert req.deployment_name == "from-env-var"
    assert env.name == "from-env-var"


@pytest.mark.parametrize("name", ["has space", "has/slash", "a" * 65])
def test_environment_invalid_name(servicer, client, name):
    with pytest.raises(InvalidError, match="Invalid Environment name"):
        Environment.from_name(name).hydrate(client)


def test_environment_role_proto_round_trip():
    for role in ("viewer", "contributor"):
        assert _role_from_proto(_role_to_proto(role)) == role

    assert _role_to_proto("viewer") == api_pb2.ENVIRONMENT_ROLE_VIEWER
    assert _role_to_proto("contributor") == api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR
    assert _role_from_proto(api_pb2.ENVIRONMENT_ROLE_VIEWER) == "viewer"
    assert _role_from_proto(api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR) == "contributor"

    with pytest.raises(ValueError, match="Unknown environment role"):
        _role_from_proto(api_pb2.ENVIRONMENT_ROLE_UNSPECIFIED)

    with pytest.raises(ValueError, match="Unknown environment role"):
        _role_from_proto(999)

    with pytest.raises(InvalidError, match="Invalid Environment role"):
        _role_to_proto("admin")


def test_environment_objects_create(servicer, client):
    Environment.objects.create("new-env", client=client)
    assert "new-env" in servicer.environments
    assert servicer.environments["new-env"] not in servicer.environment_managed

    Environment.objects.create("restricted-env", restricted=True, client=client)
    assert "restricted-env" in servicer.environments
    assert servicer.environment_managed[servicer.environments["restricted-env"]] is True


def test_environment_objects_list(servicer, client):
    envs = Environment.objects.list(client=client)
    names = [e.name for e in envs]
    assert names == list(servicer.environments)


def test_environment_objects_delete(servicer, client):
    Environment.objects.create("to-delete", client=client)
    assert "to-delete" in servicer.environments

    Environment.objects.delete("to-delete", client=client)
    assert "to-delete" not in servicer.environments


def test_environment_get_members(servicer, client):
    env = Environment.from_name("main", client=client)
    env.hydrate()
    assert env.members.list() == {"users": {}, "service_users": {}}

    servicer.environment_members[env.object_id] = {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
        "us-2": api_pb2.ENVIRONMENT_ROLE_VIEWER,
        "sv-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
    }
    assert env.members.list() == {
        "users": {"alice": "contributor", "bob": "viewer"},
        "service_users": {"alice-bot": "contributor"},
    }


def test_environment_update_members(servicer, client):
    env = Environment.from_name("main", client=client)

    env.members.update(users={"alice": "contributor"}, service_users={"alice-bot": "viewer"})
    assert servicer.environment_members[env.object_id] == {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
        "sv-1": api_pb2.ENVIRONMENT_ROLE_VIEWER,
    }

    # Updating one role does not remove unrelated members
    env.members.update(users={"alice": "viewer", "bob": "contributor"})
    assert servicer.environment_members[env.object_id] == {
        "us-1": api_pb2.ENVIRONMENT_ROLE_VIEWER,
        "us-2": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
        "sv-1": api_pb2.ENVIRONMENT_ROLE_VIEWER,
    }


def test_environment_update_members_unknown_principal(client):
    env = Environment.from_name("main", client=client)

    with pytest.raises(InvalidError, match="User 'eve' not found in workspace"):
        env.members.update(users={"eve": "viewer"})

    with pytest.raises(InvalidError, match="Service user 'unknown-bot' not found in workspace"):
        env.members.update(service_users={"unknown-bot": "viewer"})


def test_environment_remove_members(servicer, client):
    env = Environment.from_name("main", client=client)
    env.hydrate()
    servicer.environment_members[env.object_id] = {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
        "us-2": api_pb2.ENVIRONMENT_ROLE_VIEWER,
        "sv-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
    }

    env.members.remove(users=["alice"], service_users=["alice-bot"])
    assert servicer.environment_members[env.object_id] == {
        "us-2": api_pb2.ENVIRONMENT_ROLE_VIEWER,
    }


def test_environment_update_members_aggregates_errors(servicer, client):
    env = Environment.from_name("main", client=client)
    env.hydrate()

    async def fail_for_alice_and_bot(servicer, stream):
        request = await stream.recv_message()
        if request.user_id == "us-1" or request.service_user_id == "sv-1":
            raise GRPCError(Status.NOT_FOUND, "missing principal")
        await stream.send_message(Empty())

    with servicer.intercept() as ctx:
        ctx.set_responder("EnvironmentRoleSet", fail_for_alice_and_bot)
        with pytest.raises(WorkspaceManagementError) as excinfo:
            env.members.update(
                users={"alice": "viewer", "bob": "viewer"},
                service_users={"alice-bot": "viewer"},
            )

    msg = str(excinfo.value)
    assert "2 errors occurred while updating Environment members" in msg
    assert "User 'alice': missing principal" in msg
    assert "Service user 'alice-bot': missing principal" in msg
    assert "bob" not in msg  # bob's RPC succeeded


def test_environment_update_members_single_error_labeled(servicer, client):
    env = Environment.from_name("main", client=client)
    env.hydrate()

    async def fail_for_alice(servicer, stream):
        request = await stream.recv_message()
        if request.user_id == "us-1":
            raise GRPCError(Status.NOT_FOUND, "boom")
        await stream.send_message(Empty())

    with servicer.intercept() as ctx:
        ctx.set_responder("EnvironmentRoleSet", fail_for_alice)
        with pytest.raises(WorkspaceManagementError) as excinfo:
            env.members.update(users={"alice": "viewer", "bob": "viewer"})

    msg = str(excinfo.value)
    assert "1 error occurred while updating Environment members" in msg
    assert "User 'alice': boom" in msg


def test_environment_remove_members_user_and_service_user_with_same_name(servicer, client):
    """A user and a service user can share a name — both removals must be sent."""
    env = Environment.from_name("main", client=client)
    env.hydrate()
    servicer.workspace_service_users["sv-3"] = "alice"  # collide with the workspace user named "alice"
    servicer.environment_members[env.object_id] = {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,  # user "alice"
        "sv-3": api_pb2.ENVIRONMENT_ROLE_VIEWER,  # service user "alice"
    }

    env.members.remove(users=["alice"], service_users=["alice"])

    assert servicer.environment_members[env.object_id] == {}


def test_environment_remove_members_not_a_member(servicer, client):
    env = Environment.from_name("main", client=client)
    env.hydrate()
    servicer.environment_members[env.object_id] = {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
    }

    with pytest.raises(InvalidError, match="User 'bob' is not a member"):
        env.members.remove(users=["bob"])

    with pytest.raises(InvalidError, match="Service user 'alice-bot' is not a member"):
        env.members.remove(service_users=["alice-bot"])
