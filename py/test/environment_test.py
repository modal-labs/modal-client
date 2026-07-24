# Copyright Modal Labs 2025
import pytest

from google.protobuf.empty_pb2 import Empty
from grpclib import GRPCError, Status

from modal.environments import Environment
from modal.exception import DeprecationError, InvalidError, WorkspaceManagementError
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


def test_environment_objects_create(servicer, client):
    Environment.objects.create("new-env", client=client)
    assert "new-env" in servicer.environments
    assert servicer.environments["new-env"] not in servicer.environment_managed

    Environment.objects.create("restricted-env", restricted=True, client=client)
    assert "restricted-env" in servicer.environments
    assert servicer.environment_managed[servicer.environments["restricted-env"]] is True

    Environment.objects.create("public-env", experimental_options={"is_public": True}, client=client)
    assert "public-env" in servicer.environments
    assert servicer.environment_type[servicer.environments["public-env"]] == api_pb2.ENVIRONMENT_TYPE_PUBLIC


def test_environment_objects_list(servicer, client):
    envs = Environment.objects.list(client=client)
    names = [e.name for e in envs]
    assert names == list(servicer.environments)


def test_environment_objects_delete(servicer, client):
    Environment.objects.create("to-delete", client=client)
    assert "to-delete" in servicer.environments

    Environment.objects.delete("to-delete", client=client)
    assert "to-delete" not in servicer.environments


def test_environment_get_roles(servicer, client):
    env = Environment.from_name("main", client=client)
    env.hydrate()

    assert env.roles.list() == {
        "users": {"alice": "contributor", "bob": "contributor", "carol": "contributor"},
        "service_users": {"alice-bot": "contributor", "ops-bot": "contributor"},
    }

    servicer.environment_members[env.object_id] = {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
        "us-2": api_pb2.ENVIRONMENT_ROLE_VIEWER,
        "sv-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
    }
    assert env.roles.list() == {
        "users": {"alice": "contributor", "bob": "viewer", "carol": "contributor"},
        "service_users": {"alice-bot": "contributor", "ops-bot": "contributor"},
    }


def test_environment_update_roles(servicer, client):
    env = Environment.from_name("main", client=client)

    env.roles.update(users={"alice": "contributor"}, service_users={"alice-bot": "viewer"})
    assert servicer.environment_members[env.object_id] == {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
        "sv-1": api_pb2.ENVIRONMENT_ROLE_VIEWER,
    }

    # Updating one role does not remove unrelated members
    env.roles.update(users={"alice": "viewer", "bob": "contributor"})
    assert servicer.environment_members[env.object_id] == {
        "us-1": api_pb2.ENVIRONMENT_ROLE_VIEWER,
        "us-2": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
        "sv-1": api_pb2.ENVIRONMENT_ROLE_VIEWER,
    }


def test_environment_update_roles_unknown_principal(client):
    env = Environment.from_name("main", client=client)

    with pytest.raises(InvalidError, match="User 'eve' not found in workspace"):
        env.roles.update(users={"eve": "viewer"})

    with pytest.raises(InvalidError, match="Service user 'unknown-bot' not found in workspace"):
        env.roles.update(service_users={"unknown-bot": "viewer"})


def test_environment_members_remove(servicer, client):
    # `remove` lives only on the deprecated `members` manager, not on `roles`.
    env = Environment.from_name("main", client=client)
    env.hydrate()
    servicer.environment_members[env.object_id] = {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
        "us-2": api_pb2.ENVIRONMENT_ROLE_VIEWER,
        "sv-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
    }

    assert not hasattr(env.roles, "remove")
    with pytest.warns(DeprecationError, match="deprecated"):
        env.members.remove(users=["alice"], service_users=["alice-bot"])
    assert servicer.environment_members[env.object_id] == {
        "us-2": api_pb2.ENVIRONMENT_ROLE_VIEWER,
    }


def test_environment_update_roles_aggregates_errors(servicer, client):
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
            env.roles.update(
                users={"alice": "viewer", "bob": "viewer"},
                service_users={"alice-bot": "viewer"},
            )

    msg = str(excinfo.value)
    assert "2 errors occurred while updating Environment roles" in msg
    assert "User 'alice': missing principal" in msg
    assert "Service user 'alice-bot': missing principal" in msg
    assert "bob" not in msg  # bob's RPC succeeded


def test_environment_update_roles_single_error_labeled(servicer, client):
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
            env.roles.update(users={"alice": "viewer", "bob": "viewer"})

    msg = str(excinfo.value)
    assert "1 error occurred while updating Environment roles" in msg
    assert "User 'alice': boom" in msg


def test_environment_members_remove_user_and_service_user_with_same_name(servicer, client):
    """A user and a service user can share a name — both removals must be sent."""
    env = Environment.from_name("main", client=client)
    env.hydrate()
    servicer.workspace_service_users["sv-3"] = "alice"  # collide with the workspace user named "alice"
    servicer.environment_members[env.object_id] = {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,  # user "alice"
        "sv-3": api_pb2.ENVIRONMENT_ROLE_VIEWER,  # service user "alice"
    }

    with pytest.warns(DeprecationError, match="deprecated"):
        env.members.remove(users=["alice"], service_users=["alice"])

    assert servicer.environment_members[env.object_id] == {}


def test_environment_members_remove_unknown_principal(client):
    env = Environment.from_name("main", client=client)

    with pytest.warns(DeprecationError, match="deprecated"):
        with pytest.raises(InvalidError, match="User 'eve' not found in workspace"):
            env.members.remove(users=["eve"])

    with pytest.warns(DeprecationError, match="deprecated"):
        with pytest.raises(InvalidError, match="Service user 'unknown-bot' not found in workspace"):
            env.members.remove(service_users=["unknown-bot"])


def test_environment_members_remove_without_role_is_noop(servicer, client):
    env = Environment.from_name("main", client=client)
    env.hydrate()
    servicer.environment_members[env.object_id] = {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
    }

    with pytest.warns(DeprecationError, match="deprecated"):
        env.members.remove(users=["bob"], service_users=["alice-bot"])
    assert servicer.environment_members[env.object_id] == {
        "us-1": api_pb2.ENVIRONMENT_ROLE_CONTRIBUTOR,
    }


def test_environment_members_deprecated(servicer, client):
    # `Environment.members` is a deprecated alias for `Environment.roles`; each method warns.
    env = Environment.from_name("main", client=client)
    env.hydrate()

    with pytest.warns(DeprecationError, match="deprecated"):
        result = env.members.list()
    assert result == {
        "users": {"alice": "contributor", "bob": "contributor", "carol": "contributor"},
        "service_users": {"alice-bot": "contributor", "ops-bot": "contributor"},
    }
