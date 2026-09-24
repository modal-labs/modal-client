# Copyright Modal Labs 2022
import os
import pytest
import sys
import tempfile
import time
from unittest import mock

from modal import App, Sandbox, Secret
from modal._load_context import LoadContext
from modal._resolver import Resolver
from modal._utils.async_utils import TaskContext, synchronizer
from modal.exception import AlreadyExistsError, InvalidError, NotFoundError
from modal_proto import api_pb2

from .supports.skip import skip_old_py, skip_windows


def dummy(): ...


def test_secret_from_dict(servicer, client):
    app = App(include_source=False)
    secret = Secret.from_dict({"FOO": "hello, world"})
    app.function(secrets=[secret])(dummy)
    with app.run(client=client):
        assert secret.object_id == "st-0"
        assert servicer.secrets["st-0"] == {"FOO": "hello, world"}


@skip_old_py("python-dotenv requires python3.8 or higher", (3, 8))
def test_secret_from_dotenv(servicer, client):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(os.path.join(tmpdirname, ".env"), "w") as f:
            f.write("# My settings\nUSER=user\nPASSWORD=abc123\n")

        with open(os.path.join(tmpdirname, ".env-dev"), "w") as f:
            f.write("# My settings\nUSER=user2\nPASSWORD=abc456\n")

        app = App(include_source=False)
        secret = Secret.from_dotenv(tmpdirname)
        app.function(secrets=[secret])(dummy)
        with app.run(client=client):
            assert secret.object_id == "st-0"
            assert secret._get_keys() == {"USER", "PASSWORD"}
            assert servicer.secrets["st-0"] == {"USER": "user", "PASSWORD": "abc123"}

        app = App(include_source=False)
        secret = Secret.from_dotenv(tmpdirname, filename=".env-dev")
        app.function(secrets=[secret])(dummy)
        with app.run(client=client):
            assert secret.object_id == "st-1"
            assert secret._get_keys() == {"USER", "PASSWORD"}
            assert servicer.secrets["st-1"] == {"USER": "user2", "PASSWORD": "abc456"}


@skip_windows("uses sandbox to repro app-ness of secret, and sandboxe tests use subprocess")
def test_secret_from_dotenv_lazy(client, servicer, monkeypatch):
    monkeypatch.setenv("MODAL_SANDBOX_V2", "0")
    with servicer.intercept() as ctx:
        dummy_app = App.lookup("blah", client=client, create_if_missing=True)
        Sandbox.create(client=client, secrets=[Secret.from_dotenv()], app=dummy_app)
        req = ctx.pop_request("SecretGetOrCreate")
        assert (
            req.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_ANONYMOUS_OWNED_BY_APP
        )  # use the sandbox's app id
        assert req.app_id == dummy_app.app_id

        ctx.calls.clear()

        Secret.from_dotenv(client=client).hydrate()
        req = ctx.pop_request("SecretGetOrCreate")
        # there is no app in this case - not sure if this should be allowed long term...
        assert req.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL


def test_secret_load_env_dict_from_dict():
    secret = Secret.from_dict({"FOO": "hello, world", "SKIP": None, "BAR": "1234"})
    _secret = synchronizer._translate_in(secret)
    # None values are filtered out
    assert _secret._load_env_dict() == {"FOO": "hello, world", "BAR": "1234"}


@skip_old_py("python-dotenv requires python3.8 or higher", (3, 8))
def test_secret_load_env_dict_from_dotenv():
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(os.path.join(tmpdirname, ".env"), "w") as f:
            f.write("# My settings\nUSER=user\nPASSWORD=abc123\n")

        secret = Secret.from_dotenv(tmpdirname)
        _secret = synchronizer._translate_in(secret)
        assert _secret._load_env_dict() == {"USER": "user", "PASSWORD": "abc123"}


@mock.patch.dict(os.environ, {"FOO": "easy", "BAR": "1234"})
def test_secret_from_local_environ(servicer, client):
    app = App(include_source=False)
    secret = Secret.from_local_environ(["FOO", "BAR"])
    app.function(secrets=[secret])(dummy)
    with app.run(client=client):
        assert secret.object_id == "st-0"
        assert servicer.secrets["st-0"] == {"FOO": "easy", "BAR": "1234"}

    with pytest.raises(InvalidError, match="NOTFOUND"):
        Secret.from_local_environ(["FOO", "NOTFOUND"])


def test_init_types():
    with pytest.raises(InvalidError):
        Secret.from_dict({"foo": 1.0})  # type: ignore


def test_secret_from_dict_none(servicer, client):
    app = App(include_source=False)
    secret = Secret.from_dict({"FOO": os.getenv("xyz"), "BAR": os.environ.get("abc"), "BAZ": "baz"})
    app.function(secrets=[secret])(dummy)
    with app.run(client=client):
        assert servicer.secrets["st-0"] == {"BAZ": "baz"}


def test_secret_from_name(servicer, client):
    # Deploy secret
    name = "my-secret"
    Secret.objects.create(name, {"FOO": "123"}, client=client)

    # Look up secret
    secret = Secret.from_name(name)
    assert secret.name == name
    secret.hydrate(client)
    secret_id = secret.object_id

    info = secret.info()
    assert info.name == name
    assert info.created_by == servicer.default_username
    assert info.environment_name == servicer.get_environment()

    # Look up secret through app
    app = App()
    secret = Secret.from_name("my-secret")
    app.function(secrets=[secret])(dummy)
    with app.run(client=client):
        assert secret.object_id == secret_id

    Secret.objects.delete("my-secret", client=client)
    with pytest.raises(NotFoundError):
        Secret.from_name("my-secret").hydrate(client)
    Secret.objects.delete("my-secret", client=client, allow_missing=True)


def test_secret_from_id(servicer, client):
    # Deploy secret
    name = "my-secret"
    Secret.objects.create(name, {"FOO": "123"}, client=client)

    # Look up secret
    secret = Secret.from_name(name)
    assert secret.name == name
    secret.hydrate(client)
    secret_id = secret.object_id

    # Now look it up by ID
    secret = Secret._from_id(secret_id, client=client)
    assert not secret._is_hydrated

    secret.hydrate(client)

    assert secret.object_id == secret_id
    info = secret.info()
    assert info.name == name
    assert info.created_by == servicer.default_username
    assert info.environment_name == servicer.get_environment()

    # Look up secret through app
    app = App()
    secret = Secret._from_id(secret_id, client=client)
    app.function(secrets=[secret])(dummy)
    with app.run(client=client):
        assert secret.object_id == secret_id

        info = secret.info()
        assert info.name == name
        assert info.created_by == servicer.default_username
        assert info.environment_name == servicer.get_environment()

    Secret.objects.delete(name, client=client)
    with pytest.raises(NotFoundError):
        Secret._from_id(secret_id, client=client).hydrate(client)


def test_secret_from_name_double_resolve(client, servicer):
    # Checks that Resolver logic is set up to *not* re-lookup
    # secrets that are defined by name, since those are unlikely
    # to change their id (`skip_reload=True`).
    name = "my-secret"
    Secret.objects.create(name, {"FOO": "123"}, client=client)
    secret = Secret.from_name(name)

    @synchronizer.wrap
    async def wrapped_test(secret, client):
        assert secret.name == name

        resolver = Resolver()
        async with TaskContext() as tc:
            load_context = LoadContext(client=client, task_context=tc)
            await resolver.load(secret, load_context)
        return secret.object_id

    with servicer.intercept() as ctx:
        wrapped_test(secret, client)  # type: ignore
        req = ctx.pop_request("SecretGetOrCreate")
        assert req.deployment_name == name
        assert req.object_creation_type == api_pb2.OBJECT_CREATION_TYPE_UNSPECIFIED
        ctx.calls.clear()
        wrapped_test(secret, client)  # type: ignore
        assert len(ctx.calls) == 0


def test_secret_list(servicer, client):
    for i in range(5):
        Secret.objects.create(f"test-secret-{i}", {"FOO": "123"}, client=client)
    if sys.platform == "win32":
        time.sleep(1 / 32)

    secrets = Secret.objects.list(client=client)
    assert len(secrets) == 5
    assert all(s.name.startswith("test-secret-") for s in secrets)
    assert all(s.info().created_by == servicer.default_username for s in secrets)


def test_secret_create(servicer, client):
    env_dict = {"FOO": "123"}
    Secret.objects.create(name="test-secret-create", env_dict=env_dict, client=client)
    Secret.from_name("test-secret-create").hydrate(client)
    with pytest.raises(AlreadyExistsError):
        Secret.objects.create(name="test-secret-create", env_dict=env_dict, client=client)
    Secret.objects.create(name="test-secret-create", env_dict=env_dict, allow_existing=True, client=client)
    with pytest.raises(InvalidError, match="Invalid Secret name"):
        Secret.objects.create(name="has space", env_dict=env_dict, client=client)


def test_secret_update(servicer, client):
    # Create a secret first
    Secret.objects.create(name="test-secret-update", env_dict={"FOO": "123", "BAR": "456"}, client=client)
    secret = Secret.from_name("test-secret-update")
    secret.hydrate(client)

    # Verify initial state
    assert servicer.secrets[secret.object_id] == {"FOO": "123", "BAR": "456"}

    # Update: overwrite one key, add a new key
    secret.update({"FOO": "new-value", "BAZ": "789"})

    # BAR should be unchanged, FOO overwritten, BAZ added
    assert servicer.secrets[secret.object_id] == {"FOO": "new-value", "BAR": "456", "BAZ": "789"}

    # Validate types at runtime
    with pytest.raises(InvalidError):
        secret.update("not-a-dict")  # type: ignore

    with pytest.raises(InvalidError):
        secret.update({1: "val"})  # type: ignore

    with pytest.raises(InvalidError):
        secret.update({"key": 123})  # type: ignore


def test_secret_keys(client, servicer):
    # Remote reference
    Secret.objects.create(name="test-secret-update", env_dict={"FOO": "123", "BAR": "456"}, client=client)
    secret = Secret.from_name("test-secret-update", client=client)

    # Getting keys on a remote reference requires hydration
    keys = secret._get_keys()
    assert secret._is_hydrated
    assert keys == {"FOO", "BAR"}

    # Update: overwrite one key, add a new key
    secret.update({"FOO": "new-value", "BAZ": "789"})

    # Getting keys without refresh shouldn't make a new RPC, but `.update()` should update the keyset
    with servicer.intercept() as ctx:
        keys = secret._get_keys(refresh=False)
        assert len(ctx.calls) == 0

    assert keys == {"FOO", "BAR", "BAZ"}

    secret = Secret.from_dict({"a": "b"})
    with servicer.intercept() as ctx:
        keys = secret._get_keys()
        assert len(ctx.calls) == 0

    assert keys == {"a"}

    # This hydration is only here to propagate the test client. Since `.update` is `@live_method`, it
    # would hydrate anyway regardless
    secret.hydrate(client)
    secret.update({"c": "d"})

    with servicer.intercept() as ctx:
        keys = secret._get_keys()
        assert len(ctx.calls) == 0

    assert keys == {"a", "c"}

    secret1 = Secret.from_name("test-secret-update", client=client)
    secret1.hydrate()
    secret2 = Secret.from_name("test-secret-update", client=client)
    secret2.hydrate()

    secret1.update({"a": "b"})
    secret2.update({"c": "d"})

    assert "a" in secret1._get_keys()
    assert "c" not in secret1._get_keys()
    assert "a" not in secret2._get_keys()
    assert "c" in secret2._get_keys()

    assert secret1._get_keys(refresh=True) == secret2._get_keys(refresh=True)


def test_secret_keys_respect_environment(client):
    Secret.objects.create("test-secret", {"key_in_env1": "value"}, client=client)
    Secret.objects.create("test-secret", {"key_in_env2": "value"}, environment_name="env2", client=client)

    s1 = Secret.from_name("test-secret", client=client)  # should only have `key_in_env1`
    s2 = Secret.from_name("test-secret", environment_name="env2", client=client)  # should only have `key_in_env2`
    s3 = Secret.from_name("test-secret", environment_name="env2", client=client)  # should only have `key_in_env2`

    # First call will hydrate
    assert s1._get_keys() == {"key_in_env1"}
    assert s2._get_keys() == {"key_in_env2"}

    s3.update({"this_key_should_only_show_up_in_env2_also": "a"})

    assert s1._get_keys() == {"key_in_env1"}
    assert s1._get_keys(refresh=True) == {"key_in_env1"}

    assert s2._get_keys() == {"key_in_env2"}
    assert s2._get_keys(refresh=True) == {"key_in_env2", "this_key_should_only_show_up_in_env2_also"}

    # Make sure that if this secret is hydrated within a running App in a separate environment,
    # `._get_keys()` respects that environment when refreshing
    s4 = Secret.from_name("test-secret", client=client)

    app = App()

    @app.function(secrets=[s4], serialized=True)
    def f():
        pass

    with app.run(environment_name="env2", client=client):
        f.local()

    # Since s4 is resolved w.r.t. the environment of the running App, which is `env2`, we should see
    # `env2` keys here
    assert s4._get_keys() == {"key_in_env2", "this_key_should_only_show_up_in_env2_also"}

    s3.update({"one_more_update": "a"})

    # Subsequent refreshes should maintain the environment that the underlying handle was hydrated in
    assert s4._get_keys(refresh=True) == {"key_in_env2", "this_key_should_only_show_up_in_env2_also", "one_more_update"}
