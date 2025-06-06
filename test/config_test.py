# Copyright Modal Labs 2022
import os
import pathlib
import pytest
import subprocess
import sys

import toml

import modal
from modal._utils.async_utils import synchronize_api
from modal.config import Config, _lookup_workspace, config
from modal.exception import InvalidError


class CLIException(Exception):
    pass


def _cli(args, env={}):
    lib_dir = pathlib.Path(modal.__file__).parent.parent
    args = [sys.executable, "-m", "modal.cli.entry_point"] + args
    env = {
        **os.environ,
        **env,
        # For windows
        "PYTHONUTF8": "1",
    }
    ret = subprocess.run(args, cwd=lib_dir, env=env, capture_output=True)
    stdout = ret.stdout.decode()
    stderr = ret.stderr.decode()
    if ret.returncode != 0:
        raise CLIException(f"Failed with {ret.returncode} stdout: {stdout} stderr: {stderr}")
    return stdout


def _get_config(env={}):
    stdout = _cli(["config", "show", "--no-redact"], env=env)
    return eval(stdout)


def test_config():
    config = _get_config()
    assert config["server_url"]


def test_config_env_override():
    config = _get_config(env={"MODAL_SERVER_URL": "xyz.corp"})
    assert config["server_url"] == "xyz.corp"


def test_config_store_user(servicer, modal_config):
    with modal_config(show_on_error=True) as config_file_path:
        env = {"MODAL_SERVER_URL": servicer.client_addr}

        servicer.required_creds = {"abc": "xyz", "foo": "bar1", "foo2": "bar2", "ABC": "XYZ", "foo3": "bar3"}

        # No token by default
        config = _get_config(env=env)
        assert config["token_id"] is None

        # Set creds to abc / xyz
        _cli(["token", "set", "--token-id", "abc", "--token-secret", "xyz"], env=env)

        # Make sure an incorrect token fails
        with pytest.raises(CLIException):
            _cli(["token", "set", "--token-id", "abc", "--token-secret", "incorrect"], env=env)

        # Set creds to foo / bar1 for the prof_1 profile
        _cli(
            ["token", "set", "--token-id", "foo", "--token-secret", "bar1", "--profile", "prof_1", "--no-activate"],
            env=env,
        )

        # Set creds to foo2 / bar2 for the prof_2 profile (given as an env var)
        _cli(
            ["token", "set", "--token-id", "foo2", "--token-secret", "bar2", "--no-activate"],
            env={"MODAL_PROFILE": "prof_2", **env},
        )

        # Now these should be stored in the user's home directory
        config = _get_config(env=env)
        assert config["token_id"] == "abc"
        assert config["token_secret"] == "xyz"

        # Make sure it can be overridden too
        config = _get_config(env={"MODAL_TOKEN_ID": "foo", **env})
        assert config["token_id"] == "foo"
        assert config["token_secret"] == "xyz"

        # Check that the profile is named after the workspace username by default
        config = _get_config(env={"MODAL_PROFILE": "test-username", **env})
        assert config["token_id"] == "abc"
        assert config["token_secret"] == "xyz"

        # Check that we can get the prof_1 env creds too
        config = _get_config(env={"MODAL_PROFILE": "prof_1", **env})
        assert config["token_id"] == "foo"
        assert config["token_secret"] == "bar1"

        # Check that we can get the prof_2 env creds too
        config = _get_config(env={"MODAL_PROFILE": "prof_2", **env})
        assert config["token_id"] == "foo2"
        assert config["token_secret"] == "bar2"

        # Check that an empty string falls back to the active profile
        config = _get_config(env={"MODAL_PROFILE": "", **env})
        assert config["token_secret"] == "xyz"

        # Test that only the first profile was explicitly activated
        for profile, profile_config in toml.load(config_file_path).items():
            if profile == "test-username":
                assert profile_config["active"] is True
            else:
                assert "active" not in profile_config

        # Check that we can overwrite the default profile
        _cli(["token", "set", "--token-id", "ABC", "--token-secret", "XYZ"], env=env)
        assert toml.load(config_file_path)["test-username"]["token_id"] == "ABC"

        # Check that we activate a profile by default while setting a token
        _cli(
            ["token", "set", "--token-id", "foo3", "--token-secret", "bar3", "--profile", "prof_3"],
            env=env,
        )
        for profile, profile_config in toml.load(config_file_path).items():
            if profile == "prof_3":
                assert profile_config["active"] is True
            else:
                assert "active" not in profile_config


def test_config_env_override_arbitrary_env():
    """config.override_locally() replaces existing env var if not part of config."""
    key = "NVIDIA_VISIBLE_DEVICES"
    value = "0,1"

    # Place old value in memory.
    os.environ[key] = "none"

    # Expect value to be overwritten.
    config.override_locally(key, value)
    assert os.getenv(key) == value


@pytest.mark.asyncio
async def test_workspace_lookup(servicer, server_url_env):
    servicer.required_creds = {"ak-abc": "as-xyz"}
    resp = await synchronize_api(_lookup_workspace).aio(servicer.client_addr, "ak-abc", "as-xyz")
    assert resp.username == "test-username"


@pytest.mark.parametrize("arg", ["false", "'false'", "'False'", "'0'", 0, "''"])
def test_config_boolean(modal_config, arg):
    modal_toml = f"""
    [prof-1]
    token_id = 'ak-abc'
    token_secret = 'as_xyz'
    force_build = {arg}
    """
    with modal_config(modal_toml):
        assert not Config().get("force_build", "prof-1")


def test_malformed_config_better(modal_config):
    modal_toml = """
    token_id = 'ak-abc'
    token_secret = 'as_xyz'
    """
    with pytest.raises(InvalidError, match="must contain table sections"):
        with modal_config(modal_toml):
            pass

    modal_toml = """
    [default]
    token_id
    """
    with pytest.raises(InvalidError, match="Key name found without value"):
        with modal_config(modal_toml):
            pass
