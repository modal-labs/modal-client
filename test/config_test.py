# Copyright Modal Labs 2022
import os
import pathlib
import subprocess
import sys
import tempfile

import modal
from modal.config import config


def _cli(args, env={}):
    lib_dir = pathlib.Path(modal.__file__).parent.parent
    args = [sys.executable, "-m", "modal.cli.entry_point"] + args
    env = {
        **os.environ,
        **env,
        # For windows
        "PYTHONUTF8": "1",
    }
    ret = subprocess.run(args, cwd=lib_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = ret.stdout.decode()
    stderr = ret.stderr.decode()
    if ret.returncode != 0:
        raise Exception(f"Failed with {ret.returncode} stdout: {stdout} stderr: {stderr}")
    return stdout


def _get_config(env={}):
    stdout = _cli(["config", "show"], env=env)
    return eval(stdout)


def test_config():
    config = _get_config()
    assert config["server_url"]


def test_config_env_override():
    config = _get_config(env={"MODAL_SERVER_URL": "xyz.corp"})
    assert config["server_url"] == "xyz.corp"


def test_config_store_user(servicer):
    # Can't reopen a TemporaryFile on windows if it's already open.
    t = tempfile.NamedTemporaryFile(delete=False)
    t.close()

    env = {
        "MODAL_CONFIG_PATH": t.name,
        "MODAL_SERVER_URL": servicer.remote_addr,
    }

    # No token by default
    config = _get_config(env=env)
    assert config["token_id"] is None

    # Set creds to abc / xyz
    _cli(["token", "set", "--token-id", "abc", "--token-secret", "xyz"], env=env)

    # Set creds to foo / bar1 for the prof_1 profile
    _cli(["token", "set", "--token-id", "foo", "--token-secret", "bar1", "--profile", "prof_1"], env=env)

    # Set creds to foo / bar2 for the prof_2 profile (given as an env var)
    _cli(["token", "set", "--token-id", "foo", "--token-secret", "bar2"], env={"MODAL_PROFILE": "prof_2", **env})

    # Now these should be stored in the user's home directory
    config = _get_config(env=env)
    assert config["token_id"] == "abc"
    assert config["token_secret"] == "xyz"

    # Make sure it can be overridden too
    config = _get_config(env={"MODAL_TOKEN_ID": "foo", **env})
    assert config["token_id"] == "foo"
    assert config["token_secret"] == "xyz"

    # Check that we can get the prof_1 env creds too
    config = _get_config(env={"MODAL_PROFILE": "prof_1", **env})
    assert config["token_id"] == "foo"
    assert config["token_secret"] == "bar1"

    # Check that we can get the prof_1 env creds too
    config = _get_config(env={"MODAL_PROFILE": "prof_2", **env})
    assert config["token_id"] == "foo"
    assert config["token_secret"] == "bar2"

    os.remove(t.name)


def test_config_env_override_arbitrary_env():
    """config.override_locally() replaces existing env var if not part of config."""
    key = "NVIDIA_VISIBLE_DEVICES"
    value = "0,1"

    # Place old value in memory.
    os.environ[key] = "none"

    # Expect value to be overwritten.
    config.override_locally(key, value)
    assert os.getenv(key) == value
