import pathlib
import subprocess
import sys
import tempfile

import polyester


def _cli(args, env={}):
    lib_dir = pathlib.Path(polyester.__file__).parent.parent
    args = [sys.executable, "-m", "polyester.cli"] + args
    env = {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", **env}  # This is a dumb Python 3.6 issue
    ret = subprocess.run(args, cwd=lib_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        raise Exception(f"Failed with {ret.returncode} stdout: {ret.stdout} stderr: {ret.stderr}")
    return ret.stdout


def _get_config(env={}):
    stdout = _cli(["config", "show"], env=env)
    return eval(stdout)


def test_config():
    config = _get_config()
    assert config["server_url"]


def test_config_env_override():
    config = _get_config(env={"POLYESTER_SERVER_URL": "xyz.corp"})
    assert config["server_url"] == "xyz.corp"


def test_config_store_user():
    with tempfile.NamedTemporaryFile() as t:
        env = {"POLYESTER_CONFIG_PATH": t.name}

        # No token by default
        config = _get_config(env=env)
        assert config["token_id"] is None

        # Set creds to abc / xyz
        _cli(["token", "set", "abc", "xyz"], env=env)

        # Set creds to foo / bar1 for the prof_1 profile
        _cli(["token", "set", "foo", "bar1", "--env", "prof_1"], env=env)

        # Set creds to foo / bar2 for the prof_2 profile (given as an env var)
        _cli(["token", "set", "foo", "bar2"], env={"POLYESTER_ENV": "prof_2", **env})

        # Now these should be stored in the user's home directory
        config = _get_config(env=env)
        assert config["token_id"] == "abc"
        assert config["token_secret"] == "xyz"

        # Make sure it can be overridden too
        config = _get_config(env={"POLYESTER_TOKEN_ID": "foo", **env})
        assert config["token_id"] == "foo"
        assert config["token_secret"] == "xyz"

        # Check that we can get the prof_1 env creds too
        config = _get_config(env={"POLYESTER_ENV": "prof_1", **env})
        assert config["token_id"] == "foo"
        assert config["token_secret"] == "bar1"

        # Check that we can get the prof_1 env creds too
        config = _get_config(env={"POLYESTER_ENV": "prof_2", **env})
        assert config["token_id"] == "foo"
        assert config["token_secret"] == "bar2"
