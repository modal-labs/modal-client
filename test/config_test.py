import pathlib
import subprocess
import sys
import tempfile

import polyester


def _cli(args, env={}):
    lib_dir = pathlib.Path(polyester.__file__).parent.parent
    args = [sys.executable, "-m", "polyester.cli"] + args
    ret = subprocess.run(args, capture_output=True, cwd=lib_dir, env=env)
    assert ret.stderr == b""
    assert ret.returncode == 0
    return ret.stdout


def _get_config(env={}):
    stdout = _cli(["config-show"], env=env)
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
        _cli(["creds-set", "abc", "xyz"], env=env)

        # Now these should be stored in the user's home directory
        config = _get_config(env=env)
        assert config["token_id"] == "abc"
        assert config["token_secret"] == "xyz"
