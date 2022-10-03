import os
import traceback

import pytest_asyncio
import typer.testing

from modal import cli
from modal.client import Client

dummy_app_file = """
import modal

import other_module

stub = modal.Stub("my_app")

# Sanity check that the module is imported properly
import sys
mod = sys.modules[__name__]
assert mod.stub == stub
"""

dummy_other_module_file = "x = 42"


@pytest_asyncio.fixture
async def set_env_client(aio_client):
    try:
        Client.set_env_client(aio_client)
        yield
    finally:
        Client.set_env_client(None)


def _run(args, expected_exit_code=0):
    runner = typer.testing.CliRunner()
    res = runner.invoke(cli.entrypoint_cli, args)
    if res.exit_code != expected_exit_code:
        print(res.stdout, "Trace:")
        traceback.print_tb(res.exc_info[2])
        assert res.exit_code == expected_exit_code
    return res


def test_app_deploy_success(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        # Deploy as a script in cwd
        _run(["app", "deploy", "myapp.py"])

        # Deploy as a module
        _run(["app", "deploy", "myapp"])

        # Deploy as a script with an absolute path
        _run(["app", "deploy", os.path.abspath("myapp.py")])

    assert "my_app" in servicer.deployed_apps


def test_app_deploy_with_name(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        _run(["app", "deploy", "myapp.py", "--name", "my_app_foo"])

    assert "my_app_foo" in servicer.deployed_apps


dummy_aio_app_file = """
from modal.aio import AioStub

stub = AioStub("my_aio_app")
"""


def test_aio_app_deploy_success(servicer, mock_dir, set_env_client):
    with mock_dir({"myaioapp.py": dummy_aio_app_file}):
        _run(["app", "deploy", "myaioapp.py"])

    assert "my_aio_app" in servicer.deployed_apps


def test_app_deploy_no_such_module():
    res = _run(["app", "deploy", "does_not_exist.py"], 1)
    assert "No such file or directory" in res.stdout


def test_secret_list(servicer, set_env_client):
    res = _run(["secret", "list"])
    assert "dummy-secret-0" not in res.stdout
    servicer.created_secrets = 2

    res = _run(["secret", "list"])
    assert "dummy-secret-0" in res.stdout
    assert "dummy-secret-1" in res.stdout


def test_secret_create(servicer, set_env_client):
    # fail without any keys
    _run(["secret", "create", "foo"], 2)

    _run(["secret", "create", "foo", "bar=baz"])
    assert servicer.created_secrets == 1
