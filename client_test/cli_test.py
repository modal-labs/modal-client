# Copyright Modal Labs 2022
import os
import traceback

import click
import click.testing
import pytest_asyncio

from modal.cli.entry_point import entrypoint_cli
from modal.client import Client
from modal_proto import api_pb2

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
    runner = click.testing.CliRunner()
    res = runner.invoke(entrypoint_cli, args)
    if res.exit_code != expected_exit_code:
        print(res.stdout, "Trace:")
        traceback.print_tb(res.exc_info[2])
        assert res.exit_code == expected_exit_code
    return res


def test_app_deploy_success(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        # Deploy as a script in cwd
        _run(["deploy", "myapp.py"])

        # Deploy as a module
        _run(["deploy", "myapp"])

        # Deploy as a script with an absolute path
        _run(["deploy", os.path.abspath("myapp.py")])

    assert "my_app" in servicer.deployed_apps


def test_app_deploy_with_name(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        _run(["deploy", "myapp.py", "--name", "my_app_foo"])

    assert "my_app_foo" in servicer.deployed_apps


dummy_aio_app_file = """
from modal.aio import AioStub

stub = AioStub("my_aio_app")
"""


def test_aio_app_deploy_success(servicer, mock_dir, set_env_client):
    with mock_dir({"myaioapp.py": dummy_aio_app_file}):
        _run(["deploy", "myaioapp.py"])

    assert "my_aio_app" in servicer.deployed_apps


def test_app_deploy_no_such_module():
    res = _run(["deploy", "does_not_exist.py"], 1)
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


def test_app_token_new(servicer, server_url_env):
    _run(["token", "new", "--env", "_test"])


def test_run(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["run", stub_file.as_posix()])


def test_run_detach(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["run", "--detach", stub_file.as_posix()])
    assert servicer.app_state == {"ap-1": api_pb2.APP_STATE_DETACHED}


def test_deploy(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["deploy", "--name=deployment_name", stub_file.as_posix()])
    assert servicer.app_state == {"ap-1": api_pb2.APP_STATE_DEPLOYED}


def test_run_custom_stub(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "custom_stub.py"
    res = _run(["run", stub_file.as_posix(), "foo"], expected_exit_code=1)
    assert "stub variable" in res.stdout  # error output
    _run(["run", stub_file.as_posix() + "::my_stub.foo"])


def test_run_aiostub(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "async_stub.py"
    _run(["run", stub_file.as_posix()])
    assert len(servicer.client_calls) == 1


def test_run_local_entrypoint(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"

    res = _run(["run", stub_file.as_posix() + "::stub.main"])  # explicit name
    assert "called locally" in res.stdout
    assert len(servicer.client_calls) == 2

    res = _run(["run", stub_file.as_posix()])  # only one entry-point, no name needed
    assert "called locally" in res.stdout
    assert len(servicer.client_calls) == 4


def test_run_parse_args(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"
    res = _run(["run", stub_file.as_posix()], expected_exit_code=1)
    assert "You need to specify an entrypoint" in res.stdout

    valid_call_args = [
        (
            [
                "run",
                f"{stub_file.as_posix()}::stub.dt_arg",
                "--dt",
                "2022-10-31",
            ],
            "the day is 31",
        ),
        (["run", f"{stub_file.as_posix()}::.dt_arg", "--dt=2022-10-31"], "the day is 31"),
        (["run", f"{stub_file.as_posix()}::.int_arg", "--i=200"], "200"),
        (["run", f"{stub_file.as_posix()}::.default_arg"], "10"),
        (["run", f"{stub_file.as_posix()}::.unannotated_arg", "--i=2022-10-31"], "'2022-10-31'"),
    ]
    for args, expected in valid_call_args:
        res = _run(args)
        assert expected in res.stdout
        assert len(servicer.client_calls) == 0
