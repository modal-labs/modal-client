# Copyright Modal Labs 2022
import os
import sys
import traceback
import unittest.mock
from contextlib import asynccontextmanager
from unittest import mock

import click
import click.testing
import pytest
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
        print("stdout:", repr(res.stdout))
        traceback.print_tb(res.exc_info[2])
        print(res.exception, file=sys.stderr)
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
    assert "No such file or directory" in str(res.exception)
    res = _run(["deploy", "does.not.exist"], 1)
    assert "No module named 'does'" in str(res.exception)


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


def test_app_token_new(servicer, set_env_client, server_url_env):
    with unittest.mock.patch("webbrowser.open_new_tab", lambda url: False):
        _run(["token", "new", "--env", "_test"])


def test_run(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["run", stub_file.as_posix()])
    _run(["run", stub_file.as_posix() + "::stub"])
    _run(["run", stub_file.as_posix() + "::stub.foo"])
    _run(["run", stub_file.as_posix() + "::foo"])
    _run(["run", stub_file.as_posix() + "::bar"], expected_exit_code=1)
    file_with_entrypoint = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"
    _run(["run", file_with_entrypoint.as_posix()])
    _run(["run", file_with_entrypoint.as_posix() + "::main"])
    _run(["run", file_with_entrypoint.as_posix() + "::stub.main"])


def test_help_message_unspecified_function(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "stub_with_multiple_functions.py"
    result = _run(["run", stub_file.as_posix()], expected_exit_code=2)

    # should suggest available functions on the stub:
    assert "foo" in result.stdout
    assert "bar" in result.stdout

    result = _run(
        ["run", stub_file.as_posix(), "--help"], expected_exit_code=2
    )  # TODO: help should not return non-zero
    # help should also available functions on the stub:
    assert "foo" in result.stdout
    assert "bar" in result.stdout


def test_run_detach(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["run", "--detach", stub_file.as_posix()])
    assert servicer.app_state == {"ap-1": api_pb2.APP_STATE_DETACHED}


def test_deploy(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["deploy", "--name=deployment_name", stub_file.as_posix()])
    assert servicer.app_state == {"ap-1": api_pb2.APP_STATE_DEPLOYED}


def test_run_custom_stub(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "custom_stub.py"
    res = _run(["run", stub_file.as_posix() + "::stub"], expected_exit_code=1)
    assert "Could not find" in res.stdout
    res = _run(["run", stub_file.as_posix() + "::stub.foo"], expected_exit_code=1)
    assert "Could not find" in res.stdout

    _run(["run", stub_file.as_posix() + "::my_stub.foo"])
    _run(["run", stub_file.as_posix() + "::foo"])


def test_run_aiostub(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "async_stub.py"
    _run(["run", stub_file.as_posix()])
    assert len(servicer.client_calls) == 1


def test_run_local_entrypoint(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"

    res = _run(["run", stub_file.as_posix() + "::stub.main"])  # explicit name
    assert "called locally" in res.stdout
    assert len(servicer.client_calls) == 2

    res = _run(["run", stub_file.as_posix()])  # only one entry-point, no name needed
    assert "called locally" in res.stdout
    assert len(servicer.client_calls) == 4


def test_run_parse_args(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"
    res = _run(["run", stub_file.as_posix()], expected_exit_code=2)
    assert "You need to specify a Modal function or local entrypoint to run" in res.stdout

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
        (["run", f"{stub_file.as_posix()}::dt_arg", "--dt=2022-10-31"], "the day is 31"),
        (["run", f"{stub_file.as_posix()}::int_arg", "--i=200"], "200"),
        (["run", f"{stub_file.as_posix()}::default_arg"], "10"),
        (["run", f"{stub_file.as_posix()}::unannotated_arg", "--i=2022-10-31"], "'2022-10-31'"),
        # TODO: fix class references
        # (["run", f"{stub_file.as_posix()}::ALifecycle.some_method", "--i=hello"], "'hello'"),
    ]
    for args, expected in valid_call_args:
        res = _run(args)
        assert expected in res.stdout
        assert len(servicer.client_calls) == 0


@pytest.fixture
def fresh_main_thread_assertion_module(test_dir):
    modules_to_unload = [n for n in sys.modules.keys() if "main_thread_assertion" in n]
    assert len(modules_to_unload) <= 1
    for mod in modules_to_unload:
        sys.modules.pop(mod)
    yield test_dir / "supports" / "app_run_tests" / "main_thread_assertion.py"


def test_no_user_code_in_synchronicity_run(servicer, set_env_client, test_dir, fresh_main_thread_assertion_module):
    pytest._did_load_main_thread_assertion = False
    _run(["run", fresh_main_thread_assertion_module.as_posix()])
    assert pytest._did_load_main_thread_assertion
    print()


def test_no_user_code_in_synchronicity_deploy(servicer, set_env_client, test_dir, fresh_main_thread_assertion_module):
    pytest._did_load_main_thread_assertion = False
    _run(["deploy", "--name", "foo", fresh_main_thread_assertion_module.as_posix()])
    assert pytest._did_load_main_thread_assertion


def test_serve(servicer, set_env_client, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "webhook.py"
    _run(["serve", stub_file.as_posix(), "--timeout", "3"], expected_exit_code=0)


def test_shell(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"

    def mock_get_pty_info() -> api_pb2.PTYInfo:
        rows, cols = (64, 128)
        return api_pb2.PTYInfo(
            enabled=True,
            winsz_rows=rows,
            winsz_cols=cols,
            env_term=os.environ.get("TERM"),
            env_colorterm=os.environ.get("COLORTERM"),
            env_term_program=os.environ.get("TERM_PROGRAM"),
        )

    @asynccontextmanager
    async def noop_async_context_manager(*args, **kwargs):
        yield

    ran_cmd = None

    @servicer.function_body
    def dummy_exec(cmd: str):
        nonlocal ran_cmd
        ran_cmd = cmd

    with mock.patch("rich.console.Console.is_terminal", True), mock.patch(
        "modal._pty.get_pty_info", mock_get_pty_info
    ), mock.patch("modal._pty.write_stdin_to_pty_stream", noop_async_context_manager):
        _run(["shell", stub_file.as_posix() + "::foo"])
    assert ran_cmd == "/bin/bash"
