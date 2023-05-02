# Copyright Modal Labs 2022
import os
import sys
import tempfile
import traceback
import unittest.mock
from contextlib import asynccontextmanager
from unittest import mock
from typing import List, Optional

import click
import click.testing
import pytest
import pytest_asyncio

from modal.cli.entry_point import entrypoint_cli
from modal import Client
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


def _run(args: List[str], expected_exit_code: int = 0, expected_stderr: Optional[str] = ""):
    runner = click.testing.CliRunner(mix_stderr=False)
    with mock.patch.object(sys, "argv", args):
        res = runner.invoke(entrypoint_cli, args)
    if res.exit_code != expected_exit_code:
        print("stdout:", repr(res.stdout))
        traceback.print_tb(res.exc_info[2])
        print(res.exception, file=sys.stderr)
        assert res.exit_code == expected_exit_code
    if expected_stderr is not None:
        assert res.stderr == expected_stderr
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


def test_secret_create(servicer, set_env_client):
    # fail without any keys
    _run(["secret", "create", "foo"], 2, None)

    _run(["secret", "create", "foo", "bar=baz"])
    assert len(servicer.secrets) == 1


def test_secret_list(servicer, set_env_client):
    res = _run(["secret", "list"])
    assert "dummy-secret-0" not in res.stdout

    _run(["secret", "create", "foo", "bar=baz"])
    _run(["secret", "create", "bar", "baz=buz"])

    res = _run(["secret", "list"])
    assert "dummy-secret-0" in res.stdout
    assert "dummy-secret-1" in res.stdout
    assert "dummy-secret-2" not in res.stdout


def test_app_token_new(servicer, set_env_client, server_url_env):
    with unittest.mock.patch("webbrowser.open_new_tab", lambda url: False):
        _run(["token", "new", "--env", "_test"])


def test_run(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["run", stub_file.as_posix()])
    _run(["run", stub_file.as_posix() + "::stub"])
    _run(["run", stub_file.as_posix() + "::stub.foo"])
    _run(["run", stub_file.as_posix() + "::foo"])
    _run(["run", stub_file.as_posix() + "::bar"], expected_exit_code=1, expected_stderr=None)
    file_with_entrypoint = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"
    _run(["run", file_with_entrypoint.as_posix()])
    _run(["run", file_with_entrypoint.as_posix() + "::main"])
    _run(["run", file_with_entrypoint.as_posix() + "::stub.main"])


def test_local_entrypoint_no_remote_calls(servicer, set_env_client, test_dir):
    file = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"
    res = _run(["run", file.as_posix()])
    assert "Warning: no remote function calls were made" not in res.stdout

    file = test_dir / "supports" / "app_run_tests" / "local_entrypoint_no_remote.py"
    res = _run(["run", file.as_posix()])
    assert "Warning: no remote function calls were made" in res.stdout


def test_help_message_unspecified_function(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "stub_with_multiple_functions.py"
    result = _run(["run", stub_file.as_posix()], expected_exit_code=2, expected_stderr=None)

    # should suggest available functions on the stub:
    assert "foo" in result.stderr
    assert "bar" in result.stderr

    result = _run(
        ["run", stub_file.as_posix(), "--help"], expected_exit_code=2, expected_stderr=None
    )  # TODO: help should not return non-zero
    # help should also available functions on the stub:
    assert "foo" in result.stderr
    assert "bar" in result.stderr


def test_run_states(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["run", stub_file.as_posix()])
    assert servicer.app_state_history["ap-1"] == [
        api_pb2.APP_STATE_INITIALIZING,
        api_pb2.APP_STATE_EPHEMERAL,
        api_pb2.APP_STATE_STOPPED,
    ]


def test_run_detach(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["run", "--detach", stub_file.as_posix()])
    assert servicer.app_state_history["ap-1"] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DETACHED]


def test_run_quiet(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    # Just tests that the command runs without error for now (tests end up defaulting to `show_progress=False` anyway,
    # without a TTY).
    _run(["run", "--quiet", stub_file.as_posix()])


def test_deploy(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    _run(["deploy", "--name=deployment_name", stub_file.as_posix()])
    assert servicer.app_state_history["ap-1"] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def test_run_custom_stub(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "custom_stub.py"
    res = _run(["run", stub_file.as_posix() + "::stub"], expected_exit_code=1, expected_stderr=None)
    assert "Could not find" in res.stderr
    res = _run(["run", stub_file.as_posix() + "::stub.foo"], expected_exit_code=1, expected_stderr=None)
    assert "Could not find" in res.stderr

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
    res = _run(["run", stub_file.as_posix()], expected_exit_code=2, expected_stderr=None)
    assert "You need to specify a Modal function or local entrypoint to run" in res.stderr

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


def test_app_descriptions(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "prints_desc_stub.py"
    _run(["run", "--detach", stub_file.as_posix() + "::stub.foo"])

    create_reqs = [s for s in servicer.requests if isinstance(s, api_pb2.AppCreateRequest)]
    assert len(create_reqs) == 1
    assert create_reqs[0].detach
    description = create_reqs[0].description
    assert "prints_desc_stub.py::stub.foo" in description
    assert "run --detach " not in description

    _run(["serve", "--timeout", "0.0", stub_file.as_posix()])
    create_reqs = [s for s in servicer.requests if isinstance(s, api_pb2.AppCreateRequest)]
    assert len(create_reqs) == 2
    description = create_reqs[1].description
    assert "prints_desc_stub.py" in description
    assert "serve" not in description
    assert "--timeout 0.0" not in description


def test_logs(servicer, server_url_env):
    servicer.done = True
    res = _run(["app", "logs", "ap-123"], expected_exit_code=0)
    assert res.stdout == "hello, world (1)\n"  # from servicer mock


def test_volume_get(set_env_client):
    volume_name = "my-shared-volume"
    _run(["volume", "create", volume_name])
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "w") as f:
            f.write("foo bar baz")
            f.flush()
        _run(["volume", "put", volume_name, upload_path, "test.txt"])

        _run(["volume", "get", volume_name, "test.txt", tmpdir])
        with open(os.path.join(tmpdir, "test.txt"), "r") as f:
            assert f.read() == "foo bar baz"
