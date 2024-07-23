# Copyright Modal Labs 2022-2023
import asyncio
import contextlib
import json
import os
import platform
import pytest
import re
import subprocess
import sys
import tempfile
import traceback
from pickle import dumps
from typing import List
from unittest import mock

import click
import click.testing
import toml

from modal.cli.entry_point import entrypoint_cli
from modal.exception import InvalidError
from modal_proto import api_pb2

from .supports.skip import skip_windows

dummy_app_file = """
import modal

import other_module

app = modal.App("my_app")

# Sanity check that the module is imported properly
import sys
mod = sys.modules[__name__]
assert mod.app == app
"""

dummy_other_module_file = "x = 42"


def _run(args: List[str], expected_exit_code: int = 0, expected_stderr: str = "", expected_error: str = ""):
    runner = click.testing.CliRunner(mix_stderr=False)
    with mock.patch.object(sys, "argv", args):
        res = runner.invoke(entrypoint_cli, args)
    if res.exit_code != expected_exit_code:
        print("stdout:", repr(res.stdout))
        print("stderr:", repr(res.stderr))
        traceback.print_tb(res.exc_info[2])
        print(res.exception, file=sys.stderr)
        assert res.exit_code == expected_exit_code
    if expected_stderr:
        assert re.search(expected_stderr, res.stderr), "stderr does not match expected string"
    if expected_error:
        assert re.search(expected_error, str(res.exception)), "exception message does not match expected string"
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


def test_secret_create(servicer, set_env_client):
    # fail without any keys
    _run(["secret", "create", "foo"], 2, None)

    _run(["secret", "create", "foo", "bar=baz"])
    assert len(servicer.secrets) == 1

    # Creating the same one again should fail
    _run(["secret", "create", "foo", "bar=baz"], expected_exit_code=1)

    # But it should succeed with --force
    _run(["secret", "create", "foo", "bar=baz", "--force"])


def test_secret_list(servicer, set_env_client):
    res = _run(["secret", "list"])
    assert "dummy-secret-0" not in res.stdout

    _run(["secret", "create", "foo", "bar=baz"])
    _run(["secret", "create", "bar", "baz=buz"])
    _run(["secret", "create", "eric", "baz=bu 123z=b\n\t\r #(Q)JO5️⃣5️⃣😤WMLE🔧:GWam "])

    res = _run(["secret", "list"])
    assert "dummy-secret-0" in res.stdout
    assert "dummy-secret-1" in res.stdout
    assert "dummy-secret-2" in res.stdout
    assert "dummy-secret-3" not in res.stdout


def test_app_token_new(servicer, set_env_client, server_url_env, modal_config):
    with modal_config() as config_file_path:
        _run(["token", "new", "--profile", "_test"])
        assert "_test" in toml.load(config_file_path)


def test_app_setup(servicer, set_env_client, server_url_env, modal_config):
    with modal_config() as config_file_path:
        _run(["setup", "--profile", "_test"])
        assert "_test" in toml.load(config_file_path)


def test_run(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    _run(["run", app_file.as_posix()])
    _run(["run", app_file.as_posix() + "::app"])
    _run(["run", app_file.as_posix() + "::foo"])
    _run(["run", app_file.as_posix() + "::bar"], expected_exit_code=1, expected_stderr=None)
    file_with_entrypoint = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"
    _run(["run", file_with_entrypoint.as_posix()])
    _run(["run", file_with_entrypoint.as_posix() + "::main"])
    _run(["run", file_with_entrypoint.as_posix() + "::app.main"])


def test_run_stub(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "app_was_once_stub.py"
    with pytest.warns(match="App"):
        _run(["run", app_file.as_posix()])
    with pytest.warns(match="App"):
        _run(["run", app_file.as_posix() + "::foo"])


def test_run_stub_2(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "app_was_once_stub_2.py"
    with pytest.warns(match="`app`"):
        _run(["run", app_file.as_posix()])
    _run(["run", app_file.as_posix() + "::stub"])
    _run(["run", app_file.as_posix() + "::foo"])


def test_run_stub_with_app(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "app_and_stub.py"
    with pytest.warns(match="`app`"):
        _run(["run", app_file.as_posix()])
    _run(["run", app_file.as_posix() + "::stub"])
    _run(["run", app_file.as_posix() + "::foo"])


def test_run_async(servicer, set_env_client, test_dir):
    sync_fn = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"
    res = _run(["run", sync_fn.as_posix()])
    assert "called locally" in res.stdout

    async_fn = test_dir / "supports" / "app_run_tests" / "local_entrypoint_async.py"
    res = _run(["run", async_fn.as_posix()])
    assert "called locally (async)" in res.stdout


def test_run_generator(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "generator.py"
    result = _run(["run", app_file.as_posix()], expected_exit_code=1)
    assert "generator functions" in str(result.exception)


def test_help_message_unspecified_function(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "app_with_multiple_functions.py"
    result = _run(["run", app_file.as_posix()], expected_exit_code=2, expected_stderr=None)

    # should suggest available functions on the app:
    assert "foo" in result.stderr
    assert "bar" in result.stderr

    result = _run(
        ["run", app_file.as_posix(), "--help"], expected_exit_code=2, expected_stderr=None
    )  # TODO: help should not return non-zero
    # help should also available functions on the app:
    assert "foo" in result.stderr
    assert "bar" in result.stderr


def test_run_states(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    _run(["run", app_file.as_posix()])
    assert servicer.app_state_history["ap-1"] == [
        api_pb2.APP_STATE_INITIALIZING,
        api_pb2.APP_STATE_EPHEMERAL,
        api_pb2.APP_STATE_STOPPED,
    ]


def test_run_detach(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    _run(["run", "--detach", app_file.as_posix()])
    assert servicer.app_state_history["ap-1"] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DETACHED]


def test_run_quiet(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    # Just tests that the command runs without error for now (tests end up defaulting to `show_progress=False` anyway,
    # without a TTY).
    _run(["run", "--quiet", app_file.as_posix()])


def test_deploy(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    _run(["deploy", "--name=deployment_name", app_file.as_posix()])
    assert servicer.app_state_history["ap-1"] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]


def test_run_custom_app(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "custom_app.py"
    res = _run(["run", app_file.as_posix() + "::app"], expected_exit_code=1, expected_stderr=None)
    assert "Could not find" in res.stderr
    res = _run(["run", app_file.as_posix() + "::app.foo"], expected_exit_code=1, expected_stderr=None)
    assert "Could not find" in res.stderr

    _run(["run", app_file.as_posix() + "::foo"])


def test_run_aiofunc(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "async_app.py"
    _run(["run", app_file.as_posix()])
    assert len(servicer.client_calls) == 1


def test_run_local_entrypoint(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"

    res = _run(["run", app_file.as_posix() + "::app.main"])  # explicit name
    assert "called locally" in res.stdout
    assert len(servicer.client_calls) == 2

    res = _run(["run", app_file.as_posix()])  # only one entry-point, no name needed
    assert "called locally" in res.stdout
    assert len(servicer.client_calls) == 4


def test_run_local_entrypoint_invalid_with_app_run(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "local_entrypoint_invalid.py"

    res = _run(["run", app_file.as_posix()], expected_exit_code=1)
    assert "app is already running" in str(res.exception.__cause__).lower()
    assert "unreachable" not in res.stdout
    assert len(servicer.client_calls) == 0


def test_run_parse_args_entrypoint(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"
    res = _run(["run", app_file.as_posix()], expected_exit_code=2, expected_stderr=None)
    assert "You need to specify a Modal function or local entrypoint to run" in res.stderr

    valid_call_args = [
        (
            [
                "run",
                f"{app_file.as_posix()}::app.dt_arg",
                "--dt",
                "2022-10-31",
            ],
            "the day is 31",
        ),
        (["run", f"{app_file.as_posix()}::dt_arg", "--dt=2022-10-31"], "the day is 31"),
        (["run", f"{app_file.as_posix()}::int_arg", "--i=200"], "200 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::default_arg"], "10 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::unannotated_arg", "--i=2022-10-31"], "'2022-10-31' <class 'str'>"),
        (["run", f"{app_file.as_posix()}::unannotated_default_arg"], "10 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::optional_arg", "--i=20"], "20 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::optional_arg"], "None <class 'NoneType'>"),
        (["run", f"{app_file.as_posix()}::optional_arg_postponed"], "None <class 'NoneType'>"),
    ]
    if sys.version_info >= (3, 10):
        valid_call_args.extend(
            [
                (["run", f"{app_file.as_posix()}::optional_arg_pep604", "--i=20"], "20 <class 'int'>"),
                (["run", f"{app_file.as_posix()}::optional_arg_pep604"], "None <class 'NoneType'>"),
            ]
        )
    for args, expected in valid_call_args:
        res = _run(args)
        assert expected in res.stdout
        assert len(servicer.client_calls) == 0

    if sys.version_info >= (3, 10):
        res = _run(["run", f"{app_file.as_posix()}::unparseable_annot", "--i=20"], expected_exit_code=1)
        assert "Parameter `i` has unparseable annotation: typing.Union[int, str]" in str(res.exception)

    if sys.version_info <= (3, 10):
        res = _run(["run", f"{app_file.as_posix()}::optional_arg_pep604"], expected_exit_code=1)
        assert "Unable to generate command line interface for app entrypoint." in str(res.exception)


def test_run_parse_args_function(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"
    res = _run(["run", app_file.as_posix()], expected_exit_code=2, expected_stderr=None)
    assert "You need to specify a Modal function or local entrypoint to run" in res.stderr

    # HACK: all the tests use the same arg, i.
    @servicer.function_body
    def print_type(i):
        print(repr(i), type(i))

    valid_call_args = [
        (["run", f"{app_file.as_posix()}::int_arg_fn", "--i=200"], "200 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::ALifecycle.some_method", "--i=hello"], "'hello' <class 'str'>"),
        (["run", f"{app_file.as_posix()}::ALifecycle.some_method_int", "--i=42"], "42 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::optional_arg_fn"], "None <class 'NoneType'>"),
    ]
    for args, expected in valid_call_args:
        res = _run(args)
        assert expected in res.stdout


def test_run_user_script_exception(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "raises_error.py"
    res = _run(["run", app_file.as_posix()], expected_exit_code=1)
    assert res.exc_info[1].user_source == str(app_file.resolve())


@pytest.fixture
def fresh_main_thread_assertion_module(test_dir):
    modules_to_unload = [n for n in sys.modules.keys() if "main_thread_assertion" in n]
    assert len(modules_to_unload) <= 1
    for mod in modules_to_unload:
        sys.modules.pop(mod)
    yield test_dir / "supports" / "app_run_tests" / "main_thread_assertion.py"


def test_no_user_code_in_synchronicity_run(servicer, set_env_client, test_dir, fresh_main_thread_assertion_module):
    pytest._did_load_main_thread_assertion = False  # type: ignore
    _run(["run", fresh_main_thread_assertion_module.as_posix()])
    assert pytest._did_load_main_thread_assertion  # type: ignore
    print()


def test_no_user_code_in_synchronicity_deploy(servicer, set_env_client, test_dir, fresh_main_thread_assertion_module):
    pytest._did_load_main_thread_assertion = False  # type: ignore
    _run(["deploy", "--name", "foo", fresh_main_thread_assertion_module.as_posix()])
    assert pytest._did_load_main_thread_assertion  # type: ignore


def test_serve(servicer, set_env_client, server_url_env, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "webhook.py"
    _run(["serve", app_file.as_posix(), "--timeout", "3"], expected_exit_code=0)


@pytest.fixture
def mock_shell_pty():
    def mock_get_pty_info(shell: bool) -> api_pb2.PTYInfo:
        rows, cols = (64, 128)
        return api_pb2.PTYInfo(
            enabled=True,
            winsz_rows=rows,
            winsz_cols=cols,
            env_term=os.environ.get("TERM"),
            env_colorterm=os.environ.get("COLORTERM"),
            env_term_program=os.environ.get("TERM_PROGRAM"),
            pty_type=api_pb2.PTYInfo.PTY_TYPE_SHELL,
        )

    captured_out = []
    fake_stdin = [b"echo foo\n", b"exit\n"]

    async def write_to_fd(fd: int, data: bytes):
        nonlocal captured_out
        captured_out.append((fd, data))

    @contextlib.asynccontextmanager
    async def fake_stream_from_stdin(handle_input, use_raw_terminal=False):
        async def _write():
            message_index = 0
            while True:
                if message_index == len(fake_stdin):
                    break
                data = fake_stdin[message_index]
                await handle_input(data, message_index)
                message_index += 1

        write_task = asyncio.create_task(_write())
        yield
        write_task.cancel()

    with mock.patch("rich.console.Console.is_terminal", True), mock.patch(
        "modal._pty.get_pty_info", mock_get_pty_info
    ), mock.patch("modal.runner.get_pty_info", mock_get_pty_info), mock.patch(
        "modal._utils.shell_utils.stream_from_stdin", fake_stream_from_stdin
    ), mock.patch("modal._sandbox_shell.write_to_fd", write_to_fd):
        yield fake_stdin, captured_out


@skip_windows("modal shell is not supported on Windows.")
def test_shell(servicer, set_env_client, test_dir, mock_shell_pty):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    webhook_app_file = test_dir / "supports" / "app_run_tests" / "webhook.py"
    cls_app_file = test_dir / "supports" / "app_run_tests" / "cls.py"
    fake_stdin, captured_out = mock_shell_pty

    fake_stdin.clear()
    fake_stdin.extend([b'echo "Hello World"\n', b"exit\n"])

    # Function is explicitly specified
    _run(["shell", app_file.as_posix() + "::foo"])

    shell_prompt = servicer.sandbox_shell_prompt.encode("utf-8")

    # first captured message is the empty message the mock server sends
    assert captured_out == [(1, shell_prompt), (1, b"Hello World\n")]
    captured_out.clear()

    # Function is explicitly specified
    _run(["shell", webhook_app_file.as_posix() + "::foo"])
    assert captured_out == [(1, shell_prompt), (1, b"Hello World\n")]
    captured_out.clear()

    # Function must be inferred
    _run(["shell", webhook_app_file.as_posix()])
    assert captured_out == [(1, shell_prompt), (1, b"Hello World\n")]
    captured_out.clear()

    _run(["shell", cls_app_file.as_posix()])
    assert captured_out == [(1, shell_prompt), (1, b"Hello World\n")]
    captured_out.clear()


@skip_windows("modal shell is not supported on Windows.")
def test_shell_cmd(servicer, set_env_client, test_dir, mock_shell_pty):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    _, captured_out = mock_shell_pty
    _run(["shell", "--cmd", "pwd", app_file.as_posix() + "::foo"])
    expected_output = subprocess.run(["pwd"], capture_output=True, check=True).stdout
    shell_prompt = servicer.sandbox_shell_prompt.encode("utf-8")
    assert captured_out == [(1, shell_prompt), (1, expected_output)]


def test_shell_unsuported_cmds_fails_on_windows(servicer, set_env_client, mock_shell_pty):
    expected_exit_code = 1 if platform.system() == "Windows" else 0
    res = _run(["shell"], expected_exit_code=expected_exit_code)

    if expected_exit_code != 0:
        assert re.search("Windows", str(res.exception)), "exception message does not match expected string"


def test_app_descriptions(servicer, server_url_env, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "prints_desc_app.py"
    _run(["run", "--detach", app_file.as_posix() + "::foo"])

    create_reqs = [s for s in servicer.requests if isinstance(s, api_pb2.AppCreateRequest)]
    assert len(create_reqs) == 1
    assert create_reqs[0].app_state == api_pb2.APP_STATE_DETACHED
    description = create_reqs[0].description
    assert "prints_desc_app.py::foo" in description
    assert "run --detach " not in description

    _run(["serve", "--timeout", "0.0", app_file.as_posix()])
    create_reqs = [s for s in servicer.requests if isinstance(s, api_pb2.AppCreateRequest)]
    assert len(create_reqs) == 2
    description = create_reqs[1].description
    assert "prints_desc_app.py" in description
    assert "serve" not in description
    assert "--timeout 0.0" not in description


def test_logs(servicer, server_url_env, set_env_client, mock_dir):
    async def app_done(self, stream):
        await stream.recv_message()
        log = api_pb2.TaskLogs(data="hello\n", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT)
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="1", items=[log]))
        await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("AppGetLogs", app_done)

        res = _run(["app", "logs", "ap-123"])
        assert res.stdout == "hello\n"

        with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
            res = _run(["deploy", "myapp.py", "--name", "my-app", "--stream-logs"])
            assert res.stdout.endswith("hello\n")

    _run(
        ["app", "logs", "app-123", "-n", "my-app"],
        expected_exit_code=2,
        expected_stderr="Must pass either an ID or a name",
    )

    _run(
        ["app", "logs", "-n", "does-not-exist"],
        expected_exit_code=1,
        expected_error="Could not find a deployed app named 'does-not-exist'",
    )


def test_app_stop(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        # Deploy as a module
        _run(["deploy", "myapp"])

    res = _run(["app", "list"])
    assert re.search("my_app .+ deployed", res.stdout)

    _run(["app", "stop", "-n", "my_app"])

    # Note that the mock servicer doesn't report "stopped" app statuses
    # so we just check that it's not reported as deployed
    res = _run(["app", "list"])
    assert not re.search("my_app .+ deployed", res.stdout)


def test_nfs_get(set_env_client, servicer):
    nfs_name = "my-shared-nfs"
    _run(["nfs", "create", nfs_name])
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "w") as f:
            f.write("foo bar baz")
            f.flush()
        _run(["nfs", "put", nfs_name, upload_path, "test.txt"])

        _run(["nfs", "get", nfs_name, "test.txt", tmpdir])
        with open(os.path.join(tmpdir, "test.txt"), "r") as f:
            assert f.read() == "foo bar baz"


def test_volume_cli(set_env_client):
    _run(["volume", "--help"])


def test_volume_get(servicer, set_env_client):
    vol_name = "my-test-vol"
    _run(["volume", "create", vol_name])
    file_path = "test.txt"
    file_contents = b"foo bar baz"
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "wb") as f:
            f.write(file_contents)
            f.flush()
        _run(["volume", "put", vol_name, upload_path, file_path])

        _run(["volume", "get", vol_name, file_path, tmpdir])
        with open(os.path.join(tmpdir, file_path), "rb") as f:
            assert f.read() == file_contents

        download_path = os.path.join(tmpdir, "download.txt")
        _run(["volume", "get", vol_name, file_path, download_path])
        with open(download_path, "rb") as f:
            assert f.read() == file_contents

    with tempfile.TemporaryDirectory() as tmpdir2:
        _run(["volume", "get", vol_name, "/", tmpdir2])
        with open(os.path.join(tmpdir2, file_path), "rb") as f:
            assert f.read() == file_contents


def test_volume_put_force(servicer, set_env_client):
    vol_name = "my-test-vol"
    _run(["volume", "create", vol_name])
    file_path = "test.txt"
    file_contents = b"foo bar baz"
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "wb") as f:
            f.write(file_contents)
            f.flush()

        # File upload
        _run(["volume", "put", vol_name, upload_path, file_path])  # Seed the volume
        with servicer.intercept() as ctx:
            _run(["volume", "put", vol_name, upload_path, file_path], expected_exit_code=2, expected_stderr=None)
            assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            _run(["volume", "put", vol_name, upload_path, file_path, "--force"])
            assert not ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

        # Dir upload
        _run(["volume", "put", vol_name, tmpdir])  # Seed the volume
        with servicer.intercept() as ctx:
            _run(["volume", "put", vol_name, tmpdir], expected_exit_code=2, expected_stderr=None)
            assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            _run(["volume", "put", vol_name, tmpdir, "--force"])
            assert not ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files


def test_volume_rm(servicer, set_env_client):
    vol_name = "my-test-vol"
    _run(["volume", "create", vol_name])
    file_path = "test.txt"
    file_contents = b"foo bar baz"
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "wb") as f:
            f.write(file_contents)
            f.flush()
        _run(["volume", "put", vol_name, upload_path, file_path])

        _run(["volume", "get", vol_name, file_path, tmpdir])
        with open(os.path.join(tmpdir, file_path), "rb") as f:
            assert f.read() == file_contents

        _run(["volume", "rm", vol_name, file_path])
        _run(["volume", "get", vol_name, file_path], expected_exit_code=1, expected_stderr=None)


def test_volume_ls(servicer, set_env_client):
    vol_name = "my-test-vol"
    _run(["volume", "create", vol_name])

    fnames = ["a", "b", "c"]
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in fnames:
            src_path = os.path.join(tmpdir, f"{fname}.txt")
            with open(src_path, "w") as f:
                f.write(fname * 5)
            _run(["volume", "put", vol_name, src_path, f"data/{fname}.txt"])

    res = _run(["volume", "ls", vol_name])
    assert "data" in res.stdout

    res = _run(["volume", "ls", vol_name, "data"])
    for fname in fnames:
        assert f"{fname}.txt" in res.stdout

    res = _run(["volume", "ls", vol_name, "data", "--json"])
    res_dict = json.loads(res.stdout)
    assert len(res_dict) == len(fnames)
    for entry, fname in zip(res_dict, fnames):
        assert entry["Filename"] == f"data/{fname}.txt"
        assert entry["Type"] == "file"


def test_volume_create_delete(servicer, server_url_env, set_env_client):
    vol_name = "test-delete-vol"
    _run(["volume", "create", vol_name])
    assert vol_name in _run(["volume", "list"]).stdout
    _run(["volume", "delete", "--yes", vol_name])
    assert vol_name not in _run(["volume", "list"]).stdout


@pytest.mark.parametrize("command", [["run"], ["deploy"], ["serve", "--timeout=1"], ["shell"]])
@pytest.mark.usefixtures("set_env_client", "mock_shell_pty")
@skip_windows("modal shell is not supported on Windows.")
def test_environment_flag(test_dir, servicer, command):
    @servicer.function_body
    def nothing(
        arg=None,
    ):  # hacky - compatible with both argless modal run and interactive mode which always sends an arg...
        pass

    app_file = test_dir / "supports" / "app_run_tests" / "app_with_lookups.py"
    with servicer.intercept() as ctx:
        ctx.add_response(
            "MountGetOrCreate",
            api_pb2.MountGetOrCreateResponse(
                mount_id="mo-123",
                handle_metadata=api_pb2.MountHandleMetadata(content_checksum_sha256_hex="abc123"),
            ),
            request_filter=lambda req: req.deployment_name.startswith("modal-client-mount")
            and req.namespace == api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )  # built-in client lookup
        ctx.add_response(
            "SharedVolumeGetOrCreate",
            api_pb2.SharedVolumeGetOrCreateResponse(shared_volume_id="sv-123"),
            request_filter=lambda req: req.deployment_name == "volume_app" and req.environment_name == "staging",
        )
        _run(command + ["--env=staging", str(app_file)])

    app_create: api_pb2.AppCreateRequest = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "staging"


@pytest.mark.parametrize("command", [["run"], ["deploy"], ["serve", "--timeout=1"], ["shell"]])
@pytest.mark.usefixtures("set_env_client", "mock_shell_pty")
@skip_windows("modal shell is not supported on Windows.")
def test_environment_noflag(test_dir, servicer, command, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "some_weird_default_env")

    @servicer.function_body
    def nothing(
        arg=None,
    ):  # hacky - compatible with both argless modal run and interactive mode which always sends an arg...
        pass

    app_file = test_dir / "supports" / "app_run_tests" / "app_with_lookups.py"
    with servicer.intercept() as ctx:
        ctx.add_response(
            "MountGetOrCreate",
            api_pb2.MountGetOrCreateResponse(
                mount_id="mo-123",
                handle_metadata=api_pb2.MountHandleMetadata(content_checksum_sha256_hex="abc123"),
            ),
            request_filter=lambda req: req.deployment_name.startswith("modal-client-mount")
            and req.namespace == api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )  # built-in client lookup
        ctx.add_response(
            "SharedVolumeGetOrCreate",
            api_pb2.SharedVolumeGetOrCreateResponse(shared_volume_id="sv-123"),
            request_filter=lambda req: req.deployment_name == "volume_app"
            and req.environment_name == "some_weird_default_env",
        )
        _run(command + [str(app_file)])

    app_create: api_pb2.AppCreateRequest = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "some_weird_default_env"


def test_cls(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "cls.py"

    _run(["run", app_file.as_posix(), "--x", "42", "--y", "1000"])
    _run(["run", f"{app_file.as_posix()}::AParametrized.some_method", "--x", "42", "--y", "1000"])


def test_profile_list(servicer, server_url_env, modal_config):
    config = """
    [test-profile]
    token_id = "ak-abc"
    token_secret = "as-xyz"

    [other-profile]
    token_id = "ak-123"
    token_secret = "as-789"
    active = true
    """

    with modal_config(config):
        res = _run(["profile", "list"])
        table_rows = res.stdout.split("\n")
        assert re.search("Profile .+ Workspace", table_rows[1])
        assert re.search("test-profile .+ test-username", table_rows[3])
        assert re.search("other-profile .+ test-username", table_rows[4])

        res = _run(["profile", "list", "--json"])
        json_data = json.loads(res.stdout)
        assert json_data[0]["name"] == "test-profile"
        assert json_data[0]["workspace"] == "test-username"
        assert json_data[1]["name"] == "other-profile"
        assert json_data[1]["workspace"] == "test-username"

        orig_env_token_id = os.environ.get("MODAL_TOKEN_ID")
        orig_env_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        os.environ["MODAL_TOKEN_ID"] = "ak-abc"
        os.environ["MODAL_TOKEN_SECRET"] = "as-xyz"
        try:
            res = _run(["profile", "list"])
            assert "Using test-username workspace based on environment variables" in res.stdout
        finally:
            if orig_env_token_id:
                os.environ["MODAL_TOKEN_ID"] = orig_env_token_id
            else:
                del os.environ["MODAL_TOKEN_ID"]
            if orig_env_token_secret:
                os.environ["MODAL_TOKEN_SECRET"] = orig_env_token_secret
            else:
                del os.environ["MODAL_TOKEN_SECRET"]


def test_list_apps(servicer, mock_dir, set_env_client):
    res = _run(["app", "list"])
    assert "my_app_foo" not in res.stdout

    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        _run(["deploy", "myapp.py", "--name", "my_app_foo"])

    res = _run(["app", "list"])
    assert "my_app_foo" in res.stdout

    res = _run(["app", "list", "--json"])
    assert json.loads(res.stdout)

    _run(["volume", "create", "my-vol"])
    res = _run(["app", "list"])
    assert "my-vol" not in res.stdout


def test_dict_create_list_delete(servicer, server_url_env, set_env_client):
    _run(["dict", "create", "foo-dict"])
    _run(["dict", "create", "bar-dict"])
    res = _run(["dict", "list"])
    assert "foo-dict" in res.stdout
    assert "bar-dict" in res.stdout

    _run(["dict", "delete", "bar-dict", "--yes"])
    res = _run(["dict", "list"])
    assert "foo-dict" in res.stdout
    assert "bar-dict" not in res.stdout


def test_dict_show_get_clear(servicer, server_url_env, set_env_client):
    # Kind of hacky to be modifying the attributes on the servicer like this
    key = ("baz-dict", api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, os.environ.get("MODAL_ENVIRONMENT", "main"))
    dict_id = "di-abc123"
    servicer.deployed_dicts[key] = dict_id
    servicer.dicts[dict_id] = {dumps("a"): dumps(123), dumps("b"): dumps("blah")}

    res = _run(["dict", "items", "baz-dict"])
    assert re.search(r" Key .+ Value", res.stdout)
    assert re.search(r" a .+ 123 ", res.stdout)
    assert re.search(r" b .+ blah ", res.stdout)

    res = _run(["dict", "items", "baz-dict", "1"])
    assert re.search(r"\.\.\. .+ \.\.\.", res.stdout)
    assert "blah" not in res.stdout

    res = _run(["dict", "items", "baz-dict", "2"])
    assert "..." not in res.stdout

    res = _run(["dict", "items", "baz-dict", "--json"])
    assert '"Key": "a"' in res.stdout
    assert '"Value": 123' in res.stdout
    assert "..." not in res.stdout

    assert _run(["dict", "get", "baz-dict", "a"]).stdout == "123\n"
    assert _run(["dict", "get", "baz-dict", "b"]).stdout == "blah\n"

    res = _run(["dict", "clear", "baz-dict", "--yes"])
    assert servicer.dicts[dict_id] == {}


def test_queue_create_list_delete(servicer, server_url_env, set_env_client):
    _run(["queue", "create", "foo-queue"])
    _run(["queue", "create", "bar-queue"])
    res = _run(["queue", "list"])
    assert "foo-queue" in res.stdout
    assert "bar-queue" in res.stdout

    _run(["queue", "delete", "bar-queue", "--yes"])

    res = _run(["queue", "list"])
    assert "foo-queue" in res.stdout
    assert "bar-queue" not in res.stdout


def test_queue_peek_len_clear(servicer, server_url_env, set_env_client):
    # Kind of hacky to be modifying the attributes on the servicer like this
    name = "queue-who"
    key = (name, api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, os.environ.get("MODAL_ENVIRONMENT", "main"))
    queue_id = "qu-abc123"
    servicer.deployed_queues[key] = queue_id
    servicer.queue = {b"": [dumps("a"), dumps("b"), dumps("c")], b"alt": [dumps("x"), dumps("y")]}

    assert _run(["queue", "peek", name]).stdout == "a\n"
    assert _run(["queue", "peek", name, "-p", "alt"]).stdout == "x\n"
    assert _run(["queue", "peek", name, "3"]).stdout == "a\nb\nc\n"
    assert _run(["queue", "peek", name, "3", "--partition", "alt"]).stdout == "x\ny\n"

    assert _run(["queue", "len", name]).stdout == "3\n"
    assert _run(["queue", "len", name, "--partition", "alt"]).stdout == "2\n"
    assert _run(["queue", "len", name, "--total"]).stdout == "5\n"

    _run(["queue", "clear", name, "--yes"])
    assert _run(["queue", "len", name]).stdout == "0\n"
    assert _run(["queue", "peek", name, "--partition", "alt"]).stdout == "x\n"

    _run(["queue", "clear", name, "--all", "--yes"])
    assert _run(["queue", "len", name, "--total"]).stdout == "0\n"
    assert _run(["queue", "peek", name, "--partition", "alt"]).stdout == ""


@pytest.mark.parametrize("name", [".main", "_main", "'-main'", "main/main", "main:main"])
def test_create_environment_name_invalid(servicer, set_env_client, name):
    assert isinstance(
        _run(
            ["environment", "create", name],
            1,
        ).exception,
        InvalidError,
    )


@pytest.mark.parametrize("name", ["main", "main_-123."])
def test_create_environment_name_valid(servicer, set_env_client, name):
    assert (
        "Environment created"
        in _run(
            ["environment", "create", name],
            0,
        ).stdout
    )


@pytest.mark.parametrize(("name", "set_name"), (("main", "main/main"), ("main", "'-main'")))
def test_update_environment_name_invalid(servicer, set_env_client, name, set_name):
    assert isinstance(
        _run(
            ["environment", "update", name, "--set-name", set_name],
            1,
        ).exception,
        InvalidError,
    )


@pytest.mark.parametrize(("name", "set_name"), (("main", "main_-123."), ("main:main", "main2")))
def test_update_environment_name_valid(servicer, set_env_client, name, set_name):
    assert (
        "Environment updated"
        in _run(
            ["environment", "update", name, "--set-name", set_name],
            0,
        ).stdout
    )


def test_call_update_environment_suffix(servicer, set_env_client):
    _run(["environment", "update", "main", "--set-web-suffix", "_"])
