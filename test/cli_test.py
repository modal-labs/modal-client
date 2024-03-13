# Copyright Modal Labs 2022-2023
import json
import os
import pytest
import re
import sys
import tempfile
import traceback
from typing import List, Optional
from unittest import mock

import click
import click.testing
import pytest_asyncio
import toml

from modal import Client
from modal._utils.async_utils import asyncnullcontext
from modal.cli.entry_point import entrypoint_cli
from modal_proto import api_pb2

from .supports.skip import skip_windows

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
async def set_env_client(client):
    try:
        Client.set_env_client(client)
        yield
    finally:
        Client.set_env_client(None)


def _run(args: List[str], expected_exit_code: int = 0, expected_stderr: Optional[str] = ""):
    runner = click.testing.CliRunner(mix_stderr=False)
    with mock.patch.object(sys, "argv", args):
        res = runner.invoke(entrypoint_cli, args)
    if res.exit_code != expected_exit_code:
        print("stdout:", repr(res.stdout))
        print("stderr:", repr(res.stderr))
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


def test_run_async(servicer, set_env_client, test_dir):
    sync_fn = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"
    res = _run(["run", sync_fn.as_posix()])
    assert "called locally" in res.stdout

    async_fn = test_dir / "supports" / "app_run_tests" / "local_entrypoint_async.py"
    res = _run(["run", async_fn.as_posix()])
    assert "called locally (async)" in res.stdout


def test_run_generator(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "generator.py"
    result = _run(["run", stub_file.as_posix()], expected_exit_code=1)
    assert "generator functions" in str(result.exception)


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


def test_run_aiofunc(servicer, set_env_client, test_dir):
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


def test_run_local_entrypoint_invalid_with_stub_run(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "local_entrypoint_invalid.py"

    res = _run(["run", stub_file.as_posix()], expected_exit_code=1)
    assert "app is already running" in str(res.exception.__cause__).lower()
    assert "unreachable" not in res.stdout
    assert len(servicer.client_calls) == 0


def test_run_parse_args_entrypoint(servicer, set_env_client, test_dir):
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
        (["run", f"{stub_file.as_posix()}::int_arg", "--i=200"], "200 <class 'int'>"),
        (["run", f"{stub_file.as_posix()}::default_arg"], "10 <class 'int'>"),
        (["run", f"{stub_file.as_posix()}::unannotated_arg", "--i=2022-10-31"], "'2022-10-31' <class 'str'>"),
        (["run", f"{stub_file.as_posix()}::unannotated_default_arg"], "10 <class 'int'>"),
        (["run", f"{stub_file.as_posix()}::optional_arg", "--i=20"], "20 <class 'int'>"),
        (["run", f"{stub_file.as_posix()}::optional_arg"], "None <class 'NoneType'>"),
        (["run", f"{stub_file.as_posix()}::optional_arg_postponed"], "None <class 'NoneType'>"),
    ]
    if sys.version_info >= (3, 10):
        valid_call_args.extend(
            [
                (["run", f"{stub_file.as_posix()}::optional_arg_pep604", "--i=20"], "20 <class 'int'>"),
                (["run", f"{stub_file.as_posix()}::optional_arg_pep604"], "None <class 'NoneType'>"),
            ]
        )
    for args, expected in valid_call_args:
        res = _run(args)
        assert expected in res.stdout
        assert len(servicer.client_calls) == 0

    if sys.version_info >= (3, 10):
        res = _run(["run", f"{stub_file.as_posix()}::unparseable_annot", "--i=20"], expected_exit_code=1)
        assert "Parameter `i` has unparseable annotation: typing.Union[int, str]" in str(res.exception)

    if sys.version_info <= (3, 10):
        res = _run(["run", f"{stub_file.as_posix()}::optional_arg_pep604"], expected_exit_code=1)
        assert "Unable to generate command line interface for app entrypoint." in str(res.exception)


def test_run_parse_args_function(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"
    res = _run(["run", stub_file.as_posix()], expected_exit_code=2, expected_stderr=None)
    assert "You need to specify a Modal function or local entrypoint to run" in res.stderr

    # HACK: all the tests use the same arg, i.
    @servicer.function_body
    def print_type(i):
        print(repr(i), type(i))

    valid_call_args = [
        (["run", f"{stub_file.as_posix()}::int_arg_fn", "--i=200"], "200 <class 'int'>"),
        (["run", f"{stub_file.as_posix()}::ALifecycle.some_method", "--i=hello"], "'hello' <class 'str'>"),
        (["run", f"{stub_file.as_posix()}::ALifecycle.some_method_int", "--i=42"], "42 <class 'int'>"),
        (["run", f"{stub_file.as_posix()}::optional_arg_fn"], "None <class 'NoneType'>"),
    ]
    for args, expected in valid_call_args:
        res = _run(args)
        assert expected in res.stdout


def test_run_user_script_exception(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "raises_error.py"
    res = _run(["run", stub_file.as_posix()], expected_exit_code=1)
    assert res.exc_info[1].user_source == str(stub_file.resolve())


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
    stub_file = test_dir / "supports" / "app_run_tests" / "webhook.py"
    _run(["serve", stub_file.as_posix(), "--timeout", "3"], expected_exit_code=0)


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
        )

    captured_out = []

    async def write_to_fd(fd: int, data: bytes):
        nonlocal captured_out
        captured_out.append((fd, data))

    with mock.patch("rich.console.Console.is_terminal", True), mock.patch(
        "modal._pty.get_pty_info", mock_get_pty_info
    ), mock.patch("modal._container_exec.get_pty_info", mock_get_pty_info), mock.patch(
        "modal._container_exec.handle_exec_input", asyncnullcontext
    ), mock.patch("modal._container_exec._write_to_fd", write_to_fd):
        yield captured_out


@skip_windows("modal shell is not supported on Windows.")
def test_shell(servicer, set_env_client, test_dir, mock_shell_pty):
    stub_file = test_dir / "supports" / "app_run_tests" / "default_stub.py"
    webhook_stub_file = test_dir / "supports" / "app_run_tests" / "webhook.py"

    # Function is explicitly specified
    _run(["shell", stub_file.as_posix() + "::foo"])
    assert mock_shell_pty == [(1, b"Hello World")]
    mock_shell_pty.clear()

    # Function is explicitly specified
    _run(["shell", webhook_stub_file.as_posix() + "::foo"])
    assert mock_shell_pty == [(1, b"Hello World")]
    mock_shell_pty.clear()

    # Function must be inferred
    _run(["shell", webhook_stub_file.as_posix()])
    assert mock_shell_pty == [(1, b"Hello World")]
    mock_shell_pty.clear()


def test_app_descriptions(servicer, server_url_env, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "prints_desc_stub.py"
    _run(["run", "--detach", stub_file.as_posix() + "::stub.foo"])

    create_reqs = [s for s in servicer.requests if isinstance(s, api_pb2.AppCreateRequest)]
    assert len(create_reqs) == 1
    assert create_reqs[0].app_state == api_pb2.APP_STATE_DETACHED
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


def test_nfs_get(set_env_client):
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
    file_path = b"test.txt"
    file_contents = b"foo bar baz"
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "wb") as f:
            f.write(file_contents)
            f.flush()
        _run(["volume", "put", vol_name, upload_path, file_path.decode()])

        _run(["volume", "get", vol_name, file_path.decode(), tmpdir])
        with open(os.path.join(tmpdir, file_path.decode()), "rb") as f:
            assert f.read() == file_contents

    with tempfile.TemporaryDirectory() as tmpdir2:
        _run(["volume", "get", vol_name, "**", tmpdir2])
        with open(os.path.join(tmpdir2, file_path.decode()), "rb") as f:
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
    file_path = b"test.txt"
    file_contents = b"foo bar baz"
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "wb") as f:
            f.write(file_contents)
            f.flush()
        _run(["volume", "put", vol_name, upload_path, file_path.decode()])

        _run(["volume", "get", vol_name, file_path.decode(), tmpdir])
        with open(os.path.join(tmpdir, file_path.decode()), "rb") as f:
            assert f.read() == file_contents

        _run(["volume", "rm", vol_name, file_path.decode()])
        _run(["volume", "get", vol_name, file_path.decode()], expected_exit_code=2, expected_stderr=None)


@pytest.mark.parametrize("command", [["run"], ["deploy"], ["serve", "--timeout=1"], ["shell"]])
@pytest.mark.usefixtures("set_env_client", "mock_shell_pty")
def test_environment_flag(test_dir, servicer, command):
    @servicer.function_body
    def nothing(
        arg=None,
    ):  # hacky - compatible with both argless modal run and interactive mode which always sends an arg...
        pass

    stub_file = test_dir / "supports" / "app_run_tests" / "app_with_lookups.py"
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
        _run(command + ["--env=staging", str(stub_file)])

    app_create: api_pb2.AppCreateRequest = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "staging"


@pytest.mark.parametrize("command", [["run"], ["deploy"], ["serve", "--timeout=1"], ["shell"]])
@pytest.mark.usefixtures("set_env_client", "mock_shell_pty")
def test_environment_noflag(test_dir, servicer, command, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "some_weird_default_env")

    @servicer.function_body
    def nothing(
        arg=None,
    ):  # hacky - compatible with both argless modal run and interactive mode which always sends an arg...
        pass

    stub_file = test_dir / "supports" / "app_run_tests" / "app_with_lookups.py"

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
        _run(command + [str(stub_file)])

    app_create: api_pb2.AppCreateRequest = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "some_weird_default_env"


def test_cls(servicer, set_env_client, test_dir):
    stub_file = test_dir / "supports" / "app_run_tests" / "cls.py"

    _run(["run", stub_file.as_posix(), "--x", "42", "--y", "1000"])
    _run(["run", f"{stub_file.as_posix()}::AParametrized.some_method", "--x", "42", "--y", "1000"])


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
