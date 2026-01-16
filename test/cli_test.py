# Copyright Modal Labs 2022-2023
import asyncio
import json
import os
import platform
import pytest
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from pickle import dumps
from unittest.mock import MagicMock

import toml

from modal import App, Sandbox
from modal._serialization import PICKLE_PROTOCOL, serialize
from modal._utils.grpc_testing import InterceptionContext
from modal.exception import DeprecationError, InvalidError
from modal_proto import api_pb2

from . import helpers
from .conftest import run_cli_command
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


def windows_sleep():
    # For some time-sensitive operations, we need brief sleeps on Windows so that events
    # don't appear to have happened at the same time with the low-resolution clock
    if platform.system() == "Windows":
        time.sleep(1 / 32)


def test_app_deploy_success(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        # Deploy as a script in cwd
        run_cli_command(["deploy", "myapp.py"])

        # Deploy as a module
        run_cli_command(["deploy", "-m", "myapp"])

        # Deploy as a script with an absolute path
        run_cli_command(["deploy", os.path.abspath("myapp.py")])

    app_names = {app_name for (_, app_name) in servicer.deployed_apps}
    assert "my_app" in app_names


def test_app_deploy_with_name(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        run_cli_command(["deploy", "myapp.py", "--name", "my_app_foo"])

    app_names = {app_name for (_, app_name) in servicer.deployed_apps}
    assert "my_app_foo" in app_names


def test_secret_create_list_delete(servicer, set_env_client):
    # fail without any keys
    run_cli_command(["secret", "create", "foo"], 2, None)

    run_cli_command(["secret", "create", "foo", "VAR=foo"])
    windows_sleep()
    assert "foo" in run_cli_command(["secret", "list"]).stdout

    # Creating the same one again should fail
    run_cli_command(["secret", "create", "foo", "VAR=foo"], expected_exit_code=1)

    # But it should succeed with --force
    run_cli_command(["secret", "create", "foo", "VAR=foo", "--force"])

    # Create a few more
    run_cli_command(["secret", "create", "bar", "VAR=bar"])
    run_cli_command(["secret", "create", "buz", "VAR=buz"])
    windows_sleep()
    assert len(json.loads(run_cli_command(["secret", "list", "--json"]).stdout)) == 3

    # We can delete it
    run_cli_command(["secret", "delete", "foo", "--yes"])
    assert "foo" not in run_cli_command(["secret", "list"]).stdout


@pytest.mark.parametrize(
    ("env_content", "expected_exit_code", "expected_stderr"),
    [
        ("KEY1=VAL1\nKEY2=VAL2", 0, None),
        ("", 2, "You need to specify at least one key for your secret"),
        ("=VAL", 2, "You need to specify at least one key for your secret"),
        ("KEY=", 0, None),
        ("KEY=413", 0, None),  # dotenv reads everything as string...
        ("KEY", 2, "Non-string value"),  # ... except this, which is read as None
    ],
)
def test_secret_create_from_dotenv(
    servicer, set_env_client, tmp_path, env_content, expected_exit_code, expected_stderr
):
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    run_cli_command(
        ["secret", "create", "foo", "--from-dotenv", env_file.as_posix()],
        expected_exit_code=expected_exit_code,
        expected_stderr=expected_stderr,
    )


@pytest.mark.parametrize(
    ("json_content", "expected_exit_code", "expected_stderr"),
    [
        ('{"KEY1": "VAL1",\n"KEY2": "VAL2"}', 0, None),
        ("", 2, "Could not parse JSON file"),
        ("{}", 2, "You need to specify at least one key for your secret"),
        ('{"": ""}', 2, "Invalid key"),
        ('{"KEY": ""}', 0, None),
        ('{"KEY": "413"}', 0, None),
        ('{"KEY": null}', 2, "Non-string value"),
        ('{"KEY": 413}', 2, "Non-string value"),
        ('{"KEY": {"NESTED": "val"}}', 2, "Non-string value"),
    ],
)
def test_secret_create_from_json(servicer, set_env_client, tmp_path, json_content, expected_exit_code, expected_stderr):
    json_file = tmp_path / "test.json"
    json_file.write_text(json_content)
    run_cli_command(
        ["secret", "create", "foo", "--from-json", json_file.as_posix()],
        expected_exit_code=expected_exit_code,
        expected_stderr=expected_stderr,
    )


def test_app_token_new(servicer, set_env_client, server_url_env, modal_config):
    servicer.required_creds = {"abc": "xyz"}
    with modal_config() as config_file_path:
        run_cli_command(["token", "new", "--profile", "_test"])
        assert "_test" in toml.load(config_file_path)


def test_token_env_var_warning(servicer, set_env_client, server_url_env, modal_config, monkeypatch):
    servicer.required_creds = {"abc": "xyz"}
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-123")
    with modal_config():
        res = run_cli_command(["token", "new"])
        assert "MODAL_TOKEN_ID environment variable is" in res.stdout

    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-xyz")
    with modal_config():
        res = run_cli_command(["token", "new"])
        assert "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET environment variables are" in res.stdout


def test_token_info(servicer, set_env_client):
    res = run_cli_command(["token", "info"])
    assert "ak-test123" in res.stdout
    assert "test-workspace" in res.stdout
    assert "test-user" in res.stdout


def test_token_identity_from_env(servicer, set_env_client, monkeypatch):
    # Test that the command shows when credentials are from environment variables
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-from-env")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-from-env")

    res = run_cli_command(["token", "info"])
    # Check for key parts of the message (may have line wrapping, so check individual words)
    assert "Using" in res.stdout
    assert "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET" in res.stdout
    assert "environment" in res.stdout


def test_app_setup(servicer, set_env_client, server_url_env, modal_config):
    servicer.required_creds = {"abc": "xyz"}
    with modal_config() as config_file_path:
        run_cli_command(["setup", "--profile", "_test"])
        assert "_test" in toml.load(config_file_path)


app_file = Path("app_run_tests") / "default_app.py"
app_module = "app_run_tests.default_app"
file_with_entrypoint = Path("app_run_tests") / "local_entrypoint.py"


@pytest.mark.parametrize(
    ("run_command", "expected_exit_code", "expected_output"),
    [
        ([f"{app_file}"], 0, ""),
        ([f"{app_file}::app"], 0, ""),
        ([f"{app_file}::foo"], 0, ""),
        ([f"{app_file}::bar"], 1, ""),
        ([f"{file_with_entrypoint}"], 0, ""),
        ([f"{file_with_entrypoint}::main"], 0, ""),
        ([f"{file_with_entrypoint}::app.main"], 0, ""),
        ([f"{file_with_entrypoint}::foo"], 0, ""),
    ],
)
def test_run(servicer, set_env_client, supports_dir, monkeypatch, run_command, expected_exit_code, expected_output):
    monkeypatch.chdir(supports_dir)
    res = run_cli_command(["run"] + run_command, expected_exit_code=expected_exit_code)
    if expected_output:
        assert re.search(expected_output, res.stdout) or re.search(expected_output, res.stderr), (
            "output does not match expected string"
        )


def test_run_warns_without_module_flag(
    servicer,
    set_env_client,
    supports_dir,
    recwarn,
    monkeypatch,
):
    monkeypatch.chdir(supports_dir)
    run_cli_command(["run", "-m", f"{app_module}::foo"])
    deprecation_warnings = [w.message for w in recwarn if issubclass(w.category, DeprecationError)]
    assert not deprecation_warnings

    with pytest.warns(match=" -m "):
        run_cli_command(["run", f"{app_module}::foo"])


def test_run_async(servicer, set_env_client, test_dir):
    sync_fn = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"
    res = run_cli_command(["run", sync_fn.as_posix()])
    assert "called locally" in res.stdout

    async_fn = test_dir / "supports" / "app_run_tests" / "local_entrypoint_async.py"
    res = run_cli_command(["run", async_fn.as_posix()])
    assert "called locally (async)" in res.stdout


def test_run_generator(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "generator.py"
    result = run_cli_command(["run", app_file.as_posix()], expected_exit_code=1)
    assert "generator functions" in str(result.exception)


def test_help_message_unspecified_function(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "app_with_multiple_functions.py"
    result = run_cli_command(["run", app_file.as_posix()], expected_exit_code=1, expected_stderr=None)

    # should suggest available functions on the app:
    assert "foo" in result.stderr
    assert "bar" in result.stderr

    result = run_cli_command(
        ["run", app_file.as_posix(), "--help"], expected_exit_code=1, expected_stderr=None
    )  # TODO: help should not return non-zero
    # help should also available functions on the app:
    assert "foo" in result.stderr
    assert "bar" in result.stderr


def test_run_states(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    run_cli_command(["run", app_file.as_posix()])
    assert servicer.app_state_history["ap-1"] == [
        api_pb2.APP_STATE_INITIALIZING,
        api_pb2.APP_STATE_EPHEMERAL,
        api_pb2.APP_STATE_STOPPED,
    ]


def test_run_detach(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    run_cli_command(["run", "--detach", app_file.as_posix()])
    assert servicer.app_state_history["ap-1"] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DETACHED]


def test_run_quiet(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    # Just tests that the command runs without error for now (tests end up defaulting to `show_progress=False` anyway,
    # without a TTY).
    run_cli_command(["run", "--quiet", app_file.as_posix()])


def test_run_class_hierarchy(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "class_hierarchy.py"
    run_cli_command(["run", app_file.as_posix() + "::Wrapped.defined_on_base"])
    run_cli_command(["run", app_file.as_posix() + "::Wrapped.overridden_on_wrapped"])


def test_run_write_result(servicer, set_env_client, test_dir):
    # Note that this test only exercises local entrypoint functions,
    # because the servicer doesn't appear to mock remote execution faithfully?
    app_file = (test_dir / "supports" / "app_run_tests" / "returns_data.py").as_posix()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_cli_command(["run", "--write-result", result_file := f"{tmpdir}/result.txt", f"{app_file}::returns_str"])
        with open(result_file, "rt") as f:
            assert f.read() == "Hello!"

        run_cli_command(["run", "-w", result_file := f"{tmpdir}/result.bin", f"{app_file}::returns_bytes"])
        with open(result_file, "rb") as f:
            assert f.read().decode("utf8") == "Hello!"

        run_cli_command(
            ["run", "-w", result_file := f"{tmpdir}/result.bin", f"{app_file}::returns_int"],
            expected_exit_code=1,
            expected_error="Function must return str or bytes when using `--write-result`; got int.",
        )


@pytest.mark.parametrize(
    ["args", "success", "expected_warning"],
    [
        (["--name=deployment_name", str(app_file)], True, ""),
        (["--name=deployment_name", app_module], True, f"modal deploy -m {app_module}"),
        (["--name=deployment_name", "-m", app_module], True, ""),
    ],
)
def test_deploy(servicer, set_env_client, supports_dir, monkeypatch, args, success, expected_warning, recwarn):
    monkeypatch.chdir(supports_dir)
    run_cli_command(["deploy"] + args, expected_exit_code=0 if success else 1)
    if success:
        assert servicer.app_state_history["ap-1"] == [api_pb2.APP_STATE_INITIALIZING, api_pb2.APP_STATE_DEPLOYED]
    else:
        assert api_pb2.APP_STATE_DEPLOYED not in servicer.app_state_history["ap-1"]
    if expected_warning:
        assert len(recwarn) == 1
        assert expected_warning in str(recwarn[0].message)


def test_run_custom_app(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "custom_app.py"
    res = run_cli_command(["run", app_file.as_posix() + "::app"], expected_exit_code=1, expected_stderr=None)
    assert "Specify a Modal Function or local entrypoint to run" in res.stderr
    assert "foo / my_app.foo" in res.stderr
    res = run_cli_command(["run", app_file.as_posix() + "::app.foo"], expected_exit_code=1, expected_stderr=None)
    assert "Specify a Modal Function or local entrypoint" in res.stderr
    assert "foo / my_app.foo" in res.stderr

    run_cli_command(["run", app_file.as_posix() + "::foo"])


def test_run_aiofunc(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "async_app.py"
    run_cli_command(["run", app_file.as_posix()])
    assert len(servicer.function_call_inputs) == 1


def test_run_local_entrypoint(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"

    res = run_cli_command(["run", app_file.as_posix() + "::app.main"])  # explicit name
    assert "called locally" in res.stdout
    assert len(servicer.function_call_inputs) == 2

    res = run_cli_command(["run", app_file.as_posix()])  # only one entry-point, no name needed
    assert "called locally" in res.stdout
    assert len(servicer.function_call_inputs) == 4


def test_run_local_entrypoint_error(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "local_entrypoint.py"
    run_cli_command(
        ["run", "-iq", app_file.as_posix()],
        expected_exit_code=1,
        expected_error="To use interactive mode, remove the --quiet flag",
    )


def test_run_function_error(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"

    run_cli_command(
        ["run", "-iq", app_file.as_posix()],
        expected_exit_code=1,
        expected_error="To use interactive mode, remove the --quiet flag",
    )


def test_run_cls_error(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "cls.py"

    run_cli_command(
        ["run", "-iq", f"{app_file.as_posix()}::AParametrized.some_method", "--x", "42", "--y", "1000"],
        expected_exit_code=1,
        expected_error="To use interactive mode, remove the --quiet flag",
    )


def test_run_local_entrypoint_invalid_with_app_run(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "local_entrypoint_invalid.py"

    res = run_cli_command(["run", app_file.as_posix()], expected_exit_code=1)
    assert "app is already running" in str(res.exception.__cause__).lower()
    assert "unreachable" not in res.stdout
    assert len(servicer.function_call_inputs) == 0


def test_run_parse_args_entrypoint(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"
    res = run_cli_command(["run", app_file.as_posix()], expected_exit_code=1, expected_stderr=None)
    assert "Specify a Modal Function or local entrypoint to run" in res.stderr

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
        res = run_cli_command(args)
        assert expected in res.stdout
        assert len(servicer.function_call_inputs) == 0

    res = run_cli_command(["run", f"{app_file.as_posix()}::unparseable_annot", "--i=20"], expected_exit_code=1)
    assert "Parameter `i` has unparseable annotation: typing.Union[int, str]" in str(res.exception)

    res = run_cli_command(["run", f"{app_file.as_posix()}::unevaluatable_annot", "--i=20"], expected_exit_code=1)
    assert "Unable to generate command line interface" in str(res.exception)
    assert "no go" in str(res.exception)

    if sys.version_info <= (3, 10):
        res = run_cli_command(["run", f"{app_file.as_posix()}::optional_arg_pep604"], expected_exit_code=1)
        assert "Unable to generate command line interface for app entrypoint" in str(res.exception)
        assert "unsupported operand" in str(res.exception)


def test_run_parse_args_function(servicer, set_env_client, test_dir, recwarn):
    app_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"
    res = run_cli_command(["run", app_file.as_posix()], expected_exit_code=1, expected_stderr=None)
    assert "Specify a Modal Function or local entrypoint to run" in res.stderr

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
        res = run_cli_command(args)
        assert expected in res.stdout

    if len(recwarn):
        print("Unexpected warnings:", [str(w) for w in recwarn])
    assert len(recwarn) == 0


def test_run_literal_args_entrypoint(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"

    # Test Literal with strings
    valid_call_args = [
        (["run", f"{app_file.as_posix()}::literal_str_arg", "--mode=write"], "'write' <class 'str'>"),
        (["run", f"{app_file.as_posix()}::literal_int_arg", "--level=2"], "2 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::literal_with_default"], "'dev' <class 'str'>"),
        (["run", f"{app_file.as_posix()}::literal_with_default", "--mode=prod"], "'prod' <class 'str'>"),
        (["run", f"{app_file.as_posix()}::literal_int_with_default"], "2 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::literal_int_with_default", "--level=3"], "3 <class 'int'>"),
    ]
    for args, expected in valid_call_args:
        res = run_cli_command(args)
        assert expected in res.stdout, f"Expected {expected} in output, got: {res.stdout}"
        assert len(servicer.function_call_inputs) == 0

    # Test invalid Literal values
    res = run_cli_command(
        ["run", f"{app_file.as_posix()}::literal_str_arg", "--mode=invalid"],
        expected_exit_code=2,
        expected_stderr=None,
    )
    assert "invalid value" in res.stderr.lower()

    res = run_cli_command(
        ["run", f"{app_file.as_posix()}::literal_int_arg", "--level=99"],
        expected_exit_code=2,
        expected_stderr=None,
    )
    assert "invalid value" in res.stderr.lower()


def test_run_literal_unsupported_types(servicer, set_env_client, test_dir):
    """Test that Literal types with unsupported types are rejected."""
    app_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"

    # Boolean literals should be rejected
    res = run_cli_command(
        ["run", f"{app_file.as_posix()}::literal_bool_arg", "--val=True"],
        expected_exit_code=1,
    )
    assert "unparseable annotation" in str(res.exception).lower()

    # Mixed type literals (str + int) should be rejected
    res = run_cli_command(
        ["run", f"{app_file.as_posix()}::literal_ambiguous_arg", "--val=2"],
        expected_exit_code=1,
    )
    assert "unparseable annotation" in str(res.exception).lower()


def test_run_literal_args_function(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "cli_args.py"

    @servicer.function_body
    def print_type(level):
        print(repr(level), type(level))

    valid_call_args = [
        (["run", f"{app_file.as_posix()}::literal_arg_fn", "--level=1"], "1 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::literal_arg_fn", "--level=2"], "2 <class 'int'>"),
        (["run", f"{app_file.as_posix()}::literal_arg_fn", "--level=3"], "3 <class 'int'>"),
    ]
    for args, expected in valid_call_args:
        res = run_cli_command(args)
        assert expected in res.stdout


def test_run_user_script_exception(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "raises_error.py"
    res = run_cli_command(["run", app_file.as_posix()], expected_exit_code=1)
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
    run_cli_command(["run", fresh_main_thread_assertion_module.as_posix()])
    assert pytest._did_load_main_thread_assertion  # type: ignore
    print()


def test_no_user_code_in_synchronicity_deploy(servicer, set_env_client, test_dir, fresh_main_thread_assertion_module):
    pytest._did_load_main_thread_assertion = False  # type: ignore
    run_cli_command(["deploy", "--name", "foo", fresh_main_thread_assertion_module.as_posix()])
    assert pytest._did_load_main_thread_assertion  # type: ignore


def test_serve(servicer, set_env_client, server_url_env, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "webhook.py"
    run_cli_command(["serve", app_file.as_posix(), "--timeout", "1"], expected_exit_code=0)


def test_app_descriptions(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "prints_desc_app.py"
    run_cli_command(["run", "--detach", app_file.as_posix() + "::foo"])

    create_reqs = [s for s in servicer.requests if isinstance(s, api_pb2.AppCreateRequest)]
    assert len(create_reqs) == 1
    assert create_reqs[0].app_state == api_pb2.APP_STATE_DETACHED
    description = create_reqs[0].description
    assert "prints_desc_app.py::foo" in description
    assert "run --detach " not in description

    run_cli_command(["serve", "--timeout", "0.0", app_file.as_posix()])
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

        # TODO Fix the mock servicer to use "real" App IDs so this does not get misconstrued as a name
        # res = run_cli_command(["app", "logs", "ap-123"])
        # assert res.stdout == "hello\n"

        with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
            res = run_cli_command(["deploy", "myapp.py", "--name", "my-app", "--stream-logs"])
            assert res.stdout.endswith("hello\n")

        res = run_cli_command(["app", "logs", "my-app"])
        assert res.stdout == "hello\n"

    run_cli_command(
        ["app", "logs", "does-not-exist"],
        expected_exit_code=1,
        expected_error="Could not find a deployed app named 'does-not-exist'",
    )


def test_run_timestamps(servicer, server_url_env, set_env_client, test_dir, monkeypatch):
    from datetime import timezone

    # Use a known timestamp (2025-01-15 12:00:45 UTC)
    known_timestamp = 1736942445.0

    # Mock locale_tz to return UTC for predictable timestamp formatting
    monkeypatch.setattr("modal._utils.time_utils.locale_tz", lambda: timezone.utc)

    async def app_logs_with_timestamp(self, stream):
        await stream.recv_message()
        log = api_pb2.TaskLogs(
            data="test log message\n",
            file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT,
            timestamp=known_timestamp,
        )
        await stream.send_message(api_pb2.TaskLogsBatch(entry_id="1", items=[log]))
        await stream.send_message(api_pb2.TaskLogsBatch(app_done=True))

    with servicer.intercept() as ctx:
        ctx.set_responder("AppGetLogs", app_logs_with_timestamp)

        app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"

        # Test without --timestamps flag - should not include timestamp
        res = run_cli_command(["run", app_file.as_posix()])
        assert "test log message" in res.stdout
        # The timestamp string format is "YYYY-MM-DD HH:MM:SS+TZ" - check that it's NOT there
        assert "2025-01-15 12:00:45" not in res.stdout

        # Test with --timestamps flag - should include timestamp prefix
        res = run_cli_command(["run", "--timestamps", app_file.as_posix()])
        # Check for the full formatted line: "YYYY-MM-DD HH:MM:SS+00:00 <message>"
        expected_line = "2025-01-15 12:00:45+00:00 test log message"
        assert expected_line in res.stdout


def test_app_stop(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        # Deploy as a module
        run_cli_command(["deploy", "-m", "myapp"])

    res = run_cli_command(["app", "list"])
    assert re.search("my_app .+ deployed", res.stdout)

    run_cli_command(["app", "stop", "my_app"])

    # Note that the mock servicer doesn't report "stopped" app statuses
    # so we just check that it's not reported as deployed
    res = run_cli_command(["app", "list"])
    assert not re.search("my_app .+ deployed", res.stdout)


def test_nfs_get(set_env_client, servicer):
    nfs_name = "my-shared-nfs"
    run_cli_command(["nfs", "create", nfs_name])
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "w") as f:
            f.write("foo bar baz")
            f.flush()
        run_cli_command(["nfs", "put", nfs_name, upload_path, "test.txt"])

        run_cli_command(["nfs", "get", nfs_name, "test.txt", tmpdir])
        with open(os.path.join(tmpdir, "test.txt")) as f:
            assert f.read() == "foo bar baz"


def test_nfs_create_delete(servicer, server_url_env, set_env_client):
    name = "test-delete-nfs"
    run_cli_command(["nfs", "create", name])
    assert name in run_cli_command(["nfs", "list"]).stdout
    run_cli_command(["nfs", "delete", "--yes", name])
    assert name not in run_cli_command(["nfs", "list"]).stdout


def test_volume_cli(set_env_client):
    run_cli_command(["volume", "--help"])


def test_volume_get(servicer, set_env_client):
    vol_name = "my-test-vol"
    run_cli_command(["volume", "create", vol_name])
    file_path = "test.txt"
    file_contents = b"foo bar baz"
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "wb") as f:
            f.write(file_contents)
            f.flush()
        run_cli_command(["volume", "put", vol_name, upload_path, file_path])

        run_cli_command(["volume", "get", vol_name, file_path, tmpdir])
        with open(os.path.join(tmpdir, file_path), "rb") as f:
            assert f.read() == file_contents

        download_path = os.path.join(tmpdir, "download.txt")
        run_cli_command(["volume", "get", vol_name, file_path, download_path])
        with open(download_path, "rb") as f:
            assert f.read() == file_contents

    with tempfile.TemporaryDirectory() as tmpdir2:
        run_cli_command(["volume", "get", vol_name, "/", tmpdir2])
        with open(os.path.join(tmpdir2, file_path), "rb") as f:
            assert f.read() == file_contents


def test_volume_put_force(servicer, set_env_client):
    vol_name = "my-test-vol"
    run_cli_command(["volume", "create", vol_name])
    file_path = "test.txt"
    file_contents = b"foo bar baz"
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "wb") as f:
            f.write(file_contents)
            f.flush()

        # File upload
        run_cli_command(["volume", "put", vol_name, upload_path, file_path])  # Seed the volume
        with servicer.intercept() as ctx:
            run_cli_command(
                ["volume", "put", vol_name, upload_path, file_path], expected_exit_code=2, expected_stderr=None
            )
            assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            run_cli_command(["volume", "put", vol_name, upload_path, file_path, "--force"])
            assert not ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

        # Dir upload
        run_cli_command(["volume", "put", vol_name, tmpdir])  # Seed the volume
        with servicer.intercept() as ctx:
            run_cli_command(["volume", "put", vol_name, tmpdir], expected_exit_code=2, expected_stderr=None)
            assert ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files

            run_cli_command(["volume", "put", vol_name, tmpdir, "--force"])
            assert not ctx.pop_request("VolumePutFiles").disallow_overwrite_existing_files


def test_volume_rm(servicer, set_env_client):
    vol_name = "my-test-vol"
    run_cli_command(["volume", "create", vol_name])
    file_path = "test.txt"
    file_contents = b"foo bar baz"
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = os.path.join(tmpdir, "upload.txt")
        with open(upload_path, "wb") as f:
            f.write(file_contents)
            f.flush()
        run_cli_command(["volume", "put", vol_name, upload_path, file_path])

        run_cli_command(["volume", "get", vol_name, file_path, tmpdir])
        with open(os.path.join(tmpdir, file_path), "rb") as f:
            assert f.read() == file_contents

        run_cli_command(["volume", "rm", vol_name, file_path])
        run_cli_command(["volume", "get", vol_name, file_path], expected_exit_code=1, expected_stderr=None)


def test_volume_ls(servicer, set_env_client):
    vol_name = "my-test-vol"
    run_cli_command(["volume", "create", vol_name])

    fnames = ["a", "b", "c"]
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in fnames:
            src_path = os.path.join(tmpdir, f"{fname}.txt")
            with open(src_path, "w") as f:
                f.write(fname * 5)
            run_cli_command(["volume", "put", vol_name, src_path, f"data/{fname}.txt"])

    res = run_cli_command(["volume", "ls", vol_name])
    assert "data" in res.stdout

    res = run_cli_command(["volume", "ls", vol_name, "data"])
    for fname in fnames:
        assert f"{fname}.txt" in res.stdout

    res = run_cli_command(["volume", "ls", vol_name, "data", "--json"])
    res_dict = json.loads(res.stdout)
    assert len(res_dict) == len(fnames)
    for entry, fname in zip(res_dict, fnames):
        assert entry["Filename"] == f"data/{fname}.txt"
        assert entry["Type"] == "file"


def test_volume_create_delete(servicer, server_url_env, set_env_client):
    vol_name = "test-delete-vol"
    run_cli_command(["volume", "create", vol_name])
    windows_sleep()
    assert vol_name in run_cli_command(["volume", "list"]).stdout
    run_cli_command(["volume", "delete", "--yes", vol_name])
    assert vol_name not in run_cli_command(["volume", "list"]).stdout


def test_volume_rename(servicer, server_url_env, set_env_client):
    old_name, new_name = "foo-vol", "bar-vol"
    run_cli_command(["volume", "create", old_name])
    run_cli_command(["volume", "rename", "--yes", old_name, new_name])
    windows_sleep()
    assert new_name in run_cli_command(["volume", "list"]).stdout
    assert old_name not in run_cli_command(["volume", "list"]).stdout


@pytest.mark.parametrize("command", [["shell"]])
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
        run_cli_command(command + ["--env=staging", str(app_file)])

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
        run_cli_command(command + [str(app_file)])

    app_create: api_pb2.AppCreateRequest = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "some_weird_default_env"


def test_cls(servicer, set_env_client, test_dir):
    app_file = test_dir / "supports" / "app_run_tests" / "cls.py"

    print(run_cli_command(["run", app_file.as_posix(), "--x", "42", "--y", "1000"]))
    run_cli_command(["run", f"{app_file.as_posix()}::AParametrized.some_method", "--x", "42", "--y", "1000"])


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
        servicer.required_creds = {"ak-abc": "as-xyz", "ak-123": "as-789"}
        res = run_cli_command(["profile", "list"])
        table_rows = res.stdout.split("\n")
        assert re.search("Profile .+ Workspace", table_rows[1])
        assert re.search("test-profile .+ test-username", table_rows[3])
        assert re.search("other-profile .+ test-username", table_rows[4])

        res = run_cli_command(["profile", "list", "--json"])
        json_data = json.loads(res.stdout)
        assert json_data[0]["name"] == "test-profile"
        assert json_data[0]["workspace"] == "test-username"
        assert json_data[1]["name"] == "other-profile"
        assert json_data[1]["workspace"] == "test-username"

        orig_env_token_id = os.environ.get("MODAL_TOKEN_ID")
        orig_env_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        os.environ["MODAL_TOKEN_ID"] = "ak-abc"
        os.environ["MODAL_TOKEN_SECRET"] = "as-xyz"
        servicer.required_creds = {"ak-abc": "as-xyz"}
        try:
            res = run_cli_command(["profile", "list"])
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


def test_config_show(servicer, server_url_env, modal_config):
    config = """
    [test-profile]
    token_id = "ak-abc"
    token_secret = "as-xyz"
    active = true
    """
    with modal_config(config):
        res = run_cli_command(["config", "show"])
        assert '"token_id": "ak-abc"' in res.stdout
        assert '"token_secret": "***"' in res.stdout


def test_app_list(servicer, mock_dir, set_env_client):
    res = run_cli_command(["app", "list"])
    assert "my_app_foo" not in res.stdout

    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        run_cli_command(["deploy", "myapp.py", "--name", "my_app_foo"])

    res = run_cli_command(["app", "list"])
    assert "my_app_foo" in res.stdout

    res = run_cli_command(["app", "list", "--json"])
    assert json.loads(res.stdout)

    run_cli_command(["volume", "create", "my-vol"])
    res = run_cli_command(["app", "list"])
    assert "my-vol" not in res.stdout


def test_app_history(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        run_cli_command(["deploy", "myapp.py", "--name", "my_app_foo"])

    app_id = servicer.deployed_apps.get(("main", "my_app_foo"))

    servicer.app_deployment_history[app_id][-1]["commit_info"] = api_pb2.CommitInfo(
        vcs="git", branch="main", commit_hash="abc123"
    )

    # app should be deployed once it exists
    res = run_cli_command(["app", "history", "my_app_foo"])
    assert "v1" in res.stdout, res.stdout

    res = run_cli_command(["app", "history", "my_app_foo", "--json"])
    assert json.loads(res.stdout)

    # re-deploying an app should result in a new row in the history table
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        run_cli_command(["deploy", "myapp.py", "--name", "my_app_foo"])

    servicer.app_deployment_history[app_id][-1]["commit_info"] = api_pb2.CommitInfo(
        vcs="git", branch="main", commit_hash="def456", dirty=True
    )

    res = run_cli_command(["app", "history", "my_app_foo"])
    assert "v1" in res.stdout
    assert "v2" in res.stdout, f"{res.stdout=}"
    assert "abc123" in res.stdout
    assert "def456*" in res.stdout

    # can't fetch history for stopped apps
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        run_cli_command(["app", "stop", "my_app_foo"])

    res = run_cli_command(["app", "history", "my_app_foo", "--json"], expected_exit_code=1)


def test_app_rollback(servicer, mock_dir, set_env_client):
    with mock_dir({"myapp.py": dummy_app_file, "other_module.py": dummy_other_module_file}):
        # Deploy multiple times
        for _ in range(4):
            run_cli_command(["deploy", "myapp.py", "--name", "my_app"])
    run_cli_command(["app", "rollback", "my_app"])
    app_id = servicer.deployed_apps.get(("main", "my_app"))
    assert servicer.app_deployment_history[app_id][-1]["rollback_version"] == 3

    run_cli_command(["app", "rollback", "my_app", "v2"])
    app_id = servicer.deployed_apps.get(("main", "my_app"))
    assert servicer.app_deployment_history[app_id][-1]["rollback_version"] == 2

    run_cli_command(["app", "rollback", "my_app", "2"], expected_exit_code=2)


def test_dict_create_list_delete(servicer, server_url_env, set_env_client):
    run_cli_command(["dict", "create", "foo-dict"])
    run_cli_command(["dict", "create", "bar-dict"])
    windows_sleep()
    res = run_cli_command(["dict", "list"])
    assert "foo-dict" in res.stdout
    assert "bar-dict" in res.stdout

    run_cli_command(["dict", "delete", "bar-dict", "--yes"])
    res = run_cli_command(["dict", "list"])
    assert "foo-dict" in res.stdout
    assert "bar-dict" not in res.stdout


def test_dict_show_get_clear(servicer, server_url_env, set_env_client):
    # Kind of hacky to be modifying the attributes on the servicer like this
    key = ("baz-dict", os.environ.get("MODAL_ENVIRONMENT", "main"))
    dict_id = "di-abc123"
    servicer.deployed_dicts[key] = dict_id
    servicer.dicts[dict_id] = {
        dumps("a", protocol=PICKLE_PROTOCOL): dumps(123, protocol=PICKLE_PROTOCOL),
        dumps("b", protocol=PICKLE_PROTOCOL): dumps("blah", protocol=PICKLE_PROTOCOL),
    }

    res = run_cli_command(["dict", "items", "baz-dict"])
    assert re.search(r" Key .+ Value", res.stdout)
    assert re.search(r" a .+ 123 ", res.stdout)
    assert re.search(r" b .+ blah ", res.stdout)

    res = run_cli_command(["dict", "items", "baz-dict", "1"])
    assert re.search(r"\.\.\. .+ \.\.\.", res.stdout)
    assert "blah" not in res.stdout

    res = run_cli_command(["dict", "items", "baz-dict", "2"])
    assert "..." not in res.stdout

    res = run_cli_command(["dict", "items", "baz-dict", "--json"])
    assert '"Key": "a"' in res.stdout
    assert '"Value": 123' in res.stdout
    assert "..." not in res.stdout

    assert run_cli_command(["dict", "get", "baz-dict", "a"]).stdout == "123\n"
    assert run_cli_command(["dict", "get", "baz-dict", "b"]).stdout == "blah\n"

    res = run_cli_command(["dict", "clear", "baz-dict", "--yes"])
    assert servicer.dicts[dict_id] == {}


def test_queue_create_list_delete(servicer, server_url_env, set_env_client):
    run_cli_command(["queue", "create", "foo-queue"])
    run_cli_command(["queue", "create", "bar-queue"])
    windows_sleep()
    res = run_cli_command(["queue", "list"])
    assert "foo-queue" in res.stdout
    assert "bar-queue" in res.stdout

    run_cli_command(["queue", "delete", "bar-queue", "--yes"])

    res = run_cli_command(["queue", "list"])
    assert "foo-queue" in res.stdout
    assert "bar-queue" not in res.stdout


def test_queue_peek_len_clear(servicer, server_url_env, set_env_client):
    # Kind of hacky to be modifying the attributes on the servicer like this
    name = "queue-who"
    key = (name, os.environ.get("MODAL_ENVIRONMENT", "main"))
    queue_id = "qu-abc123"
    servicer.deployed_queues[key] = queue_id
    servicer.queue = {b"": [dumps("a"), dumps("b"), dumps("c")], b"alt": [dumps("x"), dumps("y")]}

    assert run_cli_command(["queue", "peek", name]).stdout == "a\n"
    assert run_cli_command(["queue", "peek", name, "-p", "alt"]).stdout == "x\n"
    assert run_cli_command(["queue", "peek", name, "3"]).stdout == "a\nb\nc\n"
    assert run_cli_command(["queue", "peek", name, "3", "--partition", "alt"]).stdout == "x\ny\n"

    assert run_cli_command(["queue", "len", name]).stdout == "3\n"
    assert run_cli_command(["queue", "len", name, "--partition", "alt"]).stdout == "2\n"
    assert run_cli_command(["queue", "len", name, "--total"]).stdout == "5\n"

    run_cli_command(["queue", "clear", name, "--yes"])
    assert run_cli_command(["queue", "len", name]).stdout == "0\n"
    assert run_cli_command(["queue", "peek", name, "--partition", "alt"]).stdout == "x\n"

    run_cli_command(["queue", "clear", name, "--all", "--yes"])
    assert run_cli_command(["queue", "len", name, "--total"]).stdout == "0\n"
    assert run_cli_command(["queue", "peek", name, "--partition", "alt"]).stdout == ""


@pytest.mark.parametrize("name", [".main", "_main", "'-main'", "main/main", "main:main"])
def test_create_environment_name_invalid(servicer, set_env_client, name):
    assert isinstance(
        run_cli_command(
            ["environment", "create", name],
            1,
        ).exception,
        InvalidError,
    )


@pytest.mark.parametrize("name", ["main", "main_-123."])
def test_create_environment_name_valid(servicer, set_env_client, name):
    assert (
        "Environment created"
        in run_cli_command(
            ["environment", "create", name],
            0,
        ).stdout
    )


@pytest.mark.parametrize(("name", "set_name"), (("main", "main/main"), ("main", "'-main'")))
def test_update_environment_name_invalid(servicer, set_env_client, name, set_name):
    assert isinstance(
        run_cli_command(
            ["environment", "update", name, "--set-name", set_name],
            1,
        ).exception,
        InvalidError,
    )


@pytest.mark.parametrize(("name", "set_name"), (("main", "main_-123."), ("main:main", "main2")))
def test_update_environment_name_valid(servicer, set_env_client, name, set_name):
    assert (
        "Environment updated"
        in run_cli_command(
            ["environment", "update", name, "--set-name", set_name],
            0,
        ).stdout
    )


def test_call_update_environment_suffix(servicer, set_env_client):
    run_cli_command(["environment", "update", "main", "--set-web-suffix", "_"])


def _run_subprocess(cli_cmd: list[str]) -> helpers.PopenWithCtrlC:
    p = helpers.PopenWithCtrlC(
        [sys.executable, "-m", "modal"] + cli_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
    )
    return p


@pytest.mark.timeout(10)
def test_keyboard_interrupt_during_app_load(servicer, server_url_env, token_env, supports_dir):
    ctx: InterceptionContext
    creating_function = threading.Event()

    async def stalling_function_create(servicer, req):
        creating_function.set()
        await asyncio.sleep(10)

    with servicer.intercept() as ctx:
        ctx.set_responder("FunctionCreate", stalling_function_create)

        p = _run_subprocess(["run", f"{supports_dir / 'hello.py'}::hello"])
        creating_function.wait()
        p.send_ctrl_c()
        out, err = p.communicate(timeout=5)
        print(out)
        assert "Traceback" not in err
        assert "Aborting app initialization..." in out


@pytest.mark.timeout(10)
def test_keyboard_interrupt_during_app_run(servicer, server_url_env, token_env, supports_dir):
    ctx: InterceptionContext
    waiting_for_output = threading.Event()

    async def stalling_function_get_output(servicer, req):
        waiting_for_output.set()
        await asyncio.sleep(10)

    with servicer.intercept() as ctx:
        ctx.set_responder("FunctionGetOutputs", stalling_function_get_output)

        p = _run_subprocess(["run", f"{supports_dir / 'hello.py'}::hello"])
        waiting_for_output.wait()
        p.send_ctrl_c()
        out, err = p.communicate(timeout=5)
        assert "App aborted. View run at https://modaltest.com/apps/ap-123" in out
        assert "Traceback" not in err


@pytest.mark.timeout(10)
def test_keyboard_interrupt_during_app_run_detach(servicer, server_url_env, token_env, supports_dir):
    ctx: InterceptionContext
    waiting_for_output = threading.Event()

    async def stalling_function_get_output(servicer, req):
        waiting_for_output.set()
        await asyncio.sleep(10)

    with servicer.intercept() as ctx:
        ctx.set_responder("FunctionGetOutputs", stalling_function_get_output)

        p = _run_subprocess(["run", "--detach", f"{supports_dir / 'hello.py'}::hello"])
        waiting_for_output.wait()
        p.send_ctrl_c()
        out, err = p.communicate(timeout=5)
        print(out)
        assert "Shutting down Modal client." in out
        assert "track its progress" in out
        assert "modal app stop" in out
        assert "modal app logs" in out
        assert "Traceback" not in err


@pytest.fixture
def app(client):
    app = App()
    with app.run(client=client):
        yield app


@skip_windows("modal shell is not supported on Windows.")
def test_container_exec(servicer, set_env_client, mock_shell_pty, app):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)

    fake_stdin, captured_out = mock_shell_pty

    fake_stdin.clear()
    fake_stdin.extend([b'echo "Hello World"\n', b"exit\n"])

    shell_prompt = servicer.shell_prompt

    run_cli_command(["container", "exec", "--pty", sb.object_id, "/bin/bash"])
    assert captured_out == [(1, shell_prompt), (1, b"Hello World\n")]
    captured_out.clear()

    sb.terminate()


def test_can_run_all_listed_functions_with_includes(supports_on_path, monkeypatch, set_env_client):
    monkeypatch.setenv("TERM", "dumb")  # prevents looking at ansi escape sequences

    res = run_cli_command(["run", "-m", "multifile_project.main"], expected_exit_code=1)
    print("err", res.stderr)
    # there are no runnables directly in the target module, so references need to go via the app
    func_listing = res.stderr.split("functions and local entrypoints:")[1]

    listed_runnables = set(re.findall(r"\b[\w.]+\b", func_listing))

    expected_runnables = {
        "app.a_func",
        "app.b_func",
        "app.c_func",
        "app.main_function",
        "main_function",
        "Cls.method_on_other_app_class",
        "other_app.Cls.method_on_other_app_class",
    }
    assert listed_runnables == expected_runnables

    for runnable in expected_runnables:
        assert runnable in res.stderr
        run_cli_command(["run", "-m", f"multifile_project.main::{runnable}"], expected_exit_code=0)


def test_modal_launch_vscode(monkeypatch, set_env_client, servicer):
    mock_open = MagicMock()
    monkeypatch.setattr("webbrowser.open", mock_open)
    with servicer.intercept() as ctx:
        ctx.add_response("QueueGet", api_pb2.QueueGetResponse(values=[serialize(("http://dummy", "tok"))]))
        ctx.add_response("QueueGet", api_pb2.QueueGetResponse(values=[serialize("done")]))
        run_cli_command(["launch", "vscode"])

    assert mock_open.call_count == 1


def test_run_file_with_global_lookups(servicer, set_env_client, supports_dir):
    # having module-global Function/Cls objects from .from_name constructors shouldn't
    # cause issues, and they shouldn't be runnable via CLI (for now)
    with servicer.intercept() as ctx:
        run_cli_command(["run", str(supports_dir / "app_run_tests" / "file_with_global_lookups.py")])

    (req,) = ctx.get_requests("FunctionCreate")
    assert req.function.function_name == "local_f"
    assert len(ctx.get_requests("FunctionMap")) == 1
    assert len(ctx.get_requests("FunctionGet")) == 0


def test_run_auto_infer_prefer_target_module(servicer, supports_dir, set_env_client, monkeypatch):
    monkeypatch.syspath_prepend(supports_dir / "app_run_tests")
    res = run_cli_command(["run", "-m", "multifile.util"])
    assert "ran util\nmain func" in res.stdout


@pytest.mark.parametrize("func", ["va_entrypoint", "va_function", "VaClass.va_method"])
def test_cli_run_variadic_args(servicer, set_env_client, test_dir, func):
    app_file = test_dir / "supports" / "app_run_tests" / "variadic_args.py"

    @servicer.function_body
    def print_args(*args):
        print(f"args: {args}")

    res = run_cli_command(["run", f"{app_file.as_posix()}::{func}"])
    assert "args: ()" in res.stdout

    res = run_cli_command(["run", f"{app_file.as_posix()}::{func}", "abc", "--foo=123", "--bar=456"])
    assert "args: ('abc', '--foo=123', '--bar=456')" in res.stdout

    run_cli_command(["run", f"{app_file.as_posix()}::{func}_invalid", "--foo=123"], expected_exit_code=1)


def test_server_warnings(servicer, set_env_client, supports_dir):
    res = run_cli_command(["run", f"{supports_dir / 'app_run_tests' / 'uses_experimental_options.py'}::gets_warning"])
    assert "You have been warned!" in res.stdout


def test_run_with_options(servicer, set_env_client, supports_dir):
    app_file = supports_dir / "app_run_tests" / "uses_with_options.py"
    run_cli_command(["run", f"{app_file.as_posix()}::C_with_gpu.f"])
