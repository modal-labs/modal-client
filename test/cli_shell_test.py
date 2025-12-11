# Copyright Modal Labs 2022-2023
import inspect
import platform
import pytest
import re
import subprocess
from pathlib import Path
from unittest.mock import Mock

import typer

from modal.cli.shell import _params_from_signature, _passed_forbidden_args, shell

from .conftest import run_cli_command
from .supports.skip import skip_windows

app_file = Path("app_run_tests") / "default_app.py"
app_file_as_module = "app_run_tests.default_app"
webhook_app_file = Path("app_run_tests") / "webhook.py"
cls_app_file = Path("app_run_tests") / "cls.py"


@skip_windows("modal shell is not supported on Windows.")
@pytest.mark.parametrize(
    ["flags", "rel_file", "suffix"],
    [
        ([], app_file, "::foo"),  # Function is explicitly specified
        (["-m"], app_file_as_module, "::foo"),  # Function is explicitly specified - module mode
        ([], webhook_app_file, "::foo"),  # Function is explicitly specified
        ([], webhook_app_file, ""),  # Function must be inferred
        # TODO: fix modal shell auto-detection of a single class, even if it has multiple methods
        # ([], cls_app_file, ""),  # Class must be inferred
        # ([], cls_app_file, "AParametrized"),  # class name
        ([], cls_app_file, "::AParametrized.some_method"),  # method name
    ],
)
def test_shell(servicer, set_env_client, mock_shell_pty, suffix, monkeypatch, supports_dir, rel_file, flags):
    monkeypatch.chdir(supports_dir)
    fake_stdin, captured_out = mock_shell_pty

    fake_stdin.clear()
    fake_stdin.extend([b'echo "Hello World"\n', b"exit\n"])

    shell_prompt = servicer.shell_prompt

    run_cli_command(["shell"] + flags + [str(rel_file) + suffix])

    # first captured message is the empty message the mock server sends
    assert captured_out == [(1, shell_prompt), (1, b"Hello World\n")]
    captured_out.clear()


@skip_windows("modal shell is not supported on Windows.")
def test_shell_cmd(servicer, set_env_client, test_dir, mock_shell_pty):
    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    _, captured_out = mock_shell_pty
    shell_prompt = servicer.shell_prompt
    run_cli_command(["shell", "--cmd", "pwd", app_file.as_posix() + "::foo"])
    expected_output = subprocess.run(["pwd"], capture_output=True, check=True).stdout
    assert captured_out == [(1, shell_prompt), (1, expected_output)]


@skip_windows("modal shell is not supported on Windows.")
def test_shell_preserve_token(servicer, set_env_client, mock_shell_pty, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "my-token-id")

    fake_stdin, captured_out = mock_shell_pty
    shell_prompt = servicer.shell_prompt

    fake_stdin.clear()
    fake_stdin.extend([b'echo "$MODAL_TOKEN_ID"\n', b"exit\n"])
    run_cli_command(["shell"])

    expected_output = b"my-token-id\n"
    assert captured_out == [(1, shell_prompt), (1, expected_output)]


def test_shell_unsuported_cmds_fails_on_windows(servicer, set_env_client, mock_shell_pty):
    expected_exit_code = 1 if platform.system() == "Windows" else 0
    res = run_cli_command(["shell"], expected_exit_code=expected_exit_code)

    if expected_exit_code != 0:
        assert re.search("Windows", str(res.exception)), "exception message does not match expected string"


@skip_windows("modal shell is not supported on Windows.")
@pytest.mark.parametrize("container_id", [("sb-abc123"), ("ta-abc123")])
@pytest.mark.parametrize(
    "forbidden_args",
    [
        (["--image", "debian:latest"]),
        (["--cpu", "2"]),
        (["--memory", "1024"]),
        (["--gpu", "a10g"]),
        (["--volume", "my-volume"]),
        (["--secret", "my-secret"]),
        (["--cloud", "aws"]),
        (["--region", "us-east-1"]),
        (["--add-python", "3.11"]),
        (["--env", "main"]),
        (["--add-local", "/tmp/file.txt"]),
        (["-m"]),
    ],
)
def test_shell_forbids_config_args_with_container_id(container_id, forbidden_args):
    run_cli_command(
        ["shell", container_id] + forbidden_args,
        expected_exit_code=1,
        expected_stderr="Cannot specify container configuration arguments",
    )


@skip_windows("modal shell is not supported on Windows.")
@pytest.mark.parametrize("func_ref_args", [["app.py::my_func"], ["-m", "mymodule::my_func"]])
@pytest.mark.parametrize(
    "forbidden_args",
    [
        (["--image", "debian:latest"]),
        (["--cpu", "2"]),
        (["--memory", "1024"]),
        (["--gpu", "a10g"]),
        (["--volume", "my-volume"]),
        (["--secret", "my-secret"]),
        (["--cloud", "aws"]),
        (["--region", "us-east-1"]),
        (["--add-python", "3.11"]),
        (["--add-local", "/tmp/file.txt"]),
    ],
)
def test_shell_forbids_config_args_with_function_ref(
    servicer, set_env_client, mock_shell_routing, func_ref_args, forbidden_args
):
    run_cli_command(
        ["shell"] + func_ref_args + forbidden_args,
        expected_exit_code=1,
        expected_stderr="Cannot specify container configuration arguments",
    )


@skip_windows("modal shell is not supported on Windows.")
@pytest.mark.parametrize(
    "forbidden_args",
    [
        (["--image", "debian:latest"]),
        (["--add-python", "3.11"]),
    ],
)
def test_shell_forbids_image_args_with_image_id(servicer, set_env_client, forbidden_args):
    run_cli_command(
        ["shell", "im-abc123"] + forbidden_args,
        expected_exit_code=1,
        expected_stderr="Cannot specify",
    )


def test_shell_all_shell_command_params_have_explicit_param_decls():
    sig = inspect.signature(shell)
    for param_name, param in sig.parameters.items():
        if param_name in {"ref"}:  # ... all except ref
            continue

        assert param.default.param_decls


@pytest.mark.parametrize(
    "passed_args,expected",
    [
        ({}, []),
        ({"ref": "sb-123", "cmd": "/bin/sh", "pty": True}, []),
        ({"ref": "sb-123", "cpu": 2, "memory": 1024}, ["--cpu", "--memory"]),
    ],
)
def test_passed_forbidden_args_with_shell_function(passed_args, expected):
    param_objs = _params_from_signature(shell)
    result = _passed_forbidden_args(
        param_objs, passed_args, allowed=lambda p: p in {"cmd", "pty", "ref", "use_module_mode"}
    )
    assert result == expected


def test_passed_forbidden_args_with_predicate():
    def dummy_shell(
        image: str = typer.Option(None, "--image"),
        cpu: int = typer.Option(None, "--cpu"),
        memory: int = typer.Option(None, "--memory"),
    ):
        pass

    param_objs = _params_from_signature(dummy_shell)

    _pfa = _passed_forbidden_args
    assert _pfa(param_objs, {"image": "debian:latest", "cpu": 2}, allowed=lambda p: p == "image") == ["--cpu"]
    assert _pfa(param_objs, {"image": "debian:latest", "cpu": 2}, allowed=lambda _: True) == []
    assert _pfa(param_objs, {"image": "debian:latest", "cpu": 2}, allowed=lambda _: False) == ["--image", "--cpu"]
    assert _pfa(param_objs, {"image": None, "cpu": None}, allowed=lambda _: False) == []


@pytest.fixture
def mock_shell_routing(monkeypatch):
    mocks = {}

    mocks["start_running_container"] = Mock()
    mocks["start_from_function_spec"] = Mock()
    mocks["start_from_image"] = Mock()

    monkeypatch.setattr("modal.cli.shell._start_shell_in_running_container", mocks["start_running_container"])
    monkeypatch.setattr("modal.cli.shell._start_shell_from_function_spec", mocks["start_from_function_spec"])
    monkeypatch.setattr("modal.cli.shell._start_shell_from_image", mocks["start_from_image"])

    monkeypatch.setattr("modal.cli.shell._function_spec_from_ref", Mock(return_value=Mock()))
    monkeypatch.setattr("modal.image.Image.from_id", Mock(return_value=Mock()))
    monkeypatch.setattr("modal.image.Image.from_registry", Mock(return_value=Mock()))

    return mocks


@skip_windows("modal shell is not supported on Windows.")
@pytest.mark.parametrize(
    "ref",
    [
        "sb-abc123",
        "ta-abc123",
    ],
)
def test_shell_sb_ta_ref_routes_to_running_container(servicer, set_env_client, mock_shell_routing, ref):
    run_cli_command(["shell", ref])

    mock_shell_routing["start_running_container"].assert_called_once()
    mock_shell_routing["start_from_function_spec"].assert_not_called()
    mock_shell_routing["start_from_image"].assert_not_called()


@skip_windows("modal shell is not supported on Windows.")
@pytest.mark.parametrize(
    "args",
    [
        ["im-abc123"],
        ["--image", "alpine"],
        [],
    ],
)
def test_shell_im_ref_or_param_and_default_routes_to_image_id(servicer, set_env_client, mock_shell_routing, args):
    run_cli_command(["shell"] + args)

    mock_shell_routing["start_running_container"].assert_not_called()
    mock_shell_routing["start_from_function_spec"].assert_not_called()
    mock_shell_routing["start_from_image"].assert_called_once()


@skip_windows("modal shell is not supported on Windows.")
@pytest.mark.parametrize(
    "args",
    [
        ["app.py::my_func"],
        ["my_module.py::foo"],
        ["some/path/script.py::bar"],
        ["-m", "mymodule::my_func"],
    ],
)
def test_shell_function_ref_routes_to_function_spec(servicer, set_env_client, mock_shell_routing, args):
    run_cli_command(["shell"] + args)

    mock_shell_routing["start_running_container"].assert_not_called()
    mock_shell_routing["start_from_function_spec"].assert_called_once()
    mock_shell_routing["start_from_image"].assert_not_called()
