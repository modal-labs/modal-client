# Copyright Modal Labs 2022-2023
import platform
import pytest
import re
import subprocess
from pathlib import Path

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
    from .conftest import run_cli_command

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
    from .conftest import run_cli_command

    app_file = test_dir / "supports" / "app_run_tests" / "default_app.py"
    _, captured_out = mock_shell_pty
    shell_prompt = servicer.shell_prompt
    run_cli_command(["shell", "--cmd", "pwd", app_file.as_posix() + "::foo"])
    expected_output = subprocess.run(["pwd"], capture_output=True, check=True).stdout
    assert captured_out == [(1, shell_prompt), (1, expected_output)]


@skip_windows("modal shell is not supported on Windows.")
def test_shell_preserve_token(servicer, set_env_client, mock_shell_pty, monkeypatch):
    from .conftest import run_cli_command

    monkeypatch.setenv("MODAL_TOKEN_ID", "my-token-id")

    fake_stdin, captured_out = mock_shell_pty
    shell_prompt = servicer.shell_prompt

    fake_stdin.clear()
    fake_stdin.extend([b'echo "$MODAL_TOKEN_ID"\n', b"exit\n"])
    run_cli_command(["shell"])

    expected_output = b"my-token-id\n"
    assert captured_out == [(1, shell_prompt), (1, expected_output)]


def test_shell_unsuported_cmds_fails_on_windows(servicer, set_env_client, mock_shell_pty):
    from .conftest import run_cli_command

    expected_exit_code = 1 if platform.system() == "Windows" else 0
    res = run_cli_command(["shell"], expected_exit_code=expected_exit_code)

    if expected_exit_code != 0:
        assert re.search("Windows", str(res.exception)), "exception message does not match expected string"
