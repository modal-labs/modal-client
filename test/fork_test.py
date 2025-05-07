# Copyright Modal Labs 2024
import pytest
import subprocess
import sys

from test.supports.skip import skip_windows


@skip_windows("fork not supported on windows")
@pytest.mark.parametrize(
    "test_case", ["test_stub_method", "test_stub_reference", "test_default_stub", "test_default_stub_reference"]
)
def test_process_fork(supports_dir, server_url_env, token_env, test_case):
    output = subprocess.check_output(
        [sys.executable, supports_dir / "forking.py", test_case], timeout=2, encoding="utf8"
    )

    success_pids = {int(x) for x in output.split()}
    assert len(success_pids) == 2
