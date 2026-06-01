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
    # 4 s is enough on a fast Linux runner but consistently times out on
    # GHA-hosted Intel macOS where forked Python startup can take 6–8 s.
    output = subprocess.check_output(
        [sys.executable, supports_dir / "forking.py", test_case], timeout=30, encoding="utf8"
    )

    success_pids = {int(x) for x in output.split()}
    assert len(success_pids) == 2
