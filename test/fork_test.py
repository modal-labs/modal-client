# Copyright Modal Labs 2024
import subprocess
import sys
from test.supports.skip import skip_windows


@skip_windows("fork not supported on windows")
def test_process_fork(supports_dir, server_url_env):
    output = subprocess.check_output([sys.executable, supports_dir / "forking.py"], timeout=2, encoding="utf8")

    success_pids = set(int(x) for x in output.split())
    assert len(success_pids) == 2
