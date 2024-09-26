import subprocess
import sys


def test_process_fork(supports_dir, server_url_env):
    output = subprocess.check_output([sys.executable, supports_dir / "forking.py"], timeout=2, encoding="utf8")

    success_pids = set(int(x) for x in output.split())
    assert len(success_pids) == 2
