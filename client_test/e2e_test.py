import os
import pathlib
import subprocess
import sys


def _cli(module, server_url):
    lib_dir = pathlib.Path(__file__).parent.parent
    args = [sys.executable, "-m", module]
    env = {
        **os.environ,
        "MODAL_SERVER_URL": server_url,
        # For windows
        "PYTHONUTF8": "1",
    }
    ret = subprocess.run(args, cwd=lib_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        raise Exception(f"Failed with {ret.returncode} stdout: {ret.stdout} stderr: {ret.stderr}")
    return ret.stdout


def test_run_e2e(servicer):
    _cli("modal_test_support.script", servicer.remote_addr)
