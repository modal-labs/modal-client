# Copyright Modal Labs 2022
import os
import pathlib
import subprocess
import sys
from typing import Tuple


def _cli(args, server_url, extra_env={}, check=True) -> Tuple[int, str, str]:
    lib_dir = pathlib.Path(__file__).parent.parent
    args = [sys.executable] + args
    env = {
        "MODAL_SERVER_URL": server_url,
        **os.environ,
        "PYTHONUTF8": "1",  # For windows
        **extra_env,
    }
    ret = subprocess.run(args, cwd=lib_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = ret.stdout.decode()
    stderr = ret.stderr.decode()
    if check and ret.returncode != 0:
        raise Exception(f"Failed with {ret.returncode} stdout: {stdout} stderr: {stderr}")
    return ret.returncode, stdout, stderr


def test_run_e2e(servicer):
    _cli(["-m", "modal_test_support.script"], servicer.remote_addr)


def test_run_profiler(servicer):
    _cli(["-m", "cProfile", "-m", "modal_test_support.script"], servicer.remote_addr)


def test_run_unconsumed_map(servicer):
    _, _, err = _cli(["-m", "modal_test_support.unconsumed_map"], servicer.remote_addr)
    assert "map" in err
    assert "for-loop" in err

    _, _, err = _cli(["-m", "modal_test_support.consumed_map"], servicer.remote_addr)
    assert "map" not in err
    assert "for-loop" not in err


def test_auth_failure_last_line(servicer):
    returncode, _, err = _cli(
        ["-m", "modal_test_support.script"],
        servicer.remote_addr,
        extra_env={"MODAL_TOKEN_ID": "bad", "MODAL_TOKEN_SECRET": "bad"},
        check=False,
    )
    assert returncode != 0
    assert "bad bad bad" in err.strip().split("\n")[-1]  # err msg should be on the last line
