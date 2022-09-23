import os
import pathlib
import subprocess
import sys


def _cli(args, server_url):
    lib_dir = pathlib.Path(__file__).parent.parent
    args = [sys.executable] + args
    env = {
        "MODAL_SERVER_URL": server_url,
        **os.environ,
        "PYTHONUTF8": "1",  # For windows
    }
    ret = subprocess.run(args, cwd=lib_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        raise Exception(f"Failed with {ret.returncode} stdout: {ret.stdout} stderr: {ret.stderr}")
    return ret.stdout, ret.stderr


def test_run_e2e(servicer):
    _cli(["-m", "modal_test_support.script"], servicer.remote_addr)


def test_run_profiler(servicer):
    _cli(["-m", "cProfile", "-m", "modal_test_support.script"], servicer.remote_addr)


def test_run_unconsumed_map(servicer):
    _, err = _cli(["-m", "modal_test_support.unconsumed_map"], servicer.remote_addr)
    assert b"map" in err
    assert b"for-loop" in err

    _, err = _cli(["-m", "modal_test_support.consumed_map"], servicer.remote_addr)
    assert b"map" not in err
    assert b"for-loop" not in err
