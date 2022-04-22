import os
import platform
import pytest
import shutil
import subprocess
import sys
from pathlib import Path


@pytest.fixture(scope="module")
def test_dir(request):
    """Absolute path to directory containing test file."""
    root_dir = Path(request.config.rootdir)
    test_dir = Path(os.getenv("PYTEST_CURRENT_TEST")).parent
    return root_dir / test_dir


@pytest.fixture(scope="function")
def venv_path(test_dir):
    venv_path = test_dir / "supports" / "venv"
    subprocess.run([sys.executable, "-m", "venv", venv_path, "--copies", "--system-site-packages"])
    # Install a tiny package in the venv.
    subprocess.run([venv_path / "bin" / "python", "-m", "pip", "install", "--force-reinstall", "six"])
    yield venv_path
    shutil.rmtree(venv_path)


script_path = Path("supports") / "script.py"


def test_mounted_files_script(test_dir):
    p = subprocess.run([sys.executable, str(script_path)], capture_output=True, cwd=test_dir)
    stdout = p.stdout.decode("utf-8")
    stderr = p.stderr.decode("utf-8")
    print("stdout: ", stdout)
    print("stderr: ", stderr)
    files = stdout.splitlines()

    assert len(files) == 4
    assert any(["a.py" in f for f in files])
    assert any(["c.py" in f for f in files])
    assert not any(["d.py" in f for f in files])
    assert any(["e.py" in f for f in files])
    assert any(["script.py" in f for f in files])


def test_mounted_files_package(test_dir):
    p = subprocess.run([sys.executable, "-m", "supports.package"], cwd=test_dir, capture_output=True)
    stdout = p.stdout.decode("utf-8")
    stderr = p.stderr.decode("utf-8")
    print("stdout: ", stdout)
    print("stderr: ", stderr)
    files = stdout.splitlines()

    assert len(files) == 6
    assert any(["a.py" in f for f in files])
    assert any(["c.py" in f for f in files])
    assert any(["d.py" in f for f in files])
    assert any(["e.py" in f for f in files])
    assert any(["script.py" in f for f in files])
    assert any(["package.py" in f for f in files])


@pytest.mark.skipif(platform.system() == "Windows", reason="venvs behave differently on Windows.")
def test_mounted_files_sys_prefix(test_dir, venv_path):
    # Run with venv activated, so it's on sys.prefix.
    p = subprocess.run(
        [venv_path / "bin" / "python", script_path],
        capture_output=True,
        cwd=test_dir,
    )
    stdout = p.stdout.decode("utf-8")
    stderr = p.stderr.decode("utf-8")
    print("stdout: ", stdout)
    print("stderr: ", stderr)
    files = stdout.splitlines()

    assert len(files) == 4
    assert any(["a.py" in f for f in files])
    assert any(["c.py" in f for f in files])
    assert not any(["d.py" in f for f in files])
    assert any(["e.py" in f for f in files])
    assert any(["script.py" in f for f in files])
