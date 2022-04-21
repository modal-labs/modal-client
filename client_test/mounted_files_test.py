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


script_path = Path("supports") / "mounted_files.py"


def test_function_info_script(test_dir):
    p = subprocess.run([sys.executable, str(script_path)], capture_output=True, cwd=test_dir)
    files = p.stdout.decode("utf-8").splitlines()

    assert len(files) == 1
    assert "mounted_files.py" in files[0]


def test_function_info_package(test_dir):
    p = subprocess.run([sys.executable, "-m", "supports.mounted_files"], cwd=test_dir, capture_output=True)
    files = p.stdout.decode("utf-8").splitlines()

    assert len(files) == 1
    assert "mounted_files.py" in files[0]


@pytest.mark.skipif(platform.system() == "Windows", reason="venvs behave differently on Windows.")
def test_function_info_sys_prefix(test_dir, venv_path):
    # Run with venv activated, so it's on sys.prefix.
    p = subprocess.run(
        [venv_path / "bin" / "python", script_path],
        capture_output=True,
        cwd=test_dir,
    )
    files = p.stdout.decode("utf-8").splitlines()

    assert len(files) == 1
